# ml_engine.py
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import config

log = logging.getLogger("vpa.ml")


class MLEngine:
    """
    Production inference engine for ML-driven Vertical Pod Autoscaler.
    """

    def __init__(self) -> None:
        self._device = torch.device("cpu")
        log.info("Initializing ML Engine with CPU-only execution")

        self._validate_config()

        self.model_path: Path = config.MODEL_PATH
        self.feature_scaler_path: Path = config.FEATURE_SCALER_PATH
        self.target_scaler_path: Path = config.TARGET_SCALER_PATH

        self.window_size: int = config.WINDOW_SIZE
        self.feature_count: int = config.FEATURE_COUNT
        self.feature_order: Tuple[str, ...] = tuple(config.FEATURE_ORDER)

        self.cpu_bounds = config.CPU_BOUNDS
        self.memory_bounds = config.MEMORY_BOUNDS
        self.safety_buffer: float = config.SAFETY_BUFFER

        log.info("Expected feature order: %s", self.feature_order)

        self.feature_scaler = self._load_feature_scaler()
        self.target_scaler = self._load_target_scaler()
        self.model = self._load_model()

        self._validate_model_compatibility()

        self.model.eval()

        log.info(
            "ML Engine loaded successfully "
            "(model=%s, window=%d, features=%d)",
            self.model_path.name,
            self.window_size,
            self.feature_count,
        )

    def _validate_config(self) -> None:
        required_attrs = [
            "MODEL_PATH",
            "FEATURE_SCALER_PATH",
            "TARGET_SCALER_PATH",
            "WINDOW_SIZE",
            "FEATURE_COUNT",
            "FEATURE_ORDER",
            "CPU_BOUNDS",
            "MEMORY_BOUNDS",
            "SAFETY_BUFFER",
        ]
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise RuntimeError(f"Missing required config attribute: {attr}")

        if len(config.FEATURE_ORDER) != config.FEATURE_COUNT:
            raise RuntimeError(
                "FEATURE_ORDER length does not match FEATURE_COUNT"
            )

    def _load_feature_scaler(self):
        scaler = joblib.load(self.feature_scaler_path)

        if not hasattr(scaler, "n_features_in_"):
            raise RuntimeError("Feature scaler missing n_features_in_ attribute")

        if scaler.n_features_in_ != self.feature_count:
            raise RuntimeError(
                f"Feature scaler expects {scaler.n_features_in_} features, "
                f"but config specifies {self.feature_count}"
            )

        log.info("Feature scaler loaded successfully")
        return scaler

    def _load_target_scaler(self):
        scaler = joblib.load(self.target_scaler_path)

        if not hasattr(scaler, "n_features_in_"):
            raise RuntimeError("Target scaler missing n_features_in_ attribute")

        if scaler.n_features_in_ != 2:
            raise RuntimeError(
                f"Target scaler expects {scaler.n_features_in_} outputs, expected 2"
            )

        log.info("Target scaler loaded successfully")
        return scaler



    def _load_model(self) -> torch.jit.ScriptModule:
        """
        Loads a standalone TorchScript model.
        No longer requires train_gru.py!
        """
        try:
            # Point to the new independent file
            path = config.MODEL_DIR / "k8s_gru_independent.pt"

            # Load using JIT (the standalone loader)
            model = torch.jit.load(path, map_location=self._device)

            log.info("Standalone TorchScript model loaded successfully.")
            return model

        except Exception as exc:
            log.error(f"Failed to load standalone model: {exc}")
            raise RuntimeError(f"Standalone model missing at {path}")


    def _validate_model_compatibility(self) -> None:
        dummy_input = torch.zeros(
            (1, self.window_size, self.feature_count),
            dtype=torch.float32,
            device=self._device,
        )

        with torch.no_grad():
            output = self.model(dummy_input)

        if isinstance(output, tuple):
            output = output[0]

        if output.device.type != "cpu":
            raise RuntimeError("Model output is not on CPU")

        if output.ndim != 2:
            raise RuntimeError(
                f"Model output must be 2D, got shape {tuple(output.shape)}"
            )

        if output.shape[1] != 2:
            raise RuntimeError(
                f"Model output dimension {output.shape[1]} does not match expected 2"
            )

    def predict(self, sequence: np.ndarray) -> Tuple[float, int]:
        if not isinstance(sequence, np.ndarray):
            raise ValueError("Input sequence must be a numpy array")

        expected_shape = (self.window_size, self.feature_count)
        if sequence.shape != expected_shape:
            raise ValueError(
                f"Invalid input shape {sequence.shape}, expected {expected_shape}"
            )

        scaled = self.feature_scaler.transform(sequence)
        tensor = torch.as_tensor(
            scaled, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)

        if isinstance(output, tuple):
            output = output[0]

        if output.device.type != "cpu":
            raise RuntimeError("Inference output is not on CPU")

        real = self.target_scaler.inverse_transform(output.numpy())[0]

        cpu = float(real[0]) * self.safety_buffer
        mem = float(real[1]) * self.safety_buffer

        cpu = max(self.cpu_bounds.minimum, min(cpu, self.cpu_bounds.maximum))
        cpu = max(cpu, self.cpu_bounds.floor)

        mem = max(self.memory_bounds.minimum, min(mem, self.memory_bounds.maximum))
        mem = max(mem, self.memory_bounds.floor)

        cpu = float(cpu)
        mem = int(mem)

        if cpu <= 0 or mem <= 0:
            raise RuntimeError("Sanitized prediction resulted in non-positive resources")

        return cpu, mem



    def reload_model(self, fd: int) -> None:
        """
        Loads the model directly from the verified file descriptor.
        This ensures we are loading the EXACT bits that were just hashed and verified.
        
        Note: We renamed this to 'reload_model' but used the FD logic to keep 
        compatibility with the main orchestrator's naming.
        """
        import torch
        import io
        import os

        try:
            # 1. Seek to start of the verified file descriptor
            os.lseek(fd, 0, os.SEEK_SET)
            
            # 2. Read the verified bits into memory
            model_bytes = os.read(fd, os.fstat(fd).st_size)
            buffer = io.BytesIO(model_bytes)
            
            # 3. Load into temporary variable for validation
            new_model = torch.jit.load(buffer, map_location=self._device)
            new_model.eval()

            # 4. Safety: Compatibility check (20-feature matrix)
            dummy_input = torch.zeros(
                (1, self.window_size, self.feature_count),
                dtype=torch.float32,
                device=self._device,
            )
            with torch.no_grad():
                new_model(dummy_input)

            # 5. ATOMIC SWAP
            self.model = new_model
            log.info("ML Engine: Atomic swap complete via verified FD.")

        except Exception as exc:
            log.error("ML Engine: Reload from FD failed: %s", exc)
