# state_manager.py
"""
Persistent state management for the ML-driven Kubernetes VPA controller.

Responsibilities:
- Maintain controller state safely across restarts
- Enforce schema compatibility
- Ensure atomic, crash-safe persistence
- Provide thread-safe access and updates
- Use strictly UTC, timezone-aware timestamps only

This module is intentionally defensive and boring.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import config

# Module-scoped logger only; no global logging configuration here
log = logging.getLogger("vpa.state")

STATE_SCHEMA_VERSION = 1
MAX_CONSECUTIVE_ERRORS = 100  # Hard cap to prevent unbounded growth


@dataclass(frozen=True)
class TargetState:
    """State for an individual workload."""
    last_cpu_request: float
    last_mem_request: int
    last_patch_timestamp: datetime

@dataclass(frozen=True)
class ControllerState:
    """Global state containing all discovered targets."""
    version: int
    # Key is the workload name (e.g., 'nginx-deployment')
    targets: Dict[str, TargetState] 
    consecutive_errors: int


def _validate_utc_datetime(value: Any) -> datetime:
    """
    Validate and normalize a timestamp to UTC.

    - Accepts ISO-8601 strings or datetime objects
    - Rejects naive datetimes
    - Always returns a UTC-aware datetime
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value)

    if not isinstance(value, datetime):
        raise ValueError("Invalid timestamp type")

    if value.tzinfo is None:
        raise ValueError("Naive datetime is not allowed")

    return value.astimezone(timezone.utc)


def _default_state() -> ControllerState:
    """Return a clean default controller state with an empty target map."""
    return ControllerState(
        version=STATE_SCHEMA_VERSION,
        targets={},
        consecutive_errors=0,
    )


class StateManager:
    """
    Thread-safe manager for persistent controller state.
    """

    def __init__(self):
        # Configuration ownership: the path comes exclusively from config.py
        self._file_path: Path = config.STATE_FILE_PATH

        # Defensive validation of configuration invariants
        if not self._file_path.is_absolute():
            raise ValueError("STATE_FILE_PATH must be absolute")

        if not self._file_path.parent.exists():
            raise FileNotFoundError(
                "Parent directory of STATE_FILE_PATH does not exist"
            )

        self._lock = threading.Lock()
        self._state: ControllerState = self._load_state()
    
    def _load_state(self) -> ControllerState:
        if not self._file_path.exists():
            return _default_state()

        try:
            with open(self._file_path, "r") as f:
                data = json.load(f)
            
            if data.get("version") != STATE_SCHEMA_VERSION:
                log.warning("Schema mismatch; resetting state")
                return _default_state()

            # Reconstruct the nested dictionary of TargetState objects
            raw_targets = data.get("targets", {})
            reconstructed_targets = {}
            
            for name, t_data in raw_targets.items():
                reconstructed_targets[name] = TargetState(
                    last_cpu_request=float(t_data["last_cpu_request"]),
                    last_mem_request=int(t_data["last_mem_request"]),
                    last_patch_timestamp=_validate_utc_datetime(t_data["last_patch_timestamp"])
                )

            return ControllerState(
                version=STATE_SCHEMA_VERSION,
                targets=reconstructed_targets,
                consecutive_errors=int(data.get("consecutive_errors", 0))
            )
        except Exception as e:
            log.error(f"Failed to load state: {e}. Falling back to default.")
            return _default_state()


    def _persist_locked(self) -> None:
        """Persist state atomically with JSON-safe timestamps."""
        # Create a dictionary copy and manually serialize datetimes
        dict_state = asdict(self._state)
        for name in dict_state["targets"]:
            ts = self._state.targets[name].last_patch_timestamp
            dict_state["targets"][name]["last_patch_timestamp"] = ts.isoformat()

        tmp_file = NamedTemporaryFile(
            mode="w", 
            dir=self._file_path.parent, 
            delete=False, 
            suffix=".tmp"
        )
        try:
            json.dump(dict_state, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_file.close()
            os.replace(tmp_file.name, self._file_path)
        except Exception:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
            raise
           
    
    @property
    def current(self) -> ControllerState:
        """
        Return a read-only snapshot of the current state.
        """
        with self._lock:
            return self._state
    def get_state(self, target_name: str) -> Dict[str, Any]:
        """
        Return state for a specific target. 
        Used by the PolicyEngine to check cooldowns.
        """
        with self._lock:
            target_data = self._state.targets.get(target_name)
            if not target_data:
                # Return a 'zero' timestamp so the first scale always proceeds
                return {"last_patch_ts": 0.0}
            
            return {
                "last_patch_ts": target_data.last_patch_timestamp.timestamp(),
                "cpu": target_data.last_cpu_request,
                "mem": target_data.last_mem_request
            }

    def update_state(self, target_name: str, updates: Dict[str, Any]) -> None:
        """
        Record a successful patch for a specific target.
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            # Create a copy of the target map to maintain immutability
            new_targets = dict(self._state.targets)
            new_targets[target_name] = TargetState(
                last_cpu_request=updates.get("cpu", 0.0),
                last_mem_request=updates.get("mem", 0),
                last_patch_timestamp=now
            )

            self._state = ControllerState(
                version=STATE_SCHEMA_VERSION,
                targets=new_targets,
                consecutive_errors=0
            )
            self._persist_locked()

        log.info("State updated for %s: CPU=%s MEM=%s", target_name, updates.get("cpu"), updates.get("mem"))

    def increment_error(self) -> None:
        """Global error counter for the controller health."""
        with self._lock:
            new_count = min(self._state.consecutive_errors + 1, MAX_CONSECUTIVE_ERRORS)
            self._state = ControllerState(
                version=self._state.version,
                targets=self._state.targets,
                consecutive_errors=new_count
            )
            self._persist_locked() 
    

