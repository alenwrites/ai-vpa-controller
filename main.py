# main.py

from leader_election import LeaderElection
from discovery import DiscoveryEngine
from policy_engine import PolicyEngine
from state_manager import StateManager  # Assuming you have this file built
from kubernetes.client.exceptions import ApiException
import time
import signal
import logging
import random
import threading
from typing import Dict, Any

import config
from errors import TransientError
from prometheus_client import PrometheusClient
from ml_engine import MLEngine
from safety_layer import SafetyLayer
from k8s_adapter import KubernetesAdapter
from event_logger import EventLogger
from lifecycle_manager import ModelLifecycleManager

from metrics_exporter import (
    start_metrics_server,
    observe_ai_suggested,
    observe_safety_approved,
    count_decision,
    RECONCILIATION_DURATION_SECONDS
)

# Health sidecar (MUST be imported at top level)
from health import start_health_process, record_heartbeat, set_ready


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("vpa.orchestrator")


class GracefulShutdown:
    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:
        log.info("Shutdown signal received (%s). Exiting cleanly.", signum)
        self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def sleep(self, seconds: float) -> None:
        self._stop_event.wait(seconds)


class Backoff:
    """
    Exponential backoff with jitter.

    Backoff delays intentionally extend the cycle duration and may skip
    scheduled ticks. Missed ticks are NOT replayed.
    """

    def __init__(self, *, base: float, factor: float, max_delay: float) -> None:
        self._base = base
        self._factor = factor
        self._max = max_delay
        self._attempts = 0

    def reset(self) -> None:
        self._attempts = 0

    def next_delay(self) -> float:
        delay = min(self._base * (self._factor ** self._attempts), self._max)
        self._attempts += 1
        return delay * random.uniform(0.5, 1.5)


def _validate_prometheus_metrics(
    metrics: Dict[str, Any],
    *,
    allow_zero_rss: bool,
) -> bool:
    required = {
        "rss_mib": (int, float),
        "limit_mib": (int, float),
    }

    for key, types in required.items():
        if key not in metrics:
            raise KeyError(f"Missing required Prometheus metric: {key}")

        value = metrics[key]
        if not isinstance(value, types):
            raise TypeError(f"Metric {key} has invalid type: {type(value)}")

        if value < 0:
            raise ValueError(f"Metric {key} must be non-negative, got {value}")

    if metrics["limit_mib"] == 0:
        raise ValueError("Prometheus limit_mib must be > 0")

    if metrics["rss_mib"] == 0:
        if allow_zero_rss:
            log.warning(
                "rss_mib is zero; Redis safety checks bypassed for this cycle "
                "(policy allow_zero_rss=true)"
            )
            return True
        raise ValueError(
            "rss_mib is zero and rejected by policy (allow_zero_rss=false)"
        )

    return False


def _sleep_until_next_tick(
    start_ts: float,
    interval: int,
    shutdown: GracefulShutdown,
) -> None:
    next_tick = start_ts + interval
    shutdown.sleep(max(0.0, next_tick - time.monotonic()))


def run_vpa_controller() -> None:
    log.info("ML-driven VPA controller starting")

    # --- INTEGRATION START ---
    # Start the Prometheus exporter on port 8000 (standard port for exporters)
    start_metrics_server(port=8000)
    # --- INTEGRATION END ---

    # ------------------------------------------------------------------
    # Health-first startup (CRITICAL: fork before heavy libraries)
    # ------------------------------------------------------------------
    try:
        health_proc, health_ipc = start_health_process(
            port=8080,
            heartbeat_interval_sec=config.POLL_INTERVAL,
        )
        log.info("Health sidecar process started.")
    except Exception as exc:
        log.critical("Failed to start health sidecar", exc_info=True)
        raise


    try:
        # 1. Initializing the Core Components (The Organs)
        eyes = PrometheusClient(timeout_seconds=5)
        brain = MLEngine()

        # --- CONSOLIDATED ML LIFECYCLE INTEGRATION ---
        # Wrapper to match the safety requirement of Point #4
        def safety_check_wrapper():
            # This calls the internal validation logic in your ml_engine.py
            brain._validate_model_compatibility()

        # Initialize the hardened Lifecycle Manager ONCE
        # This matches your lifecycle_manager.py's specific constructor
        model_guard = ModelLifecycleManager(
            model_root=config.MODEL_DIR,
            public_key_pem=config.PUBLIC_KEY_PEM, 
            on_load_sync=brain.reload_model,      # Connects verified FD to Brain
            post_load_check=safety_check_wrapper  # Connects Safety check
        )

        # Start the background monitoring thread
        def watch_loop():
            log.info("Model watcher thread started.")
            while not shutdown.should_stop():
                try:
                    model_guard.check_for_updates()
                except Exception as e:
                    log.error(f"Model watcher encountered an error: {e}")
                shutdown.sleep(5) # Use graceful sleep

        model_thread = threading.Thread(target=watch_loop, daemon=True)
        model_thread.start()
        # ---------------------------------------------

        firewall = SafetyLayer()
        muscles = KubernetesAdapter()

        # 2. Initializing the Governance Layers (The Guards)
        # These ensure HA, Discovery of multiple apps, and Policy enforcement
        leader_guard = LeaderElection(config.LEASE_NAME, config.K8S_NAMESPACE)
        discoverer = DiscoveryEngine()
        policy_engine = PolicyEngine()
        state_manager = StateManager(config.STATE_FILE_PATH)
        voice = EventLogger()

        # 3. Signal readiness to Kubernetes
        # At this point, Liveness/Readiness probes will start passing
        set_ready(health_ipc)
        log.info("VPA controller fully initialized and ready for reconciliation.")

    except Exception:
        log.critical("Initialization failure", exc_info=True)
        health_proc.terminate()
        health_proc.join()
        raise

    shutdown = GracefulShutdown()
    shutdown.install()

    backoff = Backoff(base=1.0, factor=2.0, max_delay=30.0)

    poll_interval = config.POLL_INTERVAL
    allow_zero_rss = getattr(config, "ALLOW_ZERO_RSS", False)



    try:
        while not shutdown.should_stop():
            cycle_start = time.monotonic()

            # 1. Heartbeat: Tell the health sidecar we are still alive
            record_heartbeat(health_ipc)

            # --- INSERT THIS CHECK HERE ---
            if not model_guard.is_alive():
                log.error("Model Lifecycle Manager thread died. Attempting restart...")
                model_guard.start()
            # ------------------------------


            # 2. HA Guard: Only the Leader performs discovery and scaling
            if not leader_guard.check_leadership():
                log.info("Standing by: Instance is not the leader.")
                _sleep_until_next_tick(cycle_start, poll_interval, shutdown)
                continue

            # 3. Discovery: Find all workloads authorized for AI scaling
            discovery = discoverer.get_authorized_workloads(config.K8S_NAMESPACE)
            if discovery.unexpected_failure:
                log.error("Discovery failed globally. Entering safety backoff.")
                shutdown.sleep(backoff.next_delay())
                continue

            # 4. Reconciliation: Process each target discovered
            # We wrap the whole loop in a timer for Prometheus
            with RECONCILIATION_DURATION_SECONDS.time():
                for target in discovery.targets:
                    try:
                        # A. Data Ingestion (The Eyes)
                        matrix = eyes.get_model_input(target.name, target.namespace)
                        current = eyes.get_current_metrics(target.name, target.namespace)

                        _validate_prometheus_metrics(current, allow_zero_rss=allow_zero_rss)

                        # B. ML Inference (The Brain)
                        raw_cpu, raw_mem = brain.predict(matrix)
                        observe_ai_suggested("cpu", raw_cpu)
                        observe_ai_suggested("memory", raw_mem)

                        # C. Validation & Policy (The Brakes)
                        # We check the Policy Engine before touching the cluster
                        state = state_manager.get_state(target.name)
                        decision = policy_engine.is_resize_allowed(
                            target_name=target.name,
                            last_update_ts=state.get("last_patch_ts", 0),
                            current_cpu=current.get("cpu_request_cores", 0), # Simplified for example
                            current_mem=current["limit_mib"],
                            proposed_cpu=raw_cpu,
                            proposed_mem=raw_mem
                        )

                        if not decision.allowed:
                            log.info(f"Policy blocked {target.name}: {decision.reason}")
                            count_decision("reject")

                            # Emit K8s Event for the block
                            voice.emit(
                                target_name=target.name,
                                target_kind=target.kind,
                                namespace=target.namespace,
                                api_version="apps/v1",
                                action=EventLogger.ACTION_POLICY_BLOCKED,
                                reason=EventLogger.REASON_POLICY_EVALUATION,
                                message=decision.reason,
                                is_warning=False
                            )
                            continue

                        # D. Safety Layer (The Firewall)
                        safe_cpu, safe_mem = firewall.validate_prediction(
                            pred_cpu=raw_cpu,
                            pred_mem=raw_mem,
                            current_mem_usage_mib=current["rss_mib"],
                            current_mem_limit_mib=current["limit_mib"],
                        )

                        # E. Actuation (The Muscles)
                        muscles.patch_resources(
                            name=target.name,
                            namespace=target.namespace,
                            kind=target.kind,
                            cpu_cores=safe_cpu,
                            mem_mib=safe_mem,
                        )

                        # F. Telemetry & Audit Trail (The Voice)
                        observe_safety_approved("cpu", safe_cpu)
                        observe_safety_approved("memory", safe_mem)

                        is_clamped = (safe_cpu != raw_cpu or safe_mem != raw_mem)
                        count_decision("allow" if not is_clamped else "clamp")

                        # This is the critical missing piece:
                        voice.emit(
                            target_name=target.name,
                            target_kind=target.kind,
                            namespace=target.namespace,
                            api_version="apps/v1",
                            action=EventLogger.ACTION_SAFETY_OVERRIDE_APPLIED if is_clamped else EventLogger.ACTION_SCALING_RECOMMENDATION_APPLIED,
                            reason=EventLogger.REASON_SAFETY_GUARDRAIL if is_clamped else EventLogger.REASON_RESOURCE_PRESSURE,
                            message=f"Applied: CPU {safe_cpu}, Mem {safe_mem} (AI wanted {raw_cpu}/{raw_mem})",
                            is_warning=is_clamped
                        )

                        # G. State Update
                        state_manager.update_state(target.name, {"cpu": safe_cpu, "mem": safe_mem})
                        backoff.reset()

            # 5. Throttle: Maintain the poll interval
            _sleep_until_next_tick(cycle_start, poll_interval, shutdown)




    finally:
        log.info("VPA controller shutting down. Cleaning up processes.")
        health_proc.terminate()
        health_proc.join()

    log.info("VPA controller shutdown complete")


if __name__ == "__main__":
    run_vpa_controller()
