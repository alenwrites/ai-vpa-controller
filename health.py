# health.py
"""
Kubernetes health sidecar running in a separate OS process.

Key guarantees:
- Survives GIL starvation, CPU-bound loops, C-extension deadlocks
- Exits promptly and cleanly on SIGTERM / SIGINT
"""

import time
import signal
import logging
import multiprocessing as mp
from dataclasses import dataclass

from fastapi import FastAPI, Response, status
import uvicorn


@dataclass(frozen=True)
class HealthConfig:
    port: int
    heartbeat_interval_sec: float
    timeout_multiplier: float = 3.0


class HealthIPC:
    """
    Explicit, process-safe shared state.
    """
    def __init__(self) -> None:
        self.last_heartbeat_ts = mp.Value("d", 0.0)
        self.process_start_ts = mp.Value("d", time.time())
        self.ready_event = mp.Event()


def _create_app(cfg: HealthConfig, ipc: HealthIPC) -> FastAPI:
    app = FastAPI(title="Health Sidecar")

    heartbeat_timeout = cfg.heartbeat_interval_sec * cfg.timeout_multiplier

    @app.get("/healthz")
    def liveness():
        now = time.time()
        with ipc.last_heartbeat_ts.get_lock():
            last = ipc.last_heartbeat_ts.value

        # Startup grace: avoid crash loops before first heartbeat
        if last == 0.0:
            return {
                "status": "starting",
                "seconds_since_last_heartbeat": None,
                "heartbeat_timeout_seconds": heartbeat_timeout,
                "uptime_seconds": now - ipc.process_start_ts.value,
            }

        age = now - last
        if age > heartbeat_timeout:
            return Response(
                content=f"Heartbeat stale: {age:.1f}s > {heartbeat_timeout:.1f}s",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return {
            "status": "alive",
            "seconds_since_last_heartbeat": age,
            "heartbeat_timeout_seconds": heartbeat_timeout,
            "uptime_seconds": now - ipc.process_start_ts.value,
        }

    @app.get("/readyz")
    def readiness():
        if not ipc.ready_event.is_set():
            return Response(
                content="Not ready",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return {"status": "ready"}

    return app


def _run_server(cfg: HealthConfig, ipc: HealthIPC) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    app = _create_app(cfg, ipc)

    server = uvicorn.Server(
        uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=cfg.port,
            log_level="error",
            lifespan="off",
        )
    )

    def _handle_signal(signum, _frame):
        # Required: tell uvicorn to exit; relying on Process termination is unsafe
        server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    server.run()


def start_health_process(
    *,
    port: int,
    heartbeat_interval_sec: float,
) -> tuple[mp.Process, HealthIPC]:
    """
    Starts the health server in a dedicated OS process.

    IMPORTANT:
    - Main loop must call record_heartbeat() at least once per
      heartbeat_interval_sec to avoid false liveness failures.
    """
    if heartbeat_interval_sec <= 0:
        raise ValueError("heartbeat_interval_sec must be > 0")

    cfg = HealthConfig(
        port=port,
        heartbeat_interval_sec=heartbeat_interval_sec,
    )
    ipc = HealthIPC()

    proc = mp.Process(
        target=_run_server,
        name="health-sidecar",
        args=(cfg, ipc),
    )
    proc.start()

    return proc, ipc


# ---- Integration helpers (used by main.py) ----

def record_heartbeat(ipc: HealthIPC) -> None:
    """
    Must be called at least once per heartbeat_interval_sec.
    Slower cadences will trigger liveness failure by design.
    """
    with ipc.last_heartbeat_ts.get_lock():
        ipc.last_heartbeat_ts.value = time.time()


def set_ready(ipc: HealthIPC) -> None:
    ipc.ready_event.set()


def revoke_ready(ipc: HealthIPC) -> None:
    ipc.ready_event.clear()
