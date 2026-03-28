"""
config.py

Battle-hardened configuration for the Kubernetes ML-driven VPA controller.

Design principles:
- Fail fast at import time
- Defensive environment parsing
- Absolute, container-safe paths
- Explicit resource unit safety
- No cluster introspection from config code
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Callable, TypeVar


# Capture the unique Pod Name from the Downward API
# Fallback to 'local-dev' if running outside of Kubernetes
POD_NAME = os.getenv("POD_NAME", "local-dev")

# ------------------------------------------------------------------------------
# Defensive Environment Parsing
# ------------------------------------------------------------------------------

T = TypeVar("T")


def read_env(name: str, default: T, cast: Callable[[str], T]) -> T:
    """
    Read and validate an environment variable.

    Raises:
        RuntimeError if value is malformed or cannot be cast.
    """
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    try:
        return cast(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Invalid value for environment variable '{name}': "
            f"expected {cast.__name__}, got '{raw}'"
        ) from exc


def read_int_env(name: str, default: int) -> int:
    """
    Strict integer parser for unit-safe memory values (MiB only).
    """
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    if not raw.isdigit():
        raise RuntimeError(
            f"Invalid value for environment variable '{name}': "
            f"expected integer MiB, got '{raw}'"
        )

    return int(raw)


def read_path_env(name: str, default: Path) -> Path:
    """
    Read a filesystem path safely.

    Empty strings never resolve to '.'.
    Always returns an absolute, resolved path.
    """
    raw = os.getenv(name)

    path = default if raw is None or raw.strip() == "" else Path(raw)
    return path.resolve()


# ------------------------------------------------------------------------------
# Infrastructure Context
# ------------------------------------------------------------------------------

COOLDOWN_SECONDS: int = read_env("COOLDOWN_SECONDS", 600, int)
MIN_CPU_CHANGE_THRESHOLD: float = read_env("MIN_CPU_CHANGE_THRESHOLD", 0.1, float)
MIN_CPU_CHANGE_ABSOLUTE: float = read_env("MIN_CPU_CHANGE_ABSOLUTE", 0.05, float)
MIN_MEM_CHANGE_MiB: int = read_int_env("MIN_MEM_CHANGE_MiB", 64)

# Maintenance Windows: List of (StartHour, EndHour) in UTC
# Default: Freeze resizes between 02:00 and 03:00 UTC
MAINTENANCE_WINDOWS_UTC: list[tuple[int, int]] = [(2, 3)]


K8S_NAMESPACE: str = read_env("K8S_NAMESPACE", "default", str)


# ------------------------------------------------------------------------------
# High Availability (Leader Election)
# ------------------------------------------------------------------------------

LEASE_NAME: str = read_env("LEASE_NAME", "ai-vpa-leader-lease", str)
# LEADER_CONFIRM_INTERVAL is used by leader_election.py
LEADER_CONFIRM_INTERVAL: int = read_env("LEADER_CONFIRM_INTERVAL", 5, int)


STATEFULSET_NAME: str = read_env(
    "STATEFULSET_NAME",
    "vpa-test-app",
    str,
)

# Laptop / kubectl port-forward safe
PROMETHEUS_URL: str = read_env(
    "PROMETHEUS_URL",
    "http://localhost:9090",
    str,
)

CONTAINER_NAME: str = read_env("CONTAINER_NAME", "nginx", str)


POLL_INTERVAL: int = read_env("POLL_INTERVAL", 3, int)

LIMIT_RATIO: float = read_env("LIMIT_RATIO", 1.3, float)

# ------------------------------------------------------------------------------
# Absolute, Laptop-Safe Paths
# ------------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).parent

MODEL_DIR: Path = read_path_env(
    "MODEL_DIR",
    BASE_DIR / "data" / "models",
)

STATE_DIR: Path = read_path_env(
    "STATE_DIR",
    BASE_DIR / "data" / "state",
)

STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE_PATH: Path = STATE_DIR / "state.json"

# ------------------------------------------------------------------------------
# ML Pipeline Metadata
# ------------------------------------------------------------------------------

WINDOW_SIZE: int = 60
FEATURE_COUNT: int = 20

FEATURE_ORDER: list[str] = [
    "cpu_usage", "mem_usage", "cpu_request", "cpu_limit",
    "mem_request", "mem_limit", "node_cpu_usage", "node_cpu_steal",
    "node_mem_available", "replicas", "latency", "cpu_throttling",
    "throttled_periods", "net_receive", "net_transmit", "pod_restarts",
    "oom_kills", "http_requests_total", "ops_per_sec", "errors_5xx"
]

# This dictionary maps the friendly name above to the actual PromQL logic.
# It ensures the 'Feature Order' is preserved during data fetching.
FEATURE_QUERIES: dict[str, str] = {
    "cpu_usage": 'rate(container_cpu_usage_seconds_total{pod=~"$POD", namespace="$NS"}[1m])',
    "mem_usage": 'container_memory_working_set_bytes{pod=~"$POD", namespace="$NS"}',
    "cpu_request": 'kube_pod_container_resource_requests{pod=~"$POD", resource="cpu"}',
    "cpu_limit": 'kube_pod_container_resource_limits{pod=~"$POD", resource="cpu"}',
    "mem_request": 'kube_pod_container_resource_requests{pod=~"$POD", resource="memory"}',
    "mem_limit": 'kube_pod_container_resource_limits{pod=~"$POD", resource="memory"}',
    "node_cpu_usage": 'sum(rate(node_cpu_seconds_total{mode!="idle"}[1m]))',
    "node_cpu_steal": 'sum(rate(node_cpu_seconds_total{mode="steal"}[1m]))',
    "node_mem_available": 'node_memory_MemAvailable_bytes',
    "replicas": 'kube_deployment_status_replicas{deployment="$DEPLOYMENT"}',
    "latency": 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))',
    "cpu_throttling": 'rate(container_cpu_cfs_throttled_seconds_total[1m])',
    "throttled_periods": 'rate(container_cpu_cfs_throttled_periods_total[1m])',
    "net_receive": 'rate(container_network_receive_bytes_total[1m])',
    "net_transmit": 'rate(container_network_transmit_bytes_total[1m])',
    "pod_restarts": 'kube_pod_container_status_restarts_total{pod=~"$POD"}',
    "oom_kills": 'kube_pod_container_status_terminated_reason{reason="OOMKilled"}',
    "http_requests_total": 'rate(http_requests_total[1m])',
    "ops_per_sec": 'rate(app_operations_total[1m])',
    "errors_5xx": 'rate(http_requests_total{status=~"5.."}[1m])'
}


SAFETY_BUFFER: float = 1.15

MODEL_PATH: Path = MODEL_DIR / "k8s_gru_checkpoint.pt"
FEATURE_SCALER_PATH: Path = MODEL_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH: Path = MODEL_DIR / "target_scaler.pkl"

# ------------------------------------------------------------------------------
# Prometheus Client Synchronization
# ------------------------------------------------------------------------------

WINDOW_SECONDS: int = 60
STEP_SECONDS: int = 1
SCRAPE_DELAY_BUFFER: int = 10
MIN_HEALTH_THRESHOLD: float = 0.90
MODEL_SEMANTICS: str = "aggregate"  # or "per_pod"

# ------------------------------------------------------------------------------
# Immutable Model Artifacts
# ------------------------------------------------------------------------------

if not MODEL_DIR.exists():
    raise RuntimeError(
        f"MODEL_DIR does not exist: '{MODEL_DIR}'. "
        "Model artifacts must be baked into the image or mounted explicitly."
    )

# ------------------------------------------------------------------------------
# Resource Governance (Laptop-Safe)
# ------------------------------------------------------------------------------

class CpuBounds(NamedTuple):
    """
    CPU bounds in cores (float).
    """
    minimum: float
    maximum: float
    floor: float


class MemoryBounds(NamedTuple):
    """
    Memory bounds in MiB (int).
    """
    minimum: int
    maximum: int
    floor: int


CPU_BOUNDS = CpuBounds(
    minimum=0.01,
    maximum=1.0,
    floor=0.1,
)

MEMORY_BOUNDS = MemoryBounds(
    minimum=32,
    maximum=1024,
    floor=128,
)

# ------------------------------------------------------------------------------
# Validation (Fail Fast)
# ------------------------------------------------------------------------------

def validate_resources() -> None:
    """
    Validate resource bounds logic and invariants.
    """
    if CPU_BOUNDS.minimum <= 0:
        raise RuntimeError("CPU minimum must be > 0 cores")

    if CPU_BOUNDS.maximum <= CPU_BOUNDS.minimum:
        raise RuntimeError(
            f"CPU maximum must be greater than minimum "
            f"(min={CPU_BOUNDS.minimum}, max={CPU_BOUNDS.maximum})"
        )

    if CPU_BOUNDS.floor < CPU_BOUNDS.minimum:
        raise RuntimeError(
            f"CPU floor must be >= minimum "
            f"(floor={CPU_BOUNDS.floor}, min={CPU_BOUNDS.minimum})"
        )

    if MEMORY_BOUNDS.minimum <= 0:
        raise RuntimeError("Memory minimum must be > 0 MiB")

    if MEMORY_BOUNDS.maximum <= MEMORY_BOUNDS.minimum:
        raise RuntimeError(
            f"Memory maximum must be greater than minimum "
            f"(min={MEMORY_BOUNDS.minimum}, max={MEMORY_BOUNDS.maximum})"
        )

    if MEMORY_BOUNDS.floor < MEMORY_BOUNDS.minimum:
        raise RuntimeError(
            f"Memory floor must be >= minimum "
            f"(floor={MEMORY_BOUNDS.floor}, min={MEMORY_BOUNDS.minimum})"
        )

    if LIMIT_RATIO < 1.0:
        raise RuntimeError("LIMIT_RATIO must be >= 1.0")

    if POLL_INTERVAL <= 0:
        raise RuntimeError("POLL_INTERVAL must be > 0 seconds")

    if COOLDOWN_SECONDS < 0:
        raise RuntimeError(
            f"COOLDOWN_SECONDS must be >= 0 (got {COOLDOWN_SECONDS})"
        )


validate_resources()
