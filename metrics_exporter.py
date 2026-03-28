# metrics_exporter.py
"""
Prometheus metrics for the VPA controller.

Design guarantees:
- No unbounded labels
- Explicit aggregation semantics
- Enum labels enforced mechanically
- No NaNs or silent drops
- Safe under multiple controller instances

Non-goals (explicit):
- Per-target observability (belongs in logs/events)
- Cross-controller semantic aggregation
- Ultra–high-frequency hot-path optimization
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterable

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    start_http_server,
)

log = logging.getLogger("vpa.exporter")

# ------------------------------------------------------------------------------
# Registry (explicit, isolated, testable)
# ------------------------------------------------------------------------------
REGISTRY = CollectorRegistry(auto_describe=True)

# ------------------------------------------------------------------------------
# Label enums (hard bounded)
# ------------------------------------------------------------------------------
RESOURCE_LABELS: Iterable[str] = ("cpu", "memory")
DIRECTION_LABELS: Iterable[str] = ("upscale", "downscale", "unchanged")
DECISION_LABELS: Iterable[str] = ("allow", "clamp", "reject")

# ------------------------------------------------------------------------------
# Label enforcement policy
# ------------------------------------------------------------------------------
# Fail-fast is intentional and preferred.
#
# Even if this is ever disabled, invalid label values MUST NOT be emitted.
# Cardinality safety is enforced by construction, not by convention.
LABEL_ENFORCEMENT_FAIL_FAST = True

# Sentinel is bounded and shared across all enums.
_INVALID_LABEL_SENTINEL = "invalid"


def _require_enum(value: str, allowed: Iterable[str], label: str) -> str:
    """
    Enforces enum-like label safety.

    Behavior:
    - If value is valid: returned unchanged
    - If value is invalid:
        * Fail-fast mode: raise ValueError
        * Non-fail-fast mode: map to a bounded sentinel ("invalid")

    Rationale:
    - Invalid values can NEVER reach Prometheus
    - Cardinality remains bounded even under misconfiguration
    """
    if value in allowed:
        return value

    msg = (
        f"Invalid value '{value}' for label '{label}'. "
        f"Allowed values: {tuple(allowed)}"
    )

    if LABEL_ENFORCEMENT_FAIL_FAST:
        raise ValueError(msg)

    # Non-fail-fast mode is still Prometheus-safe by construction
    log.error(msg)
    return _INVALID_LABEL_SENTINEL


# ------------------------------------------------------------------------------
# AI vs Safety Outputs (controller-local snapshot gauges)
# ------------------------------------------------------------------------------
AI_SUGGESTED_RESOURCE_LAST = Gauge(
    name="vpa_ai_suggested_resource_last",
    documentation=(
        "Last AI-suggested resource value observed by THIS controller instance "
        "during reconciliation.\n\n"
        "Aggregation semantics:\n"
        "- Controller-local\n"
        "- Last-observed snapshot (overwritten on each reconciliation)\n"
        "- Multiple controllers may legitimately expose different values"
    ),
    labelnames=("resource",),
    registry=REGISTRY,
)

SAFETY_APPROVED_RESOURCE_LAST = Gauge(
    name="vpa_safety_approved_resource_last",
    documentation=(
        "Last safety-approved resource value applied by THIS controller instance.\n\n"
        "Aggregation semantics:\n"
        "- Controller-local\n"
        "- Last-observed snapshot\n"
        "- Not a cluster-wide aggregate"
    ),
    labelnames=("resource",),
    registry=REGISTRY,
)

# ------------------------------------------------------------------------------
# Safety Clamp Magnitude
# ------------------------------------------------------------------------------
SAFETY_CLAMP_RATIO = Histogram(
    name="vpa_safety_clamp_ratio",
    documentation=(
        "Ratio of safety-approved value to AI-suggested value "
        "(approved / suggested), observed per reconciliation.\n\n"
        "Interpretation:\n"
        "- ~1.0  => minimal safety interference\n"
        "- <1.0  => downscaling clamp\n"
        "- >1.0  => upscaling clamp\n\n"
        "Zero-division policy:\n"
        "- If suggested == 0, this metric is NOT observed\n"
        "- The event is recorded in vpa_safety_clamp_invalid_ratio_total"
    ),
    labelnames=("resource",),
    buckets=(0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0),
    registry=REGISTRY,
)

SAFETY_CLAMP_INVALID_RATIO_TOTAL = Counter(
    name="vpa_safety_clamp_invalid_ratio_total",
    documentation=(
        "Count of reconciliation attempts where clamp ratio could not be computed "
        "because the AI-suggested value was zero.\n\n"
        "Purpose:\n"
        "- Makes invalid ratios explicit\n"
        "- Prevents silent NaNs or misleading histogram samples"
    ),
    labelnames=("resource",),
    registry=REGISTRY,
)

SAFETY_INTERVENTION_DIRECTION_TOTAL = Counter(
    name="vpa_safety_intervention_direction_total",
    documentation=(
        "Count of safety interventions by direction relative to the AI suggestion.\n\n"
        "Label safety:\n"
        "- Direction is enum-enforced\n"
        "- Invalid values collapse to a bounded sentinel if enforcement is relaxed"
    ),
    labelnames=("resource", "direction"),
    registry=REGISTRY,
)

# ------------------------------------------------------------------------------
# Controller Decision Outcomes
# ------------------------------------------------------------------------------
VPA_DECISION_OUTCOME_TOTAL = Counter(
    name="vpa_decision_outcome_total",
    documentation=(
        "Count of reconciliation outcomes observed by THIS controller instance.\n\n"
        "Semantics:\n"
        "- allow  : AI suggestion applied unchanged\n"
        "- clamp  : AI suggestion modified by safety layer\n"
        "- reject : scaling prevented entirely"
    ),
    labelnames=("decision",),
    registry=REGISTRY,
)

# ------------------------------------------------------------------------------
# Control Loop Performance (STRICT loop-level contract)
# ------------------------------------------------------------------------------
RECONCILIATION_LOOP_DURATION_SECONDS = Histogram(
    name="vpa_reconciliation_loop_duration_seconds",
    documentation=(
        "Wall-clock duration of a FULL controller reconciliation loop, "
        "measured ONCE per loop execution by THIS controller instance.\n\n"
        "STRICT CONTRACT:\n"
        "- Must be observed exactly once per reconciliation loop\n"
        "- MUST NOT be emitted per target, per object, or per workload\n"
        "- Emitting this metric inside target iteration INVALIDATES latency analysis\n\n"
        "Rationale:\n"
        "- This metric represents controller scheduling and control-loop health\n"
        "- Per-target latency belongs in logs or tracing, NOT Prometheus\n\n"
        "Violating this contract will silently corrupt SLOs and dashboards."
    ),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)


# ------------------------------------------------------------------------------
# Forward-Looking Model / Policy Health Signals
# ------------------------------------------------------------------------------
AI_SUGGESTION_CHANGE_RATIO = Histogram(
    name="vpa_ai_suggestion_change_ratio",
    documentation=(
        "Ratio of the current AI-suggested value to the PREVIOUS suggestion "
        "for the same resource, as tracked by THIS controller instance.\n\n"
        "Stateful contract:\n"
        "- Caller is responsible for tracking previous values\n"
        "- Controller-local only\n"
        "- MUST NOT be emitted per target\n"
        "- Cross-controller aggregation may hide instability\n\n"
        "Purpose:\n"
        "- Detect large swings or oscillations in model output"
    ),
    labelnames=("resource",),
    buckets=(0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0),
    registry=REGISTRY,
)

SAFETY_CLAMP_RATE_EWMA = Gauge(
    name="vpa_safety_clamp_rate_ewma",
    documentation=(
        "Exponentially-weighted moving average (EWMA) of reconciliations "
        "that resulted in a clamp decision, as computed by THIS controller instance.\n\n"
        "IMPORTANT SCOPE:\n"
        "- Diagnostic, controller-local signal\n"
        "- Cross-controller aggregation is NOT meaningful\n"
        "- NOT suitable for SLOs or naive alerting\n\n"
        "Intent:\n"
        "- Identify sustained policy restrictiveness trends locally"
    ),
    registry=REGISTRY,
)

# ------------------------------------------------------------------------------
# Reconciliation loop timing helper (guardrail API)
# ------------------------------------------------------------------------------
# NOTE:
# This helper exists purely as a usage guardrail.
# It does NOT implement controller logic and may be ignored by callers.

# NON-GOALS (intentional):
# - Measuring per-target reconciliation latency
# - Deriving object-level performance from Prometheus
#
# If per-target latency is required:
# - Use structured logs
# - Use tracing spans
# - DO NOT extend this metric

@contextmanager
def reconciliation_loop_timer():
    """
    Context manager for measuring ONE reconciliation loop duration.

    Usage:
        with reconciliation_loop_timer():
            reconcile_all_targets()

    Guardrails:
    - Encourages exactly-once emission
    - Makes per-target emission awkward and obvious
    - Keeps exporter free of business logic
    """
    start = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - start
        RECONCILIATION_LOOP_DURATION_SECONDS.observe(duration)



# ------------------------------------------------------------------------------
# Safe emitters
# ------------------------------------------------------------------------------
# Concurrency model:
# - These functions may be called concurrently
# - They rely on prometheus_client thread-safety guarantees
# - No additional locking is performed by design
# - Extremely hot-path usage may incur internal contention
#
# This exporter intentionally favors simplicity and safety over micro-optimizations.
def observe_ai_suggested(resource: str, value: float) -> None:
    resource = _require_enum(resource, RESOURCE_LABELS, "resource")
    AI_SUGGESTED_RESOURCE_LAST.labels(resource=resource).set(value)


def observe_safety_approved(resource: str, value: float) -> None:
    resource = _require_enum(resource, RESOURCE_LABELS, "resource")
    SAFETY_APPROVED_RESOURCE_LAST.labels(resource=resource).set(value)


def observe_clamp(
    resource: str,
    suggested: float,
    approved: float,
    direction: str,
) -> None:
    resource = _require_enum(resource, RESOURCE_LABELS, "resource")
    direction = _require_enum(direction, DIRECTION_LABELS, "direction")

    SAFETY_INTERVENTION_DIRECTION_TOTAL.labels(
        resource=resource, direction=direction
    ).inc()

    if suggested == 0:
        SAFETY_CLAMP_INVALID_RATIO_TOTAL.labels(resource=resource).inc()
        return

    SAFETY_CLAMP_RATIO.labels(resource=resource).observe(
        approved / suggested
    )


def count_decision(decision: str) -> None:
    decision = _require_enum(decision, DECISION_LABELS, "decision")
    VPA_DECISION_OUTCOME_TOTAL.labels(decision=decision).inc()


# ------------------------------------------------------------------------------
# Metrics Server (idempotent, defensive)
# ------------------------------------------------------------------------------
_server_lock = threading.Lock()
_server_started = False


def start_metrics_server(port: int = 8000) -> None:
    """
    Starts the Prometheus metrics HTTP server exactly once.

    Guarantees:
    - Idempotent under retries and leader election
    - No duplicate registry exposure
    - Loud failure on bind errors
    """
    global _server_started

    with _server_lock:
        if _server_started:
            log.debug("Metrics server already started; skipping.")
            return

        try:
            start_http_server(port, registry=REGISTRY)
            _server_started = True
            log.info(
                "Metrics server started at http://0.0.0.0:%d/metrics", port
            )
        except Exception:
            log.exception("Failed to start Prometheus metrics server")
            raise
