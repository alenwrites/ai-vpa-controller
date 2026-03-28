# admission_controller.py
#
# Control-plane critical Mutating Admission Webhook
#
# SAFETY & SEMANTICS (NON-NEGOTIABLE)
# ---------------------------------
# * Fail-open: pods are NEVER rejected due to webhook behavior
# * Stateless and idempotent: no UID caching, no mutation state
# * Deterministic: identical input AdmissionReview → identical patch
# * No blocking I/O or network calls on the request path
#
# IDPOTENCY GUARANTEE
# -------------------
# Kubernetes may retry admission requests with the SAME UID.
# This webhook intentionally rebuilds patches from the incoming Pod object
# on every call and does NOT cache by UID.
#
# This guarantees retry safety and prevents control-plane deadlocks.
#
# DO NOT add UID-based caching or request-local mutation state.

import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pythonjsonlogger import jsonlogger

from state_manager import StateManager

# =============================================================================
# Configuration (environment-driven, safe defaults)
# =============================================================================

CLUSTER_AI_VERSION = os.getenv("AI_POLICY_VERSION", "v1")
AI_ROLLOUT_ENABLED = os.getenv("AI_ROLLOUT_ENABLED", "true").lower() == "true"
DRY_RUN = os.getenv("AI_DRY_RUN", "false").lower() == "true"

STATE_MANAGER_TIMEOUT_SEC = float(os.getenv("STATE_MANAGER_TIMEOUT_SEC", "0.005"))

EXCLUDED_NAMESPACES = set(
    n.strip()
    for n in os.getenv(
        "AI_EXCLUDED_NAMESPACES",
        "kube-system,kube-public,kube-node-lease",
    ).split(",")
    if n.strip()
)

DISABLE_ANNOTATION = "ai.vpa.io/disable"

APP_LABEL_KEYS = [
    "app.kubernetes.io/instance", 
    "app",                        
    "statefulset.kubernetes.io/pod-name", 
]

#--------------------------
# InitContainer policy
# -------------------------

# Default: do NOT mutate initContainers
AI_MUTATE_INIT_CONTAINERS = (
    os.getenv("AI_MUTATE_INIT_CONTAINERS", "false").lower() == "true"
)

# Pod-level explicit opt-in for init containers (comma-separated names)
# Example:
# metadata.annotations:
#   ai.vpa.io/mutate-init: "db-migrate,setup"
INIT_CONTAINER_OPT_IN_ANNOTATION = os.getenv(
    "AI_INIT_CONTAINER_OPT_IN_ANNOTATION",
    "ai.vpa.io/mutate-init",
)

# =============================================================================
# Process-wide structured JSON logging
# =============================================================================

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(levelname)s %(name)s %(message)s "
    "%(event)s %(reason)s %(namespace)s %(pod)s %(app)s "
    "%(container)s %(cpu)s %(mem)s %(latency_sec)s %(dry_run)s"
)
handler.setFormatter(formatter)
root_logger.handlers = [handler]

log = logging.getLogger("vpa.admission")

# =============================================================================
# Metrics (clear, intention-revealing semantics)
# =============================================================================

pods_processed = Counter(
    "vpa_pods_processed_total",
    "Total pods processed by the admission webhook",
)

pods_mutated = Counter(
    "vpa_pods_mutated_total",
    "Pods for which a mutation patch was returned",
    ["app"],
)

pods_dry_run = Counter(
    "vpa_pods_dry_run_total",
    "Pods that would have been mutated but ran in dry-run mode",
    ["app"],
)

pods_skipped = Counter(
    "vpa_pods_skipped_total",
    "Pods skipped by reason",
    ["reason"],
)

state_latency = Histogram(
    "vpa_state_manager_latency_seconds",
    "Latency of StateManager.get_state()",
)

state_latency_breach = Counter(
    "vpa_state_manager_latency_breach_total",
    "StateManager latency exceeded budget",
)

# =============================================================================
# App / State
# =============================================================================

app = FastAPI()
state_manager = StateManager()

# =============================================================================
# Metrics endpoint
# =============================================================================


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Admission helpers
# =============================================================================


def admission_review(
    *,
    uid: str,
    allowed: bool = True,
    patch: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {"uid": uid, "allowed": allowed}

    if patch:
        patch_json = json.dumps(patch, separators=(",", ":"))
        response["patchType"] = "JSONPatch"
        response["patch"] = base64.b64encode(patch_json.encode()).decode()

    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": response,
    }


# =============================================================================
# Quantity helpers
# =============================================================================


def cpu_millicores_to_quantity(cpu_m: int) -> str:
    if cpu_m <= 0:
        raise ValueError("cpu must be > 0")
    return str(cpu_m // 1000) if cpu_m % 1000 == 0 else f"{cpu_m}m"


def memory_mib_to_quantity(mem_mib: int) -> str:
    if mem_mib <= 0:
        raise ValueError("mem must be > 0")
    return f"{mem_mib}Mi"


# =============================================================================
# Label resolution
# =============================================================================


def resolve_app_label(labels: Dict[str, str]) -> Optional[str]:
    # 1. Try standard labels first
    for key in APP_LABEL_KEYS:
        val = labels.get(key)
        if val:
            # INTEGRATION FIX: If it's a pod name like 'redis-0', strip the index 
            # so it matches the StatefulSet name 'redis'
            if "-" in val and val.split("-")[-1].isdigit():
                return "-".join(val.split("-")[:-1])
            return val
            
    return None


# =============================================================================
# Opt-out / sidecar logic
# =============================================================================


def pod_opted_out(metadata: Dict[str, Any]) -> bool:
    return (metadata.get("annotations") or {}).get(DISABLE_ANNOTATION) == "true"


def container_is_sidecar(container: Dict[str, Any]) -> bool:
    """
    Sidecar detection is explicit and conservative.

    Current rule:
    * container name ends with '-sidecar'

    No reliance on non-existent container fields.
    No vendor-specific heuristics.
    """
    return container.get("name", "").endswith("-sidecar")


def parse_init_container_opt_in(metadata: Dict[str, Any]) -> List[str]:
    """
    Parses pod-level annotation listing init containers explicitly allowed
    for mutation.

    Example:
      ai.vpa.io/mutate-init: "db-migrate,setup"
    """
    raw = (metadata.get("annotations") or {}).get(
        INIT_CONTAINER_OPT_IN_ANNOTATION, ""
    )
    return [name.strip() for name in raw.split(",") if name.strip()]


# =============================================================================
# Patch construction
# =============================================================================


def build_patch(
    *,
    base_path: str,
    container: Dict[str, Any],
    cpu_qty: str,
    mem_qty: str,
) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []

    if container.get("resources") is None:
        ops.append({"op": "add", "path": f"{base_path}/resources", "value": {}})

    for field in ("requests", "limits"):
        value = {"cpu": cpu_qty, "memory": mem_qty}
        if container.get("resources", {}).get(field) is None:
            ops.append(
                {"op": "add", "path": f"{base_path}/resources/{field}", "value": value}
            )
        else:
            ops.append(
                {
                    "op": "replace",
                    "path": f"{base_path}/resources/{field}",
                    "value": value,
                }
            )

    return ops


# =============================================================================
# Webhook handler
# =============================================================================


@app.post("/mutate")
async def mutate_pod(request: Request) -> Dict[str, Any]:
    uid = ""
    pods_processed.inc()

    try:
        body = await request.json()
        req = body.get("request", {})
        uid = req.get("uid", "")
        namespace = req.get("namespace", "")
        pod = req.get("object", {}) or {}

        metadata = pod.get("metadata", {}) or {}
        pod_name = metadata.get("name", "")
        labels = metadata.get("labels") or {}

        if namespace in EXCLUDED_NAMESPACES:
            pods_skipped.labels(reason="namespace").inc()
            return admission_review(uid=uid)

        if pod_opted_out(metadata):
            pods_skipped.labels(reason="pod_opt_out").inc()
            return admission_review(uid=uid)

        app_label = resolve_app_label(labels)
        if not app_label:
            pods_skipped.labels(reason="no_app_label").inc()
            return admission_review(uid=uid)

        if not AI_ROLLOUT_ENABLED:
            pods_skipped.labels(reason="rollout_disabled").inc()
            return admission_review(uid=uid)

        # ------------------------------------------------------------------
        # StateManager latency measurement (monotonic, correct)
        # ------------------------------------------------------------------

        start = time.perf_counter()
        ai_state = state_manager.get_state(app_label)
        latency = time.perf_counter() - start
        state_latency.observe(latency)

        if latency > STATE_MANAGER_TIMEOUT_SEC:
            state_latency_breach.inc()

        if not ai_state or ai_state.get("version") != CLUSTER_AI_VERSION:
            pods_skipped.labels(reason="no_or_bad_ai_state").inc()
            return admission_review(uid=uid)

        default_cpu = int(ai_state.get("cpu", 0))
        default_mem = int(ai_state.get("mem", 0))

        cpu_qty_default = cpu_millicores_to_quantity(default_cpu)
        mem_qty_default = memory_mib_to_quantity(default_mem)

        container_overrides = ai_state.get("containers", {}) or {}
        init_opt_in = set(parse_init_container_opt_in(metadata))

        patch: List[Dict[str, Any]] = []
        spec = pod.get("spec", {}) or {}

        def mutate_container(container, path, is_init: bool):
            name = container.get("name", "")

            if container_is_sidecar(container):
                pods_skipped.labels(reason="sidecar").inc()
                return

            if is_init:
                if not AI_MUTATE_INIT_CONTAINERS:
                    pods_skipped.labels(reason="init_container_skipped").inc()
                    log.info(
                        "init container skipped by policy",
                        extra={
                            "event": "skip_container",
                            "reason": "init_policy",
                            "namespace": namespace,
                            "pod": pod_name,
                            "container": name,
                        },
                    )
                    return

                if name not in container_overrides and name not in init_opt_in:
                    pods_skipped.labels(reason="init_container_skipped").inc()
                    return

            override = container_overrides.get(name, {})
            cpu_m = int(override.get("cpu", default_cpu))
            mem_m = int(override.get("mem", default_mem))

            cpu_q = cpu_millicores_to_quantity(cpu_m)
            mem_q = memory_mib_to_quantity(mem_m)

            patch.extend(
                build_patch(
                    base_path=path,
                    container=container,
                    cpu_qty=cpu_q,
                    mem_qty=mem_q,
                )
            )

        for i, c in enumerate(spec.get("containers") or []):
            mutate_container(c, f"/spec/containers/{i}", is_init=False)

        for i, c in enumerate(spec.get("initContainers") or []):
            mutate_container(c, f"/spec/initContainers/{i}", is_init=True)

        if not patch:
            return admission_review(uid=uid)

        if DRY_RUN:
            pods_dry_run.labels(app=app_label).inc()
            log.info(
                "dry-run mutation",
                extra={
                    "event": "dry_run",
                    "namespace": namespace,
                    "pod": pod_name,
                    "app": app_label,
                    "dry_run": True,
                },
            )
            return admission_review(uid=uid)

        pods_mutated.labels(app=app_label).inc()
        return admission_review(uid=uid, patch=patch)

    except Exception:
        log.exception(
            "webhook error, allowing pod",
            extra={"event": "error", "uid": uid},
        )
        return admission_review(uid=uid)
