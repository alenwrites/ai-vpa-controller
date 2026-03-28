# event_logger.py
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from kubernetes import client
from kubernetes.client.exceptions import ApiException

log = logging.getLogger("vpa.event_logger")


class EventLogger:
    """
    Publishes Kubernetes Events as ephemeral, best-effort operational signals.

    Events provide context for humans and debugging tools observing the cluster.
    They are NOT a durable audit log and must never affect reconciliation
    correctness or liveness.

    Taxonomy contract (stability-sensitive):
      - action: what happened (mechanical outcome)
      - reason: why it happened (stable category, closed set)
      - note:   detailed, free-form human explanation
    """

    # ---- Action taxonomy -------------------------------------------------
    # Actions describe *what happened*.
    # Closed and intentionally small to avoid semantic drift.
    ACTION_SCALING_RECOMMENDATION_APPLIED = "ScalingRecommendationApplied"
    ACTION_SAFETY_OVERRIDE_APPLIED = "SafetyOverrideApplied"
    ACTION_POLICY_BLOCKED = "PolicyBlocked"
    ACTION_NOOP = "NoOp"

    _ALLOWED_ACTIONS = {
        ACTION_SCALING_RECOMMENDATION_APPLIED,
        ACTION_SAFETY_OVERRIDE_APPLIED,
        ACTION_POLICY_BLOCKED,
        ACTION_NOOP,
    }

    # ---- Reason taxonomy -------------------------------------------------
    # Reasons describe *why* an action occurred.
    #
    # This is part of a public, stability-sensitive contract: dashboards,
    # alerts, and runbooks may depend on these values. New reasons should
    # be added deliberately and rarely.
    REASON_POLICY_EVALUATION = "PolicyEvaluation"
    REASON_RESOURCE_PRESSURE = "ResourcePressure"
    REASON_SAFETY_GUARDRAIL = "SafetyGuardrail"
    REASON_NO_CHANGE_NEEDED = "NoChangeNeeded"

    _ALLOWED_REASONS = {
        REASON_POLICY_EVALUATION,
        REASON_RESOURCE_PRESSURE,
        REASON_SAFETY_GUARDRAIL,
        REASON_NO_CHANGE_NEEDED,
    }

    # Rate-limit identical error logs to avoid log storms on RBAC/API failure
    _ERROR_LOG_COOLDOWN = 300  # seconds

    # Small backoff (in seconds) before a single retry on patch conflict
    _PATCH_RETRY_BACKOFF = 0.05

    def __init__(self):
        # Use the modern Events API exclusively (events.k8s.io/v1).
        self.events_v1 = client.EventsV1Api()
        self.source_component = "ai-vpa-controller"

        self._last_error_log_ts: float = 0.0

    def emit(
        self,
        *,
        target_name: str,
        target_kind: str,
        namespace: str,
        api_version: str,
        reason: str,
        action: str,
        message: str,
        is_warning: bool = False,
        uid: Optional[str] = None,
    ) -> None:
        """
        Emit or aggregate a Kubernetes Event.

        Aggregation key is (controller, involvedObject, reason, type).
        Repeated emissions increment the series count instead of creating
        new Event objects.

        This method must never block reconciliation.
        """
        # Enforce closed action vocabulary.
        if action not in self._ALLOWED_ACTIONS:
            self._log_emit_error(
                target_name,
                ValueError(f"Unknown event action: {action!r}"),
            )
            return

        # Enforce closed reason vocabulary to prevent silent semantic drift.
        if reason not in self._ALLOWED_REASONS:
            self._log_emit_error(
                target_name,
                ValueError(f"Unknown event reason: {reason!r}"),
            )
            return

        now = datetime.now(timezone.utc)

        # event_type is derived solely from is_warning.
        # Allowed values per the API are exactly "Normal" and "Warning".
        event_type = "Warning" if is_warning else "Normal"

        # Controller-scoped, deterministic, DNS-safe name:
        # <controller>.<object>.<reason>
        # Prevents collisions across controllers and replicas while enabling
        # native kube-apiserver aggregation.
        event_name = (
            f"{self.source_component}.{target_name}.{reason}"
        ).lower()

        involved_object = client.V1ObjectReference(
            kind=target_kind,
            name=target_name,
            namespace=namespace,
            api_version=api_version,
            uid=uid,
        )

        event = client.V1Event(
            metadata=client.V1ObjectMeta(
                name=event_name,
                namespace=namespace,
            ),
            involved_object=involved_object,
            reason=reason,          # why it happened (stable category)
            action=action,          # what happened (mechanical outcome)
            note=message,           # free-form human explanation
            type=event_type,
            reporting_controller=self.source_component,
            reporting_instance=self.source_component,
            event_time=now,         # single authoritative timestamp
        )

        try:
            # Fast path: create if this is the first occurrence.
            self.events_v1.create_namespaced_event(
                namespace=namespace,
                body=event,
            )
            return
        except ApiException as e:
            if e.status != 409:
                # Non-conflict failures (RBAC, auth, etc.) are non-fatal.
                self._log_emit_error(target_name, e)
                return

        # 409 Conflict → Event already exists, aggregate via read-modify-patch.
        self._patch_existing_event(
            namespace=namespace,
            name=event_name,
            now=now,
            target_name=target_name,
        )

    def _patch_existing_event(
        self,
        *,
        namespace: str,
        name: str,
        now: datetime,
        target_name: str,
    ) -> None:
        """
        Fetch existing Event, increment its series count, and patch it.

        A single short retry is used to tolerate concurrent updaters without
        introducing unbounded retries or blocking behavior.
        """
        for attempt in (1, 2):
            try:
                existing = self.events_v1.read_namespaced_event(
                    name=name,
                    namespace=namespace,
                )

                # Kubernetes controllers conventionally treat a missing
                # series.count as 0, incrementing to 1 on first aggregation.
                series = existing.series
                current_count = series.count if (series and series.count) else 0

                patch = {
                    "series": {
                        "count": current_count + 1,
                        "lastObservedTime": now.isoformat(),
                    },
                }

                self.events_v1.patch_namespaced_event(
                    name=name,
                    namespace=namespace,
                    body=patch,
                )
                return
            except ApiException as e:
                if e.status == 409 and attempt == 1:
                    # Minimal hardening: one short backoff and retry.
                    time.sleep(self._PATCH_RETRY_BACKOFF)
                    continue

                self._log_emit_error(target_name, e)
                return

    def _log_emit_error(self, target_name: str, exc: Exception) -> None:
        """
        Log event emission failures sparingly.

        Event failures are expected in some environments (RBAC, API disabled).
        Logging is rate-limited to avoid obscuring real controller issues.
        """
        now = time.time()
        if now - self._last_error_log_ts < self._ERROR_LOG_COOLDOWN:
            return

        self._last_error_log_ts = now
        log.warning(
            "Failed to emit Kubernetes Event for %s (suppressed for %ss): %s",
            target_name,
            self._ERROR_LOG_COOLDOWN,
            exc,
        )
