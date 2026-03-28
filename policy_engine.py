# policy_engine.py
import logging
import math
import time
from typing import Callable, List, NamedTuple, Optional, Tuple

import config

log = logging.getLogger("vpa.policy")


class PolicyDecision(NamedTuple):
    allowed: bool
    reason: str


class PolicyEngine:
    """
    Deterministic infrastructure brake to prevent resize thrashing.

    ========================= OPERATIONAL CONTRACT =========================

    This policy engine is intentionally:

    - FAIL-CLOSED
    - DETERMINISTIC
    - HUMAN-OPERATED

    It prioritizes safety and predictability over availability or numerical purity.
    """

    def __init__(
        self,
        *,
        now: Optional[Callable[[], float]] = None,
        maintenance_windows_utc: Optional[List[Tuple[int, int]]] = None,
    ):
        self._now = now or time.time

        self.cooldown_period = getattr(config, "COOLDOWN_SECONDS", 600)
        self.min_cpu_change_ratio = getattr(
            config, "MIN_CPU_CHANGE_THRESHOLD", 0.1
        )

        # HARD CONTRACT:
        # CPU values are floats expressed in *cores*.
        # This layer assumes trusted callers and does not perform unit inference.
        self.min_cpu_change_abs = getattr(
            config, "MIN_CPU_CHANGE_ABSOLUTE", 0.05
        )

        self.min_mem_change = getattr(
            config, "MIN_MEM_CHANGE_MiB", 64
        )

        self.maintenance_windows_utc: List[Tuple[int, int]] = (
            maintenance_windows_utc
            if maintenance_windows_utc is not None
            else getattr(config, "MAINTENANCE_WINDOWS_UTC", [(2, 3)])
        )

        for window in self.maintenance_windows_utc:
            if not isinstance(window, tuple) or len(window) != 2:
                raise ValueError(
                    f"Invalid maintenance window format: {window!r}"
                )
            start, end = window
            if not (0 <= start <= 23 and 0 <= end <= 24):
                raise ValueError(
                    f"Maintenance window hours must be in UTC range [0,24): {window}"
                )
            if start >= end:
                raise ValueError(
                    f"Wrapping or invalid maintenance window not supported: {window}"
                )

    # ------------------------------------------------------------------

    def is_resize_allowed(
        self,
        target_name: str,
        last_update_ts: float,
        current_cpu: float,
        current_mem: int,
        proposed_cpu: float,
        proposed_mem: int,
    ) -> PolicyDecision:
        now = self._now()

        if not self._is_valid_epoch_seconds(last_update_ts, now):
            reason = (
                f"Invalid last_update_ts for {target_name}; "
                "cooldown state unsafe. Resize denied (fail-closed)."
            )
            self._log_denial(target_name, reason)
            return PolicyDecision(False, reason)

        elapsed = now - last_update_ts
        remaining = self.cooldown_period - elapsed

        log.debug(
            "Cooldown check for %s: elapsed=%.2fs, remaining=%.2fs",
            target_name,
            elapsed,
            max(0.0, remaining),
        )

        if elapsed < self.cooldown_period:
            reason = (
                f"Cooldown active for {target_name}. "
                f"{int(remaining)}s remaining."
            )
            self._log_denial(target_name, reason)
            return PolicyDecision(False, reason)

        current_hour_utc = self._utc_hour_from_epoch(now)
        window = self._matching_maintenance_window(current_hour_utc)
        if window is not None:
            start, end = window
            reason = (
                f"Maintenance window active ({start:02d}:00–{end:02d}:00 UTC). "
                f"Resizes frozen for {target_name}."
            )
            self._log_denial(target_name, reason)
            return PolicyDecision(False, reason)

        if not self._is_valid_cpu_value(current_cpu) or not self._is_valid_cpu_value(
            proposed_cpu
        ):
            reason = (
                f"Invalid CPU values for {target_name} "
                f"(current={current_cpu}, proposed={proposed_cpu}). "
                "Resize denied (fail-closed)."
            )
            self._log_denial(target_name, reason)
            return PolicyDecision(False, reason)

        if not self._is_valid_mem_value(current_mem) or not self._is_valid_mem_value(
            proposed_mem
        ):
            reason = (
                f"Invalid memory values for {target_name} "
                f"(current={current_mem}, proposed={proposed_mem}). "
                "Resize denied (fail-closed)."
            )
            self._log_denial(target_name, reason)
            return PolicyDecision(False, reason)

        # ---- CPU significance gate ----
        #
        # FLOATING-POINT COMPARISON NOTE (INTENTIONAL):
        #
        # The comparisons below intentionally use direct IEEE-754 float
        # comparisons with no epsilon or tolerance logic.
        #
        # Rationale:
        # - This is a heuristic, operational policy gate, not a numerical algorithm.
        # - CPU values are not persisted across heterogeneous systems here.
        # - Exact boundary behavior is NOT contractually required.
        # - Minor floating-point jitter near thresholds is operationally irrelevant
        #   compared to the cooldown and magnitude guards already in place.
        #
        # Introducing epsilon logic would:
        # - Reduce determinism
        # - Increase cognitive load
        # - Create false expectations of mathematical precision
        #
        # If future requirements demand cross-system comparison or strict
        # boundary guarantees, this decision must be revisited explicitly.
        cpu_abs_delta = abs(proposed_cpu - current_cpu)

        if current_cpu > 0:
            cpu_rel_delta = cpu_abs_delta / current_cpu
        else:
            cpu_rel_delta = cpu_abs_delta

        cpu_rel_delta_log = min(cpu_rel_delta, 10.0)

        cpu_change_significant = (
            cpu_abs_delta >= self.min_cpu_change_abs
            or cpu_rel_delta >= self.min_cpu_change_ratio
        )

        mem_delta = abs(proposed_mem - current_mem)
        mem_change_significant = mem_delta >= self.min_mem_change

        if not cpu_change_significant and not mem_change_significant:
            reason = (
                f"Change too small for {target_name} "
                f"(CPU Δ: {cpu_abs_delta:.3f} cores / {cpu_rel_delta_log:.2%}, "
                f"Mem Δ: {mem_delta} MiB). Ignoring."
            )
            self._log_denial(target_name, reason)
            return PolicyDecision(False, reason)

        log.debug("Resize allowed for %s: policy criteria met.", target_name)
        return PolicyDecision(True, "Policy criteria met.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_epoch_seconds(ts: float, now: float) -> bool:
        """
        Validate that ts resembles sane UTC epoch *seconds*.

        HEURISTIC NOTE:
        - The 1e11 upper bound is a heuristic.
        - It exists to catch millisecond timestamps and corrupted state.
        - It intentionally trades theoretical correctness for operational safety.
        """
        if not isinstance(ts, (int, float)):
            return False
        if not math.isfinite(ts):
            return False
        if ts <= 0:
            return False
        if ts > now:
            return False
        if ts > 1e11:
            return False
        return True

    @staticmethod
    def _utc_hour_from_epoch(epoch_seconds: float) -> int:
        """
        Extract UTC hour [0–23] from epoch seconds.

        Time handling is intentionally "dumb by design":
        - UTC only
        - No DST
        - No leap seconds
        - No calendar logic

        Predictability > correctness in edge astronomical cases.
        """
        return int((epoch_seconds % 86400) // 3600)

    def _matching_maintenance_window(
        self, hour_utc: int
    ) -> Optional[Tuple[int, int]]:
        for start, end in self.maintenance_windows_utc:
            if start <= hour_utc < end:
                return (start, end)
        return None

    @staticmethod
    def _is_valid_cpu_value(value: float) -> bool:
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
        if value < 0:
            return False
        return True

    @staticmethod
    def _is_valid_mem_value(value: int) -> bool:
        if not isinstance(value, int):
            return False
        if value < 0:
            return False
        return True

    @staticmethod
    def _log_denial(target_name: str, reason: str) -> None:
        """
        INFO-level logging for denials is intentional.

        If volume becomes high, mitigate via log routing, filtering,
        or aggregation — not code changes.
        """
        log.info("Resize denied for %s: %s", target_name, reason)
