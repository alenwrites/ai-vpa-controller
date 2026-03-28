# safety_layer.py
import logging
import math
from typing import Tuple

import config

log = logging.getLogger("vpa.safety")


class SafetyPolicy:
    """
    Explicit safety policy container.

    This groups all safety-critical tunables in one place to avoid
    implicit, scattered config dependencies.

    Operators are expected to audit this section carefully.
    """

    # --------------------
    # Redis fork safety policy
    # --------------------

    # Redis persistence (BGSAVE / AOF rewrite) requires forking.
    # Forking is memory-critical and can temporarily require ~2x RSS.
    REDIS_FORK_MULTIPLIER: float = getattr(config, "REDIS_FORK_MULTIPLIER", 1.5)

    # Whether Redis fork safety is allowed to intentionally exceed
    # cluster MEMORY_BOUNDS.maximum.
    #
    # Rationale:
    # - Exceeding infra intent is preferable to OOM-killing Redis,
    #   which causes data loss, failover storms, and cascading failures.
    # - This must be explicit and auditable.
    ALLOW_REDIS_OOM_OVERRIDE: bool = True

    # Absolute hard cap in MiB, even if override is enabled.
    # Set to None to disable.
    ABSOLUTE_MAX_MEMORY_MIB: int | None = None

    # --------------------
    # Downscale dampening policy
    # --------------------

    # Maximum allowed per-cycle memory decrease (e.g. 0.5 == max 50% drop)
    MAX_MEM_DOWNSCALE_RATIO: float = 0.5

    # Base for downscale floor calculation:
    #
    # "usage":
    #   Floor is based on current Redis memory usage (RSS).
    #   Safer for Redis: avoids shrinking below live footprint.
    #
    # "limit":
    #   Floor is based on the current memory *limit*.
    #   Allows faster convergence but risks Redis fragmentation/OOM.
    #
    # Default is "usage" to bias toward Redis safety.
    DOWNSCALE_BASE: str = "usage"


class SafetyLayer:
    """
    Hard safety firewall between ML predictions and Kubernetes resource patching.

    Memory units: MiB
    CPU units: cores
    """

    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()

        # Infrastructure-enforced bounds
        self.cpu_min = config.CPU_BOUNDS.minimum
        self.cpu_max = config.CPU_BOUNDS.maximum
        self.mem_min = config.MEMORY_BOUNDS.minimum
        self.mem_max = config.MEMORY_BOUNDS.maximum

    def validate_prediction(
        self,
        pred_cpu: float,
        pred_mem: int,
        current_mem_usage_mib: float,
        current_mem_limit_mib: int | None = None,
    ) -> Tuple[float, int]:
        """
        Validate and clamp ML-predicted resources.

        Args:
            pred_cpu: Predicted CPU cores.
            pred_mem: Predicted memory limit in MiB.
            current_mem_usage_mib: Current Redis memory usage in MiB
                (expected: used_memory_rss converted to MiB).
            current_mem_limit_mib: Current memory limit in MiB (required
                when DOWNSCALE_BASE == "limit").

        Returns:
            (safe_cpu_cores, safe_memory_mib)
        """

        # --------------------
        # Strict input validation (fail fast)
        # --------------------
        if not isinstance(pred_cpu, (int, float)):
            raise TypeError("pred_cpu must be a number")
        if math.isnan(pred_cpu) or math.isinf(pred_cpu) or pred_cpu <= 0:
            raise ValueError(f"Invalid pred_cpu: {pred_cpu}")

        if not isinstance(pred_mem, int) or pred_mem <= 0:
            raise ValueError(f"Invalid pred_mem (MiB): {pred_mem}")

        if not isinstance(current_mem_usage_mib, (int, float)):
            raise TypeError("current_mem_usage_mib must be a number (MiB)")
        if (
            math.isnan(current_mem_usage_mib)
            or math.isinf(current_mem_usage_mib)
            or current_mem_usage_mib < 0
        ):
            raise ValueError(
                f"Invalid current_mem_usage_mib: {current_mem_usage_mib}"
            )

        if self.policy.DOWNSCALE_BASE == "limit":
            if current_mem_limit_mib is None:
                raise ValueError(
                    "current_mem_limit_mib is required when "
                    "DOWNSCALE_BASE == 'limit'"
                )
            if not isinstance(current_mem_limit_mib, int) or current_mem_limit_mib <= 0:
                raise ValueError(
                    f"Invalid current_mem_limit_mib: {current_mem_limit_mib}"
                )

        # --------------------
        # CPU safety
        # --------------------
        # Redis forks are memory-critical, not CPU-critical.
        # CPU spikes during fork are scheduler-managed, transient,
        # and do not cause process death the way OOM does.
        safe_cpu = max(self.cpu_min, min(pred_cpu, self.cpu_max))

        # --------------------
        # Memory safety (MiB)
        # --------------------

        # Step 1: Infrastructure clamp
        infra_clamped_mem = max(self.mem_min, min(pred_mem, self.mem_max))

        # Step 2: Redis fork safety
        redis_required_mem = int(
            math.ceil(current_mem_usage_mib * self.policy.REDIS_FORK_MULTIPLIER)
        )

        if redis_required_mem > infra_clamped_mem:
            if not self.policy.ALLOW_REDIS_OOM_OVERRIDE:
                raise RuntimeError(
                    "Redis fork safety requires %d MiB, exceeds infra limit %d MiB "
                    "and override is disabled."
                    % (redis_required_mem, infra_clamped_mem)
                )

            if (
                self.policy.ABSOLUTE_MAX_MEMORY_MIB is not None
                and redis_required_mem > self.policy.ABSOLUTE_MAX_MEMORY_MIB
            ):
                raise RuntimeError(
                    "Redis fork safety requires %d MiB, exceeds absolute hard cap %d MiB."
                    % (
                        redis_required_mem,
                        self.policy.ABSOLUTE_MAX_MEMORY_MIB,
                    )
                )

            log.warning(
                "Redis fork safety override: requiring %d MiB, exceeding infra "
                "maximum %d MiB. This is an intentional, auditable policy decision.",
                redis_required_mem,
                infra_clamped_mem,
            )
            safe_mem = redis_required_mem
        else:
            safe_mem = infra_clamped_mem

        # --------------------
        # Downscale dampening
        # --------------------
        # Prevent aggressive per-cycle drops that cause allocator
        # fragmentation and Redis instability.
        if self.policy.DOWNSCALE_BASE == "usage":
            downscale_base = current_mem_usage_mib
        else:
            downscale_base = current_mem_limit_mib  # validated above

        if downscale_base > 0:
            min_allowed = int(
                math.ceil(downscale_base * self.policy.MAX_MEM_DOWNSCALE_RATIO)
            )
            if safe_mem < min_allowed:
                log.warning(
                    "Memory downscale dampened (%s-based): predicted %d MiB, "
                    "raised to %d MiB.",
                    self.policy.DOWNSCALE_BASE,
                    safe_mem,
                    min_allowed,
                )
                safe_mem = min_allowed

        log.info(
            "Safety check passed: CPU %.3f -> %.3f, MEM %d MiB -> %d MiB",
            pred_cpu,
            safe_cpu,
            pred_mem,
            safe_mem,
        )

        return safe_cpu, safe_mem
