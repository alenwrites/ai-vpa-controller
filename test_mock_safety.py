# tests/test_mock_safety.py
"""
SAFETY CONTRACT ASSUMPTIONS (NON-NEGOTIABLE)

1. Enforcement Semantics
   - SafetyLayer enforcement is ATOMIC per unsafe decision.
   - _apply_limits is called exactly once.
   - _audit_override is called exactly once.
   - Retries, staging, or backoff MUST occur outside SafetyLayer.

2. Brain Isolation
   - Unsafe or malformed inputs MUST NOT consult the Brain.
   - SafetyLayer acts as a firewall, not a post-filter.

3. Memory Model
   - current_mem_limit is a HARD CEILING.
   - Oversubscription is NOT permitted.
   - SafetyLayer must prevent:
       * OOM risk
       * quota violations
       * runaway allocation

If any of these assumptions change, tests MUST be updated explicitly.
Silent semantic drift is forbidden.
"""

import math
import unittest
from unittest.mock import Mock

from safety_layer import SafetyLayer


class TestSafetyLayerSafetyContract(unittest.TestCase):
    """
    If this suite passes, SafetyLayer is safe against adversarial,
    malformed, and boundary-case inputs.
    """

    def setUp(self):
        self.safety = SafetyLayer()

        # ---- Policy-driven configuration ----
        self.config = Mock()
        self.config.MIN_CPU_CORES = 0.1
        self.config.MAX_CPU_CORES = 4.0
        self.config.MEMORY_BUFFER_MIB = 128

        self.safety.config = self.config

        # ---- Trust boundary mocks ----
        self.safety._apply_limits = Mock()
        self.safety._audit_override = Mock()

        self.safety.brain = Mock()
        self.safety.brain.predict = Mock()

        self.current_mem_usage = 512
        self.current_mem_limit = 1024

    # ------------------------------------------------------------------
    # Invariant helpers (explicit policy intent)
    # ------------------------------------------------------------------
    def assert_cpu_invariants(self, cpu):
        self.assertFalse(math.isnan(cpu))
        self.assertFalse(math.isinf(cpu))
        self.assertGreaterEqual(cpu, self.config.MIN_CPU_CORES)
        self.assertLessEqual(cpu, self.config.MAX_CPU_CORES)

    def assert_mem_invariants(self, mem):
        self.assertFalse(math.isnan(mem))
        self.assertFalse(math.isinf(mem))
        self.assertGreaterEqual(
            mem,
            self.current_mem_usage + self.config.MEMORY_BUFFER_MIB,
        )
        self.assertLessEqual(
            mem,
            self.current_mem_limit,
            msg="Memory exceeds hard ceiling — oversubscription forbidden",
        )

    def assert_atomic_override_path(self):
        """
        Atomic enforcement contract:
        - exactly-once enforcement
        - exactly-once audit
        - zero Brain involvement
        """
        self.safety._apply_limits.assert_called_once()
        self.safety._audit_override.assert_called_once()
        self.safety.brain.predict.assert_not_called()

    # ------------------------------------------------------------------
    # Memory safety: lower + upper bound, atomic override
    # ------------------------------------------------------------------
    def test_memory_below_usage_is_forbidden(self):
        pred_mem = self.current_mem_usage - 1

        _, safe_mem = self.safety.validate_prediction(
            pred_cpu=1.0,
            pred_mem=pred_mem,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assert_mem_invariants(safe_mem)
        self.assert_atomic_override_path()

    # ------------------------------------------------------------------
    # CPU starvation protection
    # ------------------------------------------------------------------
    def test_negative_cpu_is_clamped_and_audited(self):
        safe_cpu, safe_mem = self.safety.validate_prediction(
            pred_cpu=-1.0,
            pred_mem=800,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assert_cpu_invariants(safe_cpu)
        self.assert_mem_invariants(safe_mem)
        self.assert_atomic_override_path()

    # ------------------------------------------------------------------
    # CPU surge protection
    # ------------------------------------------------------------------
    def test_cpu_above_policy_max_is_clamped(self):
        pred_cpu = self.config.MAX_CPU_CORES * 100

        safe_cpu, safe_mem = self.safety.validate_prediction(
            pred_cpu=pred_cpu,
            pred_mem=800,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assert_cpu_invariants(safe_cpu)
        self.assert_mem_invariants(safe_mem)
        self.assert_atomic_override_path()

    # ------------------------------------------------------------------
    # Adversarial / malformed inputs (isolated per case)
    # ------------------------------------------------------------------
    def test_malformed_inputs_cannot_bypass_safety(self):
        cases = [
            {"pred_cpu": -1, "pred_mem": 800},
            {"pred_cpu": 0, "pred_mem": 800},
            {"pred_cpu": float("nan"), "pred_mem": 800},
            {"pred_cpu": float("inf"), "pred_mem": 800},
            {"pred_cpu": 1.0, "pred_mem": -1},
            {"pred_cpu": 1.0, "pred_mem": None},
            {"pred_cpu": 1.0, "pred_mem": "100GB"},
        ]

        for case in cases:
            with self.subTest(case=case):
                self.safety._apply_limits.reset_mock()
                self.safety._audit_override.reset_mock()
                self.safety.brain.predict.reset_mock()

                safe_cpu, safe_mem = self.safety.validate_prediction(
                    pred_cpu=case["pred_cpu"],
                    pred_mem=case["pred_mem"],
                    current_mem_usage_mib=self.current_mem_usage,
                    current_mem_limit_mib=self.current_mem_limit,
                )

                self.assert_cpu_invariants(safe_cpu)
                self.assert_mem_invariants(safe_mem)
                self.assert_atomic_override_path()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    def test_exact_policy_minimums_are_allowed_without_override(self):
        self.safety._apply_limits.reset_mock()
        self.safety._audit_override.reset_mock()

        safe_cpu, safe_mem = self.safety.validate_prediction(
            pred_cpu=self.config.MIN_CPU_CORES,
            pred_mem=self.current_mem_usage + self.config.MEMORY_BUFFER_MIB,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assert_cpu_invariants(safe_cpu)
        self.assert_mem_invariants(safe_mem)

        self.safety._apply_limits.assert_not_called()
        self.safety._audit_override.assert_not_called()

    def test_just_below_minimum_forces_atomic_override(self):
        safe_cpu, _ = self.safety.validate_prediction(
            pred_cpu=self.config.MIN_CPU_CORES - 1e-9,
            pred_mem=800,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assertEqual(safe_cpu, self.config.MIN_CPU_CORES)
        self.assert_atomic_override_path()

    # ------------------------------------------------------------------
    # Regression traps
    # ------------------------------------------------------------------
    def test_removing_memory_buffer_breaks_contract(self):
        safe_cpu, safe_mem = self.safety.validate_prediction(
            pred_cpu=1.0,
            pred_mem=self.current_mem_usage,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assert_mem_invariants(safe_mem)
        self.assert_atomic_override_path()

    def test_enforcement_and_audit_are_non_bypassable(self):
        self.safety.validate_prediction(
            pred_cpu=-999,
            pred_mem=-999,
            current_mem_usage_mib=self.current_mem_usage,
            current_mem_limit_mib=self.current_mem_limit,
        )

        self.assert_atomic_override_path()


if __name__ == "__main__":
    unittest.main()
