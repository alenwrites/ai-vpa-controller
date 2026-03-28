# prometheus_client.py
"""
Production-grade Prometheus ingestion client for ML-driven VPA.

Key guarantees:
- Deterministic (EXPECTED_POINTS, FEATURE_COUNT) output
- Vectorized, loop-free data alignment and interpolation
- Explicit feature/model contract coupling (aggregate vs per-pod semantics)
- Fail-fast on ambiguity or unsafe assumptions

Any change to features, queries, or aggregation semantics MUST be done here.
Hot-patching via config or environment variables is intentionally forbidden.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Callable, Optional
from enum import Enum
from types import MappingProxyType

from errors import TransientError

import requests
import numpy as np
import pandas as pd

import config

import time
from prometheus_client import Histogram, Counter

# Define internal telemetry metrics
# This allows us to see how long Prometheus takes to respond
PROM_SCRAPE_LATENCY = Histogram(
    'vpa_prometheus_scrape_duration_seconds',
    'Time spent performing Prometheus queries',
    ['query_type']
)

PROM_SCRAPE_ERRORS = Counter(
    'vpa_prometheus_scrape_errors_total',
    'Total number of failed Prometheus scrapes',
    ['query_type']
)



log = logging.getLogger("vpa.prom_client")


class PrometheusIngestionError(RuntimeError):
    pass


class CircuitBreakerOpen(RuntimeError):
    pass


class Aggregation(Enum):
    SUM = "sum"
    AVG = "avg"
    MAX = "max"


class MetricSemantics(Enum):
    AGGREGATE = "aggregate"        # model trained on service-level load
    REPRESENTATIVE = "per_pod"     # model trained on per-pod behavior


# ------------------------------------------------------------------------------
# Centralized, immutable feature contract.
# Queries are templated; aggregation is injected at runtime based on model mode.
# ------------------------------------------------------------------------------

_FEATURE_DEFINITIONS = {
    "cpu_usage": {
        "query": '{agg}(rate(container_cpu_usage_seconds_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "mem_usage": {
        "query": '{agg}(container_memory_working_set_bytes{{pod=~"{target_name}-.*",namespace="{namespace}"}})',
        "transform": lambda x: x / (1024.0 ** 2),
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "cpu_request": {
        "query": '{agg}(kube_pod_container_resource_requests{{pod=~"{target_name}-.*",namespace="{namespace}",resource="cpu"}})',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "cpu_limit": {
        "query": '{agg}(kube_pod_container_resource_limits{{pod=~"{target_name}-.*",namespace="{namespace}",resource="cpu"}})',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "mem_request": {
        "query": '{agg}(kube_pod_container_resource_requests{{pod=~"{target_name}-.*",namespace="{namespace}",resource="memory"}})',
        "transform": lambda x: x / (1024.0 ** 2),
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "mem_limit": {
        "query": '{agg}(kube_pod_container_resource_limits{{pod=~"{target_name}-.*",namespace="{namespace}",resource="memory"}})',
        "transform": lambda x: x / (1024.0 ** 2),
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "node_cpu_usage": {
        "query": 'sum(rate(node_cpu_seconds_total{{mode!="idle"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "node_cpu_steal": {
        "query": 'sum(rate(node_cpu_seconds_total{{mode="steal"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "node_mem_available": {
        "query": 'sum(node_memory_MemAvailable_bytes)',
        "transform": lambda x: x / (1024.0 ** 2),
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "replicas": {
        "query": 'count(kube_pod_container_info{{pod=~"{target_name}-.*",namespace="{namespace}"}})',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "latency": {
        "query": 'histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{pod=~"{target_name}-.*"}}[{step}s])))',
        "transform": lambda x: x,
        "interpolation": "linear",
        "default_aggregation": Aggregation.AVG,
        "semantics": MetricSemantics.REPRESENTATIVE,
    },
    "cpu_throttling": {
        "query": '{agg}(rate(container_cpu_cfs_throttled_seconds_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "throttled_periods": {
        "query": '{agg}(rate(container_cpu_cfs_throttled_periods_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "net_receive": {
        "query": '{agg}(rate(container_network_receive_bytes_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x / 1024.0,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "net_receive": {
        "query": '{agg}(rate(container_network_receive_bytes_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x / 1024.0,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "net_transmit": {
        "query": '{agg}(rate(container_network_transmit_bytes_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x / 1024.0,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "pod_restarts": {
        "query": '{agg}(changes(kube_pod_container_status_restarts_total{{pod=~"{target_name}-.*",namespace="{namespace}"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "oom_kills": {
        "query": '{agg}(kube_pod_container_status_last_terminated_reason{{pod=~"{target_name}-.*",namespace="{namespace}",reason="OOMKilled"}})',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "http_requests_total": {
        "query": 'sum(rate(http_requests_total{{pod=~"{target_name}-.*"}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "ops_per_sec": {
        "query": 'sum(rate(container_cpu_usage_seconds_total{{pod=~"{target_name}-.*"}}[{step}s])) * 100',
        "transform": lambda x: x,
        "interpolation": "linear",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    },
    "errors_5xx": {
        "query": 'sum(rate(http_requests_total{{pod=~"{target_name}-.*",status=~"5.."}}[{step}s]))',
        "transform": lambda x: x,
        "interpolation": "step",
        "default_aggregation": Aggregation.SUM,
        "semantics": MetricSemantics.AGGREGATE,
    }
}

FEATURE_DEFINITIONS: Dict[str, Dict[str, object]] = MappingProxyType(
    _FEATURE_DEFINITIONS
)


class PrometheusClient:
    MAX_CONSECUTIVE_FAILURES = 5
    CIRCUIT_BREAKER_RESET_SECONDS = 60
    SCRAPE_DELAY_BUFFER = getattr(config, "SCRAPE_DELAY_BUFFER", 10)

    def __init__(self) -> None:
        self.url = config.PROMETHEUS_URL.rstrip("/")
        self.feature_order = list(config.FEATURE_ORDER)

        self.window_seconds = config.WINDOW_SECONDS
        self.step_seconds = config.STEP_SECONDS
        self.expected_points = self.window_seconds // self.step_seconds

        self.model_semantics = MetricSemantics(
            getattr(config, "MODEL_SEMANTICS", "aggregate")
        )

        if set(self.feature_order) != set(FEATURE_DEFINITIONS.keys()):
            raise ValueError(
                "FEATURE_ORDER must exactly match FEATURE_DEFINITIONS keys. "
                "This coupling is strict by design."
            )

        self._failure_count = 0
        self._circuit_opened_at: Optional[float] = None

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(500, 502, 503, 504),
                allowed_methods=("GET",),
            )
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.timeouts = (2.0, 5.0)

    # --------------------------------------------------------------------------

    def get_model_input(self, target_name: str, namespace: str = "default") -> np.ndarray:
        self._check_circuit_breaker()
        start_time = time.perf_counter() # FIX: Define start_time here

        try:
            effective_end = time.time() - self.SCRAPE_DELAY_BUFFER
            effective_start = effective_end - self.window_seconds

            # 1. Initialize with Zeros (Prevents garbage data if a fetch fails)
            matrix = np.zeros(
                (self.expected_points, len(self.feature_order)), dtype=np.float64
            )

            # 2. Strict Ordered Loop (Mirroring training columns)
            for idx, feature in enumerate(self.feature_order):
                # This lookup ensures that index 'idx' matches the training dataset
                definition = FEATURE_DEFINITIONS[feature]

                agg = definition["default_aggregation"]
                if (
                    self.model_semantics is MetricSemantics.REPRESENTATIVE
                    and definition["semantics"] is MetricSemantics.AGGREGATE
                ):
                    agg = Aggregation.AVG

                query = definition["query"].format(
                    agg=agg.value,
                    target_name=target_name,
                    namespace=namespace,
                    step=self.step_seconds,
                )

                log.debug("Fetching feature index %d: %s", idx, feature)

                # Fetch and align data for this specific column
                values = self._fetch_align_and_process(
                    query=query,
                    start=effective_start,
                    end=effective_end,
                    transform=definition["transform"],
                    interpolation=definition["interpolation"],
                )
                
                # INSERT strictly at the column index matching FEATURE_ORDER
                matrix[:, idx] = values

            self._reset_failures()

            # Record success telemetry
            duration = time.perf_counter() - start_time
            PROM_SCRAPE_LATENCY.labels(query_type="matrix").observe(duration)

            return matrix

        except Exception as exc:
            PROM_SCRAPE_ERRORS.labels(query_type="matrix").inc()
            self._record_failure()
            raise TransientError("Prometheus matrix scrape failed") from exc

    # --------------------------------------------------------------------------

    def _fetch_align_and_process(
        self,
        query: str,
        start: float,
        end: float,
        transform: Callable[[np.ndarray], np.ndarray],
        interpolation: str,
    ) -> np.ndarray:
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": f"{self.step_seconds}s",
        }

        response = self.session.get(
            f"{self.url}/api/v1/query_range",
            params=params,
            timeout=self.timeouts,
        )
        response.raise_for_status()

        payload = response.json()
        if payload.get("status") != "success":
            raise PrometheusIngestionError(payload)

        results = payload.get("data", {}).get("result", [])

        # 1. Theoretical timestamp grid (Build this first)
        end_grid = int(end // self.step_seconds) * self.step_seconds
        grid = (
            end_grid
            - np.arange(self.expected_points - 1, -1, -1) * self.step_seconds
        ).astype(float)

        # 2. Resilient Data Handling
        if not results:
            # If Prometheus finds nothing, return a window of zeros.
            # This allows the GRU to keep running even if a metric is temporarily missing.
            log.warning(f"Empty result for query: {query[:50]}... returning zeros.")
            return transform(np.zeros(self.expected_points))

        if len(results) > 1:
            # If multiple series return, we log an error but take the first one
            # to avoid crashing the whole controller.
            log.error(f"Ambiguous result: {len(results)} series returned. Using the first one.")
            raw_values = results[0].get("values", [])
        else:
            raw_values = results[0].get("values", [])

        if not raw_values:
            return transform(np.zeros(self.expected_points))

        # 3. Alignment and Interpolation
        aligned = self._align_series_to_grid(raw_values, grid)
        filled = self._fill_missing(aligned, interpolation)

        return transform(filled.astype(np.float64))

    # --------------------------------------------------------------------------

    def _align_series_to_grid(self, raw_values, grid: np.ndarray) -> np.ndarray:
        aligned = np.full(len(grid), np.nan, dtype=np.float64)

        ts = np.array([float(t) for t, _ in raw_values])
        vals = np.array([float(v) for _, v in raw_values])

        indices = np.searchsorted(grid, ts)
        indices = np.clip(indices, 0, len(grid) - 1)

        deltas = np.abs(grid[indices] - ts)
        mask = deltas <= (0.5 * self.step_seconds)

        aligned[indices[mask]] = vals[mask]
        return aligned

    def _fill_missing(self, values: np.ndarray, strategy: str) -> np.ndarray:
        """
        Fill missing values with semantic-aware interpolation.
        If the entire window is empty, returns a zeroed array to prevent crashes.
        """
        # 1. Check if we have ANY data to interpolate
        mask = ~np.isnan(values)
        if mask.sum() == 0:
            # If the whole window is NaN, we assume the metric is just idle/zero.
            # This prevents the "Unfillable gaps" crash.
            return np.zeros_like(values)

        # 2. If we have some data, proceed with interpolation
        s = pd.Series(values)

        if strategy == "linear":
            # Best for memory: connects the dots in a straight line
            filled = s.interpolate(method="linear", limit_direction="both")
        elif strategy == "step":
            # Best for CPU: stays at the last known value until a new one appears
            filled = s.ffill().bfill()
        else:
            raise PrometheusIngestionError(f"Unknown interpolation strategy '{strategy}'")

        # 3. Final safety check: if we still have NaNs (should be impossible now)
        if filled.isna().any():
            return filled.fillna(0).to_numpy()

        return filled.to_numpy()

    # --------------------------------------------------------------------------

    def _record_failure(self) -> None:
        if self._circuit_opened_at is not None:
            return
        self._failure_count += 1
        if self._failure_count >= self.MAX_CONSECUTIVE_FAILURES:
            self._circuit_opened_at = time.time()
            log.error("Circuit breaker opened due to repeated ingestion failures")

    def _reset_failures(self) -> None:
        self._failure_count = 0
        self._circuit_opened_at = None

    def _check_circuit_breaker(self) -> None:
        if self._circuit_opened_at is None:
            return
        if time.time() - self._circuit_opened_at > self.CIRCUIT_BREAKER_RESET_SECONDS:
            self._reset_failures()
            return
        raise CircuitBreakerOpen("Prometheus ingestion circuit breaker is open")
   # ---------------------------------------------------------------------------

   def get_current_metrics(self, target_name: str, namespace: str) -> dict:
        """
        Fetches instant 'spot' metrics for the Safety Layer.
        """
        start_time = time.perf_counter()

        queries = {
            "rss_mib": f'sum(container_memory_rss{{pod=~"{target_name}-.*",namespace="{namespace}"}})',
            "limit_mib": f'sum(container_spec_memory_limit_bytes{{pod=~"{target_name}-.*",namespace="{namespace}"}})'
        }

        results = {}

        try:
            for key, query in queries.items():
                # Note: We use /api/v1/query (instant) not /api/v1/query_range here
                response = self.session.get(
                    f"{self.url}/api/v1/query",
                    params={"query": query},
                    timeout=self.timeouts
                )
                response.raise_for_status()
                data = response.json().get("data", {}).get("result", [])

                if data:
                    # Convert bytes to MiB
                    bytes_val = float(data[0]["value"][1])
                    results[key] = int(bytes_val / (1024 * 1024))
                else:
                    results[key] = 0

            # SUCCESS: Record latency after BOTH metrics are collected
            duration = time.perf_counter() - start_time
            PROM_SCRAPE_LATENCY.labels(query_type="instant").observe(duration)

            return results

        except Exception as e:
            # FAILURE: Increment error counter
            PROM_SCRAPE_ERRORS.labels(query_type="instant").inc()
            log.error(f"Failed to fetch instant metrics: {e}")
            # Raising this ensures the Safety Layer doesn't get bad/incomplete data
            raise TransientError("Prometheus current-metrics scrape failed") from e
