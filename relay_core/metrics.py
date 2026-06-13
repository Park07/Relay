"""Prometheus metrics — the exact names and labels from DESIGN.md §15.

The entire résumé value of Relay is *measurement* (ADR-10: metrics from commit
#1, never bolted on), so the metric names are part of the contract and live in
one place. Framing is **RED** for the gateway (Rate/Errors/Duration) and **USE**
for workers (Utilization/Saturation/Errors).

``relay_prefix_cache_hit_ratio`` and ``relay_worker_load_imbalance`` are the two
axes of the §13 Pareto frontier; ``relay_queue_depth`` is the signal the HPA
autoscales on (DESIGN.md ADR-9).

prometheus-client is an optional dependency; if it is absent (e.g. in the
in-process benchmark) we fall back to no-op stand-ins so importing this module
never fails. The live services depend on it for real.
"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore

    _PROM = True
except Exception:  # pragma: no cover - exercised only when dep is absent
    _PROM = False

    class _Noop:
        def __init__(self, *a, **k) -> None: ...
        def labels(self, *a, **k) -> _Noop:
            return self

        def inc(self, *a, **k) -> None: ...
        def dec(self, *a, **k) -> None: ...
        def set(self, *a, **k) -> None: ...
        def observe(self, *a, **k) -> None: ...

    Counter = Gauge = Histogram = _Noop  # type: ignore


# -- Gateway: RED ----------------------------------------------------------- #
REQUEST_TOTAL = Counter(
    "relay_request_total", "Request rate + error rate (RED).", ["model", "status"]
)
REQUEST_DURATION = Histogram(
    "relay_request_duration_seconds",
    "End-to-end latency → p50/p95/p99.",
    ["model", "mode"],
)

# -- Scheduler -------------------------------------------------------------- #
QUEUE_WAIT = Histogram(
    "relay_queue_wait_seconds", "How long batching makes a request wait.", ["model"]
)
QUEUE_DEPTH = Gauge(
    "relay_queue_depth", "Pending items — DRIVES AUTOSCALING (HPA).", ["model", "priority"]
)
BATCH_SIZE = Histogram(
    "relay_batch_size", "Realized batch sizes — are we filling batches?", ["model"]
)

# -- The depth-feature headline + its sibling axis -------------------------- #
PREFIX_CACHE_HIT_RATIO = Gauge(
    "relay_prefix_cache_hit_ratio",
    "Locality achieved — the depth-feature headline.",
    ["worker_id", "policy"],
)
WORKER_LOAD_IMBALANCE = Gauge(
    "relay_worker_load_imbalance",
    "Spread across workers — the other axis of the frontier.",
    [],
)

# -- Workers: USE ----------------------------------------------------------- #
WORKER_INFLIGHT = Gauge("relay_worker_inflight", "Saturation (USE).", ["worker_id"])
WORKER_GPU_UTIL = Gauge(
    "relay_worker_gpu_util", "Utilization (USE) — credible only on CUDA.", ["worker_id"]
)
TOKENS_PER_SECOND = Gauge("relay_tokens_per_second", "LLM throughput.", ["worker_id", "model"])

__all__ = [
    "REQUEST_TOTAL",
    "REQUEST_DURATION",
    "QUEUE_WAIT",
    "QUEUE_DEPTH",
    "BATCH_SIZE",
    "PREFIX_CACHE_HIT_RATIO",
    "WORKER_LOAD_IMBALANCE",
    "WORKER_INFLIGHT",
    "WORKER_GPU_UTIL",
    "TOKENS_PER_SECOND",
]
