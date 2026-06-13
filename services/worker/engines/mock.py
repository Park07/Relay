"""MockEngine — the calibrated benchmarking backbone (DESIGN.md §5.3, ADR-11).

Latency follows ``alpha + beta * b``:
  * ``alpha`` — fixed per-batch cost (kernel launch, dispatch, Python overhead),
  * ``beta``  — marginal per-item cost,
with multiplicative lognormal jitter so the tail is realistic rather than a
delta spike. The point of this engine is rigor you can check on paper:

    throughput(b) = b / (alpha + beta*b)   →  rises in b, asymptotes to 1/beta
    per-item latency falls with b; per-batch latency rises with b
                                          →  the textbook latency↔throughput knee

``alpha``/``beta`` are *calibrated* from a handful of real Ollama runs
(bench/calibrate.py) and baked in, so the mock is a disclosed simulation of
measured hardware, not a fantasy. That is what lets the whole control plane —
and every sweep in §13 — run on an 8GB Air, for free, reproducibly, while the
analytic curve keeps it honest.

Two execution surfaces, one latency law:
  * ``batch_latency_ms(b)`` — pure; the virtual-time simulator advances its clock
    by this. No real time passes, so a 50k-request sweep finishes in seconds.
  * ``infer(batch)``        — async; the real worker harness awaits the same
    latency, then streams ``ResultItem``s. Used on the live gRPC path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from relay_core.types import EngineStats, InferItem, ResultItem

# Defaults are placeholders; bench/calibrate.py overwrites these from real runs
# and writes the fitted constants to bench/results/calibration.json.
DEFAULT_ALPHA_MS = 18.0  # fixed per-batch cost
DEFAULT_BETA_MS = 7.5  # marginal per-item cost


class MockEngine:
    name = "mock"

    def __init__(
        self,
        alpha_ms: float = DEFAULT_ALPHA_MS,
        beta_ms: float = DEFAULT_BETA_MS,
        jitter_sigma: float = 0.15,
        seed: int | None = None,
        loaded_models: tuple[str, ...] = ("qwen2.5:0.5b",),
    ) -> None:
        self.alpha_ms = float(alpha_ms)
        self.beta_ms = float(beta_ms)
        self.jitter_sigma = float(jitter_sigma)
        self._rng = np.random.default_rng(seed)
        self._loaded = loaded_models
        self._batches = 0
        self._items = 0

    # -- the latency law (single source of truth) -------------------------- #
    def _base_ms(self, batch_size: int) -> float:
        return self.alpha_ms + self.beta_ms * batch_size

    def _jitter(self) -> float:
        if self.jitter_sigma <= 0:
            return 1.0
        # Median-1 multiplicative lognormal: exp(N(-s^2/2, s^2)) so the *mean*
        # is ~1 and we get a right tail rather than a symmetric one.
        s = self.jitter_sigma
        return float(np.exp(self._rng.normal(-0.5 * s * s, s)))

    def batch_latency_ms(self, batch_size: int) -> float:
        """Pure latency of running a batch of ``batch_size`` items."""
        if batch_size <= 0:
            return 0.0
        return self._base_ms(batch_size) * self._jitter()

    def throughput_analytic(self, batch_size: int) -> float:
        """Closed-form req/s used to validate the measured curve in §13.2."""
        if batch_size <= 0:
            return 0.0
        return 1000.0 * batch_size / self._base_ms(batch_size)

    # -- engine protocol --------------------------------------------------- #
    async def load(self, model: str) -> None:
        if model not in self._loaded:
            self._loaded = (*self._loaded, model)

    async def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]:
        lat = self.batch_latency_ms(len(batch))
        self._batches += 1
        self._items += len(batch)
        await asyncio.sleep(lat / 1000.0)
        bid = batch[0].request_id if batch else ""
        for item in batch:
            yield ResultItem(
                request_id=item.request_id,
                batch_id=bid,
                output=f"[mock:{self.name}] {item.input[:24]}",
                final=True,
                inference_ms=lat,
                batch_size=len(batch),
                gpu_util=min(1.0, len(batch) / 16.0),  # coarse, illustrative
                cache_hit=False,
            )

    def stats(self) -> EngineStats:
        return EngineStats(
            gpu_util=0.0,
            loaded_models=self._loaded,
        )
