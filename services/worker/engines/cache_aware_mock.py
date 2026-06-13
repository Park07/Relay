"""CacheAwareMockEngine — the engine the depth feature is measured on
(DESIGN.md §5.3, ADR-11/ADR-12).

It extends the MockEngine's ``alpha + beta*b`` decode law with the one thing the
plain mock can't express: a **per-worker KV cache** keyed on prefix hash.

  * **Hit**  (prefix already resident): the long prefill is skipped → cheap.
  * **Miss** (prefix not resident): pay the full ``prefill_ms`` to build the
    prefix's KV, then insert it (evicting LRU if the cache is full).

Latency of a batch:

    base(b)  = alpha + beta*b                      # decode, as in MockEngine
    prefill  = prefill_ms * (# distinct *missed* prefixes in the batch)
    total    = (base(b) + prefill) * jitter

Why this surfaces the routing result: each worker has its *own* cache. If the
router sends same-prefix requests to the same worker (affinity), that worker's
hit-rate climbs and it stops paying ``prefill_ms``. Round-robin scatters the
same prefixes across every worker, so *each* worker keeps missing and re-paying
the prefill — cache-hit-rate collapses and p99 balloons. Sweeping the router's
``load_cap_factor`` between those extremes traces the Pareto frontier (§13.2).

This is a *model* of a KV cache, not a real one; vLLM on CUDA confirms it later
(§13.2 experiment 4). The model is deliberately simple and disclosed.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator

import numpy as np

from relay_core.types import EngineStats, InferItem, ResultItem

DEFAULT_ALPHA_MS = 18.0
DEFAULT_BETA_MS = 7.5
# Prefill dominates for long shared prefixes — that asymmetry is the whole point.
# Calibrated alongside alpha/beta (a long system prompt / RAG block is many
# tokens of prefill); default chosen so a miss is ~10x a hit at b=1.
DEFAULT_PREFILL_MS = 160.0
# How many distinct prefixes one worker's KV cache can hold before eviction.
DEFAULT_CACHE_CAPACITY = 64


class CacheAwareMockEngine:
    name = "cache-aware-mock"

    def __init__(
        self,
        alpha_ms: float = DEFAULT_ALPHA_MS,
        beta_ms: float = DEFAULT_BETA_MS,
        prefill_ms: float = DEFAULT_PREFILL_MS,
        cache_capacity: int = DEFAULT_CACHE_CAPACITY,
        jitter_sigma: float = 0.15,
        seed: int | None = None,
        loaded_models: tuple[str, ...] = ("qwen2.5:0.5b",),
    ) -> None:
        self.alpha_ms = float(alpha_ms)
        self.beta_ms = float(beta_ms)
        self.prefill_ms = float(prefill_ms)
        self.cache_capacity = int(cache_capacity)
        self.jitter_sigma = float(jitter_sigma)
        self._rng = np.random.default_rng(seed)
        self._loaded = loaded_models
        # OrderedDict as an LRU set: key = prefix_hash, value unused.
        self._cache: OrderedDict[str, None] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    # -- cache mechanics --------------------------------------------------- #
    def _resident(self, prefix_hash: str) -> bool:
        return prefix_hash in self._cache

    def _touch(self, prefix_hash: str) -> None:
        """Mark resident + most-recently-used, evicting LRU past capacity."""
        if prefix_hash in self._cache:
            self._cache.move_to_end(prefix_hash)
            return
        self._cache[prefix_hash] = None
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)  # evict least-recently-used

    def _jitter(self) -> float:
        if self.jitter_sigma <= 0:
            return 1.0
        s = self.jitter_sigma
        return float(np.exp(self._rng.normal(-0.5 * s * s, s)))

    # -- the latency law + cache update (single source of truth) ----------- #
    def run_batch(self, batch: list[InferItem]) -> tuple[float, list[bool]]:
        """Compute batch latency and per-item hit/miss, and update the cache.

        Returns ``(latency_ms, per_item_hit)``. Pure w.r.t. time — the simulator
        advances its clock by ``latency_ms``; the cache mutation is the engine's
        only state. Called by both the simulator and the async ``infer`` path.
        """
        if not batch:
            return 0.0, []

        # Resident status is evaluated at batch *start* for every item, so all
        # items sharing a not-yet-resident prefix in the same batch see a miss
        # exactly once (we prefill it once).
        per_item_hit: list[bool] = [self._resident(it.prefix_hash) for it in batch]

        distinct_missed = {
            it.prefix_hash for it, hit in zip(batch, per_item_hit, strict=False) if not hit
        }
        for it in batch:
            self._touch(it.prefix_hash)  # everything is resident after this batch

        base = self.alpha_ms + self.beta_ms * len(batch)
        prefill = self.prefill_ms * len(distinct_missed)
        latency = (base + prefill) * self._jitter()

        hits = sum(per_item_hit)
        self.cache_hits += hits
        self.cache_misses += len(batch) - hits
        return latency, per_item_hit

    # -- engine protocol --------------------------------------------------- #
    async def load(self, model: str) -> None:
        if model not in self._loaded:
            self._loaded = (*self._loaded, model)

    async def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]:
        latency, per_item_hit = self.run_batch(batch)
        await asyncio.sleep(latency / 1000.0)
        bid = batch[0].request_id if batch else ""
        for item, hit in zip(batch, per_item_hit, strict=False):
            yield ResultItem(
                request_id=item.request_id,
                batch_id=bid,
                output=f"[{self.name}] {item.input[:24]}",
                final=True,
                inference_ms=latency,
                batch_size=len(batch),
                cache_hit=hit,
            )

    def stats(self) -> EngineStats:
        return EngineStats(
            loaded_models=self._loaded,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )
