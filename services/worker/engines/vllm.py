"""VLLMEngine — the CUDA-only validation backend (DESIGN.md §5.3, §13.2 exp. 4).

This engine exists to *confirm* on real hardware the prefix-routing win that the
CacheAwareMockEngine demonstrates locally, and to produce the real tokens/s and
scrapeable per-worker cache-hit numbers. It is the only engine that runs solely
on a CUDA cloud GPU.

Two design-doc rules are baked in:

  1. **vLLM does its own continuous batching.** Relay must NOT hand vLLM large
     pre-formed batches (that fights PagedAttention's scheduler). So this engine
     sets ``prefers_individual_dispatch = True`` and submits items one-by-one,
     letting vLLM merge them. Relay's contribution here is *inter-worker*
     placement + autoscaling; vLLM's is *intra-worker* batching — they compose.
  2. **Prefix-caching flags/metric names drift.** The design doc explicitly says
     to verify vLLM's current ``enable_prefix_caching`` flag and the exact
     cache-hit metric names on arrival. The hooks below are marked accordingly.

NEEDS CUDA + vLLM. This is the validation step that runs on a rented multi-GPU
box, not in CI and not on the Mac. Excluded from the default import path.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

from relay_core.types import EngineStats, InferItem, ResultItem


class VLLMEngine:
    name = "vllm"
    # Tell the harness to dispatch items individually (see rule 1 above).
    prefers_individual_dispatch = True

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model = model
        self._llm = None
        self._loaded: tuple[str, ...] = ()
        self._cache_hits = 0
        self._cache_misses = 0

    async def load(self, model: str) -> None:
        # VERIFY ON ARRIVAL (DESIGN.md §5.3): the prefix-caching flag name and any
        # metrics flags evolve across vLLM versions. As of writing this is
        # ``enable_prefix_caching=True``; confirm against the installed version.
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore

        self.model = model or self.model
        args = AsyncEngineArgs(
            model=self.model,
            enable_prefix_caching=True,  # <-- verify flag name/behaviour on arrival
            disable_log_stats=False,     # we want the cache-hit counters
        )
        self._llm = AsyncLLMEngine.from_engine_args(args)
        self._loaded = (self.model,)

    async def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]:
        # Per rule 1, submit each item to vLLM independently; PagedAttention does
        # the continuous batching and automatic prefix-cache reuse internally.
        from vllm import SamplingParams  # type: ignore

        assert self._llm is not None, "call load() first"
        bid = batch[0].request_id if batch else ""
        for item in batch:
            started = time.monotonic()
            sp = SamplingParams(
                max_tokens=item.params.max_tokens,
                temperature=item.params.temperature,
                top_p=item.params.top_p,
            )
            async for out in self._llm.generate(item.input, sp, request_id=item.request_id):
                finished = out.finished
                text = out.outputs[0].text if out.outputs else ""
                # VERIFY ON ARRIVAL: read cache-hit off vLLM's metrics rather than
                # this placeholder once the exact field name is confirmed.
                hit = getattr(out, "num_cached_tokens", 0) > 0
                yield ResultItem(
                    request_id=item.request_id,
                    batch_id=bid,
                    token=None if finished else text,
                    output=text if finished else None,
                    final=finished,
                    inference_ms=(time.monotonic() - started) * 1000.0,
                    batch_size=1,  # individual dispatch
                    cache_hit=hit,
                )
                if finished:
                    self._cache_hits += int(hit)
                    self._cache_misses += int(not hit)

    def stats(self) -> EngineStats:
        # VERIFY ON ARRIVAL: scrape real gpu_util + cache counters from vLLM's
        # Prometheus stats logger; these locals are a stand-in until then.
        return EngineStats(
            loaded_models=self._loaded,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )
