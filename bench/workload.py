"""Zipfian-over-shared-prefixes workload generator (DESIGN.md §13.1).

This is where most of the intellectual work lives. Prefix-aware routing does
**nothing** on uncorrelated random prompts — you would get a beautiful null
result. Real LLM traffic shares prefixes: a handful of system prompts, few-shot
blocks, and RAG documents are very hot; a long tail are rare. We model that with
a **finite Zipfian distribution over a pool of shared prefixes**.

Parameters reported in the benchmark (so a reviewer can judge the skew honestly):
  * ``pool_size``   — number of distinct shared prefixes,
  * ``skew`` (s)    — Zipf exponent; larger = more concentrated on a few prefixes,
  * ``prefix_chars``— length of the shared block (the part caching saves),
  * ``suffix_chars``— length of the unique per-request tail.

We sample the prefix *rank* by exact inverse-CDF over P(rank k) ∝ 1/k^s for
k = 1..pool_size, so the reported ``s`` is exactly the distribution used — not
numpy's unbounded ``zipf`` (the zeta distribution), which is a different and
harder-to-report object. The request's ``prefix_hash`` is set directly from the
shared block, so it is independent of the unique suffix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relay_core.types import InferItem, InferParams, Priority, prefix_hash_of

# A few plausible prefix "templates" so generated inputs read like real traffic.
# Content is irrelevant to the hash (we hash the shared block explicitly); this
# is purely so the demo prompts aren't gibberish.
_TEMPLATES = [
    "You are a helpful assistant specializing in {topic}. Follow the policy: "
    "be concise, cite sources, never speculate. Few-shot examples follow. ",
    "SYSTEM: Translate the user's request about {topic} into a structured plan. "
    "Use the schema provided in the appendix. Context document: ",
    "[RAG] Retrieved document on {topic}. Section 1: background. Section 2: "
    "methodology. Section 3: findings. Use only this document to answer. ",
]
_TOPICS = [
    "billing", "networking", "kubernetes", "genomics", "tax law", "robotics",
    "logistics", "semiconductors", "climate", "linguistics", "finance", "biology",
]


@dataclass
class WorkloadParams:
    pool_size: int = 200
    skew: float = 1.1
    prefix_chars: int = 800
    suffix_chars: int = 48
    seed: int | None = 7

    def as_dict(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "skew": self.skew,
            "prefix_chars": self.prefix_chars,
            "suffix_chars": self.suffix_chars,
            "seed": self.seed,
        }


class ZipfianPrefixWorkload:
    def __init__(self, params: WorkloadParams | None = None) -> None:
        self.p = params or WorkloadParams()
        self._rng = np.random.default_rng(self.p.seed)

        # Finite Zipf over ranks 1..pool_size, sampled by inverse CDF.
        ranks = np.arange(1, self.p.pool_size + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.p.skew)
        self._pmf = weights / weights.sum()
        self._cdf = np.cumsum(self._pmf)

        self._prefixes = [self._make_prefix(i) for i in range(self.p.pool_size)]
        self._prefix_hashes = [prefix_hash_of(pf) for pf in self._prefixes]
        self._counter = 0
        self.draw_counts = np.zeros(self.p.pool_size, dtype=np.int64)

    # -- prefix construction ---------------------------------------------- #
    def _make_prefix(self, idx: int) -> str:
        tmpl = _TEMPLATES[idx % len(_TEMPLATES)]
        topic = _TOPICS[idx % len(_TOPICS)]
        seed = tmpl.format(topic=f"{topic} (variant {idx})")
        # Pad/truncate to the configured shared-block length.
        if len(seed) < self.p.prefix_chars:
            seed = (seed * (self.p.prefix_chars // len(seed) + 1))[: self.p.prefix_chars]
        return seed[: self.p.prefix_chars]

    def _random_suffix(self) -> str:
        n = self.p.suffix_chars
        # ASCII letters/digits, cheap and unique enough.
        chars = self._rng.integers(48, 122, size=n)
        return "".join(chr(int(c)) for c in chars)

    def _sample_index(self) -> int:
        u = float(self._rng.random())
        return int(np.searchsorted(self._cdf, u, side="right"))

    # -- generation -------------------------------------------------------- #
    def next(self, priority: Priority = Priority.DEFAULT) -> InferItem:
        idx = self._sample_index()
        self.draw_counts[idx] += 1
        self._counter += 1
        prefix = self._prefixes[idx]
        item = InferItem(
            request_id=f"req-{self._counter:08d}",
            input=prefix + self._random_suffix(),
            params=InferParams(max_tokens=128, stream=False),
            prefix_hash=self._prefix_hashes[idx],
            priority=priority,
        )
        return item

    def generate(self, n: int, high_fraction: float = 0.0) -> list[InferItem]:
        out: list[InferItem] = []
        for _ in range(n):
            prio = (
                Priority.HIGH
                if high_fraction > 0 and self._rng.random() < high_fraction
                else Priority.DEFAULT
            )
            out.append(self.next(prio))
        return out

    # -- reporting --------------------------------------------------------- #
    def theoretical_topk_mass(self, k: int) -> float:
        return float(self._pmf[:k].sum())

    def realized_topk_mass(self, k: int) -> float:
        total = self.draw_counts.sum()
        if total == 0:
            return 0.0
        order = np.argsort(self.draw_counts)[::-1]
        return float(self.draw_counts[order[:k]].sum() / total)

    def distinct_drawn(self) -> int:
        return int((self.draw_counts > 0).sum())
