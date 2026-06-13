"""Longest-prefix router (DESIGN.md §8.2, radix variant).

Drop-in sibling of `PrefixRouter`. Where `PrefixRouter` hashes a single block and
walks the consistent-hash ring, this routes to the worker that already caches the
**longest matching block-prefix** of the incoming prompt — the SGLang/vLLM
cache-aware behaviour — while keeping the same two guards:

* **Bounded load.** A worker is *eligible* only if it is under the live load cap
  (`cap = cap_factor · max(avg_load, 1)`, `inf` ⇒ pure affinity). The longest
  match is taken *among eligible workers*, so a hot prefix whose natural owner is
  over cap spills to the next-best holder (or, failing that, the ring) instead of
  piling on — bounded-load consistent hashing's property, preserved.
* **Cold-prefix spread.** A prompt sharing no cached prefix falls back to the
  consistent-hash ring (keyed on its first block), so cold traffic is still spread
  deterministically and an autoscale event still only remaps O(K/N) of it.

After each placement the chosen worker is recorded in the radix tree, so the
tree tracks (an LRU-bounded approximation of) what each worker's KV cache holds.
The tree carries no load state; this router owns the cap logic, mirroring how
`PrefixRouter` keeps load out of the ring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from statistics import mean

from relay_core.hashing import BoundedLoadConsistentHashRing
from relay_core.radix import RadixPrefixTree
from relay_core.types import WorkerState
from services.scheduler.router import PURE_AFFINITY, default_admit, default_load


class RadixPrefixRouter:
    def __init__(
        self,
        workers: Iterable[WorkerState],
        load_cap_factor: float = 1.25,
        vnodes: int = 160,
        cache_capacity: int = 0,
        load_fn: Callable[[WorkerState], float] = default_load,
        admit_fn: Callable[[WorkerState], bool] = default_admit,
    ) -> None:
        self.workers: dict[str, WorkerState] = {w.worker_id: w for w in workers}
        self.ring = BoundedLoadConsistentHashRing(self.workers.values(), vnodes=vnodes)
        self.tree = RadixPrefixTree(capacity=cache_capacity)
        self.cap_factor = float(load_cap_factor)
        self.load_fn = load_fn
        self.admit_fn = admit_fn

    # -- fleet membership -------------------------------------------------- #
    def add_worker(self, w: WorkerState) -> None:
        self.workers[w.worker_id] = w
        self.ring.add(w.worker_id)

    def remove_worker(self, worker_id: str) -> None:
        self.workers.pop(worker_id, None)
        self.ring.remove(worker_id)

    def capable(self, model: str) -> list[WorkerState]:
        return [w for w in self.workers.values() if w.has_model(model)]

    # -- placement --------------------------------------------------------- #
    def pick(self, model: str, blocks: tuple[str, ...]) -> WorkerState | None:
        capable = self.capable(model)
        if not capable:
            return None

        if self.cap_factor == PURE_AFFINITY:
            under_cap = {w.worker_id for w in capable}  # everyone eligible
        else:
            avg_load = mean(self.load_fn(w) for w in capable)
            cap = self.cap_factor * max(avg_load, 1.0)
            under_cap = {w.worker_id for w in capable if self.load_fn(w) < cap and self.admit_fn(w)}

        chosen: WorkerState | None = None
        if blocks and under_cap:
            wid, depth = self.tree.match_longest(blocks, is_eligible=under_cap.__contains__)
            if wid is not None and depth > 0:
                chosen = self.workers.get(wid)

        if chosen is None:
            # Cold prefix (no eligible cache holder) → consistent-hash ring,
            # walking from the first block's position and skipping capped workers.
            key = blocks[0] if blocks else ""
            for cand_id in self.ring.walk(key):
                w = self.workers.get(cand_id)
                if w is None or not w.has_model(model):
                    continue
                if self.cap_factor == PURE_AFFINITY:
                    chosen = w
                    break
                if self.load_fn(w) < cap and self.admit_fn(w):  # type: ignore[possibly-undefined]
                    chosen = w
                    break
            if chosen is None:
                return None  # everyone capped → backpressure; caller retries

        # Record the placement so the tree reflects this worker's warmed cache.
        if blocks:
            self.tree.insert(blocks, chosen.worker_id)
        return chosen

    # -- introspection ----------------------------------------------------- #
    def affinity_owner(self, blocks: tuple[str, ...]) -> str | None:
        """Longest-prefix owner ignoring load (the natural placement); falls back
        to the ring head when no prefix is cached. For tests / placement checks."""
        if blocks:
            wid, depth = self.tree.match_longest(blocks)
            if wid is not None and depth > 0:
                return wid
        return self.ring.head(blocks[0] if blocks else "")

    def matched_blocks(self, blocks: tuple[str, ...]) -> int:
        """How many leading blocks are already cached somewhere (the reuse depth
        a placement could achieve). Used as a routing-quality signal in benches."""
        _wid, depth = self.tree.match_longest(blocks)
        return depth
