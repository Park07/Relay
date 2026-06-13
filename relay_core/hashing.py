"""A consistent-hash ring with virtual nodes.

This is the substrate for prefix-aware routing (DESIGN.md §8.2). The ring gives a
*stable* mapping from a prefix hash to an ordered preference list of workers:
walking the ring clockwise from a prefix's position yields the worker that should
own that prefix, then its successors. Two properties matter:

1. **Affinity.** The same prefix hash always maps to the same head worker (until
   the fleet changes), so its KV cache gets reused.
2. **Minimal disruption.** Adding/removing a worker only remaps the keys that
   fall in the arc that worker covers — O(K/N) keys move, not all of them. That
   is why this beats ``hash(prefix) % N`` (which remaps almost everything on a
   fleet change and would cold-flush every cache on every autoscale event).

The *bounded-load* part (don't pile a hot prefix onto one worker) is not encoded
in the ring itself — it lives in ``PrefixRouter.pick`` (services/scheduler/router.py),
which walks this ring and skips workers over their live load cap. Keeping the ring
free of live-load state keeps it deterministic and trivially testable.
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Iterable, Iterator

from relay_core.types import WorkerState


def _hash(key: str) -> int:
    # 64-bit positions are plenty for a fleet of tens of workers; blake2b is fast
    # and well-distributed (better spread than md5, no security claim needed).
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


class BoundedLoadConsistentHashRing:
    """Maps prefix hashes to an ordered worker preference list.

    ``vnodes`` virtual nodes per worker smooths the arc sizes so load is even
    *in expectation*; with too few vnodes one worker can own a disproportionate
    arc and skew the no-cap baseline. 100–200 is the usual sweet spot.
    """

    def __init__(self, workers: Iterable[WorkerState], vnodes: int = 160) -> None:
        self.vnodes = vnodes
        self._ring: dict[int, str] = {}  # position -> worker_id
        self._sorted_positions: list[int] = []
        self._worker_ids: list[str] = []
        for w in workers:
            self.add(w.worker_id)

    # -- mutation ---------------------------------------------------------- #
    def add(self, worker_id: str) -> None:
        if worker_id in self._worker_ids:
            return
        self._worker_ids.append(worker_id)
        for v in range(self.vnodes):
            pos = _hash(f"{worker_id}#{v}")
            # Collisions are astronomically unlikely at 64 bits; on the off
            # chance, last writer wins — harmless, just one fewer vnode.
            self._ring[pos] = worker_id
        self._reindex()

    def remove(self, worker_id: str) -> None:
        if worker_id not in self._worker_ids:
            return
        self._worker_ids.remove(worker_id)
        self._ring = {p: wid for p, wid in self._ring.items() if wid != worker_id}
        self._reindex()

    def _reindex(self) -> None:
        self._sorted_positions = sorted(self._ring)

    # -- query ------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._worker_ids)

    def walk(self, prefix_hash: str) -> Iterator[str]:
        """Yield distinct worker ids in ring order, starting clockwise from the
        prefix's position. The first id is the prefix's natural affinity owner;
        subsequent ids are the spillover order when that owner is over cap.
        """
        if not self._sorted_positions:
            return
        start = bisect.bisect(self._sorted_positions, _hash(prefix_hash))
        seen: set[str] = set()
        n = len(self._sorted_positions)
        for i in range(n):
            pos = self._sorted_positions[(start + i) % n]
            wid = self._ring[pos]
            if wid not in seen:
                seen.add(wid)
                yield wid
                if len(seen) == len(self._worker_ids):
                    return

    def head(self, prefix_hash: str) -> str | None:
        """The natural (uncapped) owner of a prefix — pure affinity."""
        for wid in self.walk(prefix_hash):
            return wid
        return None
