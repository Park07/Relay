"""Prefix-aware routing with bounded load — the lead depth feature
(DESIGN.md §8.2, ADR-12).

The tension this navigates:

  * **Pure affinity** (send every same-prefix request to its ring owner)
    maximizes KV-cache reuse but lets one hot prefix pin all its traffic to a
    single worker, which saturates while peers idle → terrible balance, bad p99.
  * **Pure balance** (least-loaded / round-robin) keeps load flat but scatters a
    prefix across every worker, so each one re-prefills it → locality collapses.

Bounded-load consistent hashing walks the ring from the prefix's position and
takes the first capable worker *under a load cap proportional to the current
average load*. One knob, ``load_cap_factor``, slides the whole policy space:

    load_cap_factor → ∞   ::  cap is unbounded  →  pure affinity   (max locality)
    load_cap_factor → 1   ::  cap ≈ average load →  spill off hot workers
                                                  →  round-robin-like (max balance)

The §13.2 Pareto frontier is precisely a sweep of this one number, with the two
extremes as the reference endpoints and the bounded-load middle as the result.

Load measure. "Load" here is a worker's in-flight **batch** count (``inflight``);
``free_slots = max_concurrent_batches - inflight`` is the hard overcommit guard
(a worker at zero free slots is never picked, regardless of cap). With more than
one concurrent slot per worker, ``inflight`` has enough resolution for the cap to
slide smoothly between the two extremes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from statistics import mean

from relay_core.hashing import BoundedLoadConsistentHashRing
from relay_core.types import WorkerState

# Sentinel for "pure affinity": an unbounded cap. We accept float('inf') from
# callers too; both mean the cap never binds.
PURE_AFFINITY = float("inf")


def default_load(w: WorkerState) -> float:
    return float(w.inflight)


def default_admit(w: WorkerState) -> bool:
    # In the batch-routing path a worker must have a free batch slot to take a
    # batch *now*. In the per-request admission path (where requests queue at the
    # worker) callers pass ``admit_fn=lambda w: True`` because queuing is allowed.
    return w.free_slots > 0


class PrefixRouter:
    def __init__(
        self,
        workers: Iterable[WorkerState],
        load_cap_factor: float = 1.25,
        vnodes: int = 160,
        load_fn: Callable[[WorkerState], float] = default_load,
        admit_fn: Callable[[WorkerState], bool] = default_admit,
    ) -> None:
        self.workers: dict[str, WorkerState] = {w.worker_id: w for w in workers}
        self.ring = BoundedLoadConsistentHashRing(self.workers.values(), vnodes=vnodes)
        self.cap_factor = float(load_cap_factor)
        self.load_fn = load_fn
        self.admit_fn = admit_fn

    # -- fleet membership (autoscaling) ------------------------------------ #
    def add_worker(self, w: WorkerState) -> None:
        self.workers[w.worker_id] = w
        self.ring.add(w.worker_id)

    def remove_worker(self, worker_id: str) -> None:
        self.workers.pop(worker_id, None)
        self.ring.remove(worker_id)

    def capable(self, model: str) -> list[WorkerState]:
        return [w for w in self.workers.values() if w.has_model(model)]

    # -- the placement decision (DESIGN.md §8.2) --------------------------- #
    def pick(self, model: str, prefix_hash: str) -> WorkerState | None:
        capable = self.capable(model)
        if not capable:
            return None

        # Bounded-load cap: proportional to current average load, floored at 1 so
        # an all-idle fleet still admits one batch per worker.
        if self.cap_factor == PURE_AFFINITY:
            cap = PURE_AFFINITY
        else:
            avg_load = mean(self.load_fn(w) for w in capable)
            cap = self.cap_factor * max(avg_load, 1.0)

        # Walk the ring from the prefix's hash position. The first capable worker
        # under cap with a free slot wins. Affinity owner is tried first; under
        # cap pressure we spill to its ring successors (which is *also* stable —
        # the same hot prefix spills to the same secondary, so the secondary's
        # cache warms too: bounded "replication" falls out for free).
        for wid in self.ring.walk(prefix_hash):
            w = self.workers.get(wid)
            if w is None or not w.has_model(model):
                continue
            if self.load_fn(w) < cap and self.admit_fn(w):
                return w
        return None  # everyone capped/full → backpressure; caller retries next tick

    # -- introspection for the harness ------------------------------------- #
    def affinity_owner(self, prefix_hash: str) -> str | None:
        return self.ring.head(prefix_hash)
