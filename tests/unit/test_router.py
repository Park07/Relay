"""Tests for prefix-aware bounded-load routing (services/scheduler/router.py)
and the round-robin baseline (bench/simulate.py).

These pin the *policy-defining* behaviours:
  * cap_factor → ∞ is pure affinity (always the ring head),
  * a finite cap spills a hot prefix off its overloaded owner onto a stable
    successor,
  * admit_fn gates placement (free-slot / overcommit guard),
  * round-robin is prefix-agnostic and balances load.
"""

from __future__ import annotations

from bench.simulate import RoundRobinRouter
from relay_core.types import WorkerState
from services.scheduler.router import PURE_AFFINITY, PrefixRouter


def _workers(n: int, **kw) -> list[WorkerState]:
    return [
        WorkerState(worker_id=f"w{i}", engine="mock", models=("m",), **kw)
        for i in range(n)
    ]


def test_capable_filters_by_model():
    ws = [
        WorkerState("w0", "mock", ("m",)),
        WorkerState("w1", "mock", ("other",)),
        WorkerState("w2", "mock", ("m", "other")),
    ]
    r = PrefixRouter(ws, load_cap_factor=PURE_AFFINITY)
    capable_ids = {w.worker_id for w in r.capable("m")}
    assert capable_ids == {"w0", "w2"}
    assert r.pick("nonexistent-model", "p") is None


def test_pure_affinity_always_picks_ring_head():
    r = PrefixRouter(_workers(5), load_cap_factor=PURE_AFFINITY)
    for ph in ("alpha", "beta", "gamma", "delta", "prefix-42"):
        owner = r.affinity_owner(ph)
        picked = r.pick("m", ph)
        assert picked is not None and picked.worker_id == owner


def test_pure_affinity_ignores_load():
    # Even when the owner is heavily loaded, infinite cap keeps routing to it.
    r = PrefixRouter(_workers(4, max_concurrent_batches=8), load_cap_factor=PURE_AFFINITY)
    ph = "hot-prefix"
    owner_id = r.affinity_owner(ph)
    r.workers[owner_id].inflight = 7  # nearly saturated but still has a free slot
    picked = r.pick("m", ph)
    assert picked is not None and picked.worker_id == owner_id


def test_bounded_cap_spills_off_overloaded_owner():
    # One worker is hammered; a finite cap must spill the prefix to the *next*
    # capable worker on the ring rather than piling on.
    r = PrefixRouter(_workers(4, max_concurrent_batches=8), load_cap_factor=1.25)
    ph = "spill-prefix"
    walk_order = list(r.ring.walk(ph))
    owner_id = walk_order[0]

    # Make the owner's load high while peers stay at zero, so avg is low and the
    # cap (1.25 * max(avg,1)) sits below the owner's load.
    r.workers[owner_id].inflight = 6
    picked = r.pick("m", ph)
    assert picked is not None
    assert picked.worker_id != owner_id
    # Must be the first *under-cap* successor in ring order — stable spill.
    assert picked.worker_id == walk_order[1]


def test_spill_is_stable_same_secondary_each_time():
    r = PrefixRouter(_workers(5, max_concurrent_batches=8), load_cap_factor=1.1)
    ph = "stable-spill"
    walk_order = list(r.ring.walk(ph))
    r.workers[walk_order[0]].inflight = 7
    picks = {r.pick("m", ph).worker_id for _ in range(10)}
    # Deterministic: the same secondary every time (warms one secondary cache).
    assert picks == {walk_order[1]}


def test_admit_fn_false_blocks_all_placement():
    r = PrefixRouter(
        _workers(3, max_concurrent_batches=2),
        load_cap_factor=PURE_AFFINITY,
        admit_fn=lambda w: False,
    )
    assert r.pick("m", "p") is None


def test_default_admit_blocks_full_workers():
    # default_admit requires a free batch slot. Fill every worker → backpressure.
    r = PrefixRouter(_workers(3, max_concurrent_batches=1), load_cap_factor=PURE_AFFINITY)
    for w in r.workers.values():
        w.inflight = 1  # zero free slots
    assert r.pick("m", "p") is None


def test_queue_admission_path_allows_full_workers():
    # In the per-request admission path callers pass admit_fn=lambda w: True
    # (queuing at the worker is allowed even with no free batch slot).
    r = PrefixRouter(
        _workers(3, max_concurrent_batches=1),
        load_cap_factor=PURE_AFFINITY,
        admit_fn=lambda w: True,
    )
    for w in r.workers.values():
        w.inflight = 1
    picked = r.pick("m", "p")
    assert picked is not None  # still placeable because queuing is allowed


def test_custom_load_fn_drives_the_cap():
    # Use a load_fn reading a side dict (mirrors the simulator's queued+inflight
    # metric). The owner's high custom-load must trigger a spill.
    load = {"w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0}
    r = PrefixRouter(
        _workers(4, max_concurrent_batches=8),
        load_cap_factor=1.25,
        load_fn=lambda w: load[w.worker_id],
        admit_fn=lambda w: True,
    )
    ph = "load-fn-prefix"
    walk_order = list(r.ring.walk(ph))
    load[walk_order[0]] = 20.0  # owner swamped per the custom metric
    picked = r.pick("m", ph)
    assert picked is not None and picked.worker_id == walk_order[1]


# --------------------------------------------------------------------------- #
# Round-robin baseline (prefix-agnostic least-loaded)
# --------------------------------------------------------------------------- #
def test_round_robin_balances_and_ignores_prefix():
    ws = _workers(4, max_concurrent_batches=8)
    load = {w.worker_id: 0.0 for w in ws}
    rr = RoundRobinRouter(ws, load_fn=lambda w: load[w.worker_id])

    # Always pick least-loaded; simulate placing 8 items and bumping load.
    placements = []
    for _ in range(8):
        w = rr.pick("m", "irrelevant-prefix")
        placements.append(w.worker_id)
        load[w.worker_id] += 1.0

    # With equal starting load, 8 placements over 4 workers => 2 each (balanced).
    counts = {wid: placements.count(wid) for wid in load}
    assert all(c == 2 for c in counts.values()), counts


def test_round_robin_prefix_does_not_change_choice():
    # Two routers with identical state: the only difference is the prefix passed
    # in. Round-robin must return the same worker — the prefix is ignored. (Two
    # fresh routers isolate the choice from the tie-break cycle's advancing
    # state, which would otherwise differ between two calls on one router.)
    def fresh():
        ws = _workers(4, max_concurrent_batches=8)
        # Skew loads so there is a unique least-loaded worker (no tie at all).
        ws[2].inflight = 0
        for i in (0, 1, 3):
            ws[i].inflight = 5
        return RoundRobinRouter(ws, load_fn=lambda w: float(w.inflight))

    a = fresh().pick("m", "prefix-A").worker_id
    b = fresh().pick("m", "prefix-B").worker_id
    assert a == b == "w2"  # least-loaded regardless of prefix
