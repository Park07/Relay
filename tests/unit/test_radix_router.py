"""Tests for RadixPrefixRouter — the longest-prefix router.

Pins the three behaviours the design promises: (1) a warm prefix routes to the
worker holding the longest match, (2) a hot prefix whose owner is over cap spills
instead of piling on (bounded load), (3) a cold prefix with no cached match falls
back to the consistent-hash ring. Load is injected via a stub `load_fn` so these
are deterministic and need no simulator.
"""

from relay_core.types import WorkerState
from services.scheduler.radix_router import RadixPrefixRouter
from services.scheduler.router import PURE_AFFINITY

MODEL = "m"


def mk_workers(n: int) -> list[WorkerState]:
    return [WorkerState(worker_id=f"w{i}", engine="mock", models=(MODEL,),
                        max_concurrent_batches=8) for i in range(n)]


def blk(*xs: str) -> tuple[str, ...]:
    return tuple(xs)


def test_warm_prefix_routes_to_longest_match():
    ws = mk_workers(3)
    r = RadixPrefixRouter(ws, load_cap_factor=PURE_AFFINITY)
    # Prime: send a prompt; it lands somewhere (cold → ring).
    first = r.pick(MODEL, blk("sys", "doc", "qA"))
    # A prompt sharing the first two blocks should go to the same worker (longest
    # match), regardless of where the ring would have sent it cold.
    second = r.pick(MODEL, blk("sys", "doc", "qB"))
    assert second.worker_id == first.worker_id


def test_pure_affinity_concentrates_shared_stem():
    ws = mk_workers(4)
    r = RadixPrefixRouter(ws, load_cap_factor=PURE_AFFINITY)
    owners = set()
    for i in range(20):
        w = r.pick(MODEL, blk("stem", "block", f"tail{i}"))  # all share 2 blocks
        owners.add(w.worker_id)
    # All 20, sharing the same 2-block stem, must concentrate on ONE worker.
    assert len(owners) == 1


def test_bounded_load_spills_when_owner_capped():
    ws = mk_workers(2)
    load = {"w0": 0.0, "w1": 0.0}
    r = RadixPrefixRouter(ws, load_cap_factor=1.25, load_fn=lambda w: load[w.worker_id])
    # Warm w-something with a stem.
    first = r.pick(MODEL, blk("hot", "stem", "a"))
    owner = first.worker_id
    other = "w1" if owner == "w0" else "w0"
    # Now make the owner heavily loaded and the other idle; a same-stem request
    # must spill to the other worker rather than pile onto the capped owner.
    load[owner] = 100.0
    load[other] = 0.0
    nxt = r.pick(MODEL, blk("hot", "stem", "b"))
    assert nxt.worker_id == other


def test_cold_prefix_uses_ring_and_is_deterministic():
    ws = mk_workers(3)
    r1 = RadixPrefixRouter(ws, load_cap_factor=PURE_AFFINITY)
    r2 = RadixPrefixRouter(ws, load_cap_factor=PURE_AFFINITY)
    # Same cold prefix on two fresh routers → same ring head (deterministic).
    a = r1.pick(MODEL, blk("cold", "unique", "xyz"))
    b = r2.pick(MODEL, blk("cold", "unique", "xyz"))
    assert a.worker_id == b.worker_id


def test_no_capable_worker_returns_none():
    ws = mk_workers(2)
    r = RadixPrefixRouter(ws, load_cap_factor=1.25)
    assert r.pick("other-model", blk("a", "b")) is None


def test_all_capped_returns_none_backpressure():
    ws = mk_workers(2)
    # Force every worker ineligible via admit_fn → no placement possible.
    r = RadixPrefixRouter(ws, load_cap_factor=1.25, load_fn=lambda w: 0.0,
                          admit_fn=lambda w: False)
    assert r.pick(MODEL, blk("a")) is None


def test_matched_blocks_reports_reuse_depth():
    ws = mk_workers(2)
    r = RadixPrefixRouter(ws, load_cap_factor=PURE_AFFINITY)
    r.pick(MODEL, blk("a", "b", "c", "d"))
    # A new prompt sharing 3 leading blocks → reuse depth 3.
    assert r.matched_blocks(blk("a", "b", "c", "e")) == 3
    assert r.matched_blocks(blk("x", "y")) == 0  # nothing cached from block 0


def test_eviction_capacity_forgets_cold_prefix():
    ws = mk_workers(1)  # single worker so everything lands on it
    r = RadixPrefixRouter(ws, load_cap_factor=PURE_AFFINITY, cache_capacity=1)
    r.pick(MODEL, blk("a", "b"))
    r.pick(MODEL, blk("c", "d"))  # capacity 1 → [a,b] evicted from the tree
    assert r.matched_blocks(blk("a", "b")) == 0   # forgotten
    assert r.matched_blocks(blk("c", "d")) == 2   # retained