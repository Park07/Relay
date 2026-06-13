"""Tests for the bounded-load consistent-hash ring (relay_core/hashing.py).

The ring carries the two properties the whole depth feature rests on:
deterministic affinity, and minimal disruption on fleet change (O(K/N) keys
move, not all of them). These tests pin both, plus distribution evenness.
"""

from __future__ import annotations

import statistics

from relay_core.hashing import BoundedLoadConsistentHashRing
from relay_core.types import WorkerState


def _workers(n: int) -> list[WorkerState]:
    return [WorkerState(worker_id=f"w{i}", engine="mock", models=("m",)) for i in range(n)]


def test_walk_yields_all_workers_once_in_stable_order():
    ring = BoundedLoadConsistentHashRing(_workers(5), vnodes=64)
    order = list(ring.walk("some-prefix-hash"))
    # Every worker appears exactly once.
    assert sorted(order) == ["w0", "w1", "w2", "w3", "w4"]
    assert len(order) == len(set(order))
    # And the order is stable across calls (determinism).
    assert order == list(ring.walk("some-prefix-hash"))


def test_head_is_first_of_walk_and_deterministic():
    ring = BoundedLoadConsistentHashRing(_workers(5), vnodes=64)
    for ph in ("a", "b", "c", "prefix-xyz", "0123456789abcdef"):
        assert ring.head(ph) == next(iter(ring.walk(ph)))
        # Repeated lookups never drift.
        assert ring.head(ph) == ring.head(ph)


def test_empty_ring_head_is_none_and_walk_is_empty():
    ring = BoundedLoadConsistentHashRing([], vnodes=64)
    assert ring.head("anything") is None
    assert list(ring.walk("anything")) == []
    assert len(ring) == 0


def test_add_remove_membership_and_len():
    ring = BoundedLoadConsistentHashRing(_workers(3), vnodes=32)
    assert len(ring) == 3
    ring.add("w9")
    assert len(ring) == 4
    assert "w9" in set(ring.walk("p"))
    # Idempotent add.
    ring.add("w9")
    assert len(ring) == 4
    ring.remove("w9")
    assert len(ring) == 3
    assert "w9" not in set(ring.walk("p"))
    # Idempotent remove.
    ring.remove("w9")
    assert len(ring) == 3


def test_minimal_disruption_on_add():
    # Adding one worker to an N-worker fleet should only re-home about 1/(N+1)
    # of keys (the arc the newcomer captures), not a large fraction. This is the
    # property that keeps caches warm across autoscale events.
    ring = BoundedLoadConsistentHashRing(_workers(8), vnodes=200)
    keys = [f"prefix-{i}" for i in range(5000)]
    before = {k: ring.head(k) for k in keys}

    ring.add("w8")
    after = {k: ring.head(k) for k in keys}

    moved = sum(1 for k in keys if before[k] != after[k])
    frac = moved / len(keys)
    # Ideal share for the 9th worker is ~1/9 ≈ 0.111. Allow generous slack for
    # vnode variance but assert it is nowhere near a hash-mod reshuffle (~0.89).
    assert 0.04 < frac < 0.25, f"moved fraction {frac:.3f} outside expected band"

    # Every key that moved must have moved *onto* the new worker (consistent
    # hashing only steals from neighbours; it never reshuffles unrelated keys).
    for k in keys:
        if before[k] != after[k]:
            assert after[k] == "w8"


def test_minimal_disruption_on_remove():
    ring = BoundedLoadConsistentHashRing(_workers(8), vnodes=200)
    keys = [f"prefix-{i}" for i in range(5000)]
    before = {k: ring.head(k) for k in keys}

    ring.remove("w3")
    after = {k: ring.head(k) for k in keys}

    # Only keys that were homed on w3 may change; everyone else stays put.
    for k in keys:
        if before[k] != "w3":
            assert after[k] == before[k], "removing a worker disturbed an unrelated key"
        else:
            assert after[k] != "w3"


def test_distribution_is_roughly_even():
    # With enough vnodes, ownership of random keys should be spread across the
    # fleet without any worker hogging a wildly disproportionate share.
    ring = BoundedLoadConsistentHashRing(_workers(4), vnodes=200)
    counts = {f"w{i}": 0 for i in range(4)}
    for i in range(20000):
        counts[ring.head(f"k-{i}")] += 1
    shares = [c / 20000 for c in counts.values()]
    # Perfectly even is 0.25 each. Assert the spread is modest.
    assert max(shares) < 0.35
    assert min(shares) > 0.15
    # Coefficient of variation should be small with 200 vnodes.
    cv = statistics.pstdev(shares) / statistics.mean(shares)
    assert cv < 0.25, f"ownership too uneven, cv={cv:.3f}"
