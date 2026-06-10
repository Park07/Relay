"""Tests for the Zipfian-over-shared-prefixes workload (bench/workload.py).

The workload is what makes the routing result non-trivial: requests must share
prefixes with realistic skew. We pin the distribution (pmf sums to 1, top-k mass
is monotone in skew), prefix-hash stability (the hash is independent of the
unique suffix), and that the realized empirical distribution converges to the
theoretical one as we draw more samples.
"""

from __future__ import annotations

import math

import numpy as np

from bench.workload import WorkloadParams, ZipfianPrefixWorkload
from relay_core.types import Priority


def test_pmf_sums_to_one():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=200, skew=1.1, seed=1))
    assert math.isclose(float(w._pmf.sum()), 1.0, rel_tol=1e-12)


def test_topk_mass_increases_with_skew():
    low = ZipfianPrefixWorkload(WorkloadParams(pool_size=200, skew=0.7, seed=1))
    high = ZipfianPrefixWorkload(WorkloadParams(pool_size=200, skew=1.5, seed=1))
    # More skew → more probability mass concentrated on the top prefixes.
    assert high.theoretical_topk_mass(10) > low.theoretical_topk_mass(10)


def test_theoretical_topk_mass_is_monotone_in_k():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=200, skew=1.1, seed=1))
    masses = [w.theoretical_topk_mass(k) for k in range(1, 50)]
    assert all(m2 >= m1 for m1, m2 in zip(masses, masses[1:]))
    assert masses[-1] <= 1.0 + 1e-12


def test_prefix_hash_independent_of_suffix():
    # Two draws of the *same* rank produce different inputs (unique suffixes)
    # but the SAME prefix_hash — the property the cache keys on.
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=5, skew=0.0, seed=3))
    # skew=0 → uniform; force a specific index by hashing the known prefix.
    items = [w.next() for _ in range(50)]
    by_hash: dict[str, set[str]] = {}
    for it in items:
        by_hash.setdefault(it.prefix_hash, set()).add(it.input)
    # At least one prefix drawn more than once, with distinct full inputs.
    multi = [inputs for inputs in by_hash.values() if len(inputs) > 1]
    assert multi, "expected some prefix drawn multiple times"
    # Same hash, different inputs → suffix doesn't leak into the hash.
    assert all(len(inputs) >= 2 for inputs in multi)


def test_prefix_hash_matches_pool_hashes():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=20, skew=1.0, seed=5))
    pool = set(w._prefix_hashes)
    for _ in range(100):
        assert w.next().prefix_hash in pool  # only ever draws from the pool


def test_realized_distribution_converges_to_theoretical():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=50, skew=1.1, seed=7))
    for _ in range(50000):
        w.next()
    # Realized top-10 mass should be close to the theoretical top-10 mass.
    theo = w.theoretical_topk_mass(10)
    real = w.realized_topk_mass(10)
    assert abs(real - theo) < 0.03, f"theo={theo:.3f} real={real:.3f}"


def test_distinct_drawn_grows_and_bounded_by_pool():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=30, skew=0.8, seed=11))
    for _ in range(5000):
        w.next()
    d = w.distinct_drawn()
    assert 0 < d <= 30


def test_draw_counts_track_total():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=40, skew=1.2, seed=2))
    n = 3000
    for _ in range(n):
        w.next()
    assert int(w.draw_counts.sum()) == n


def test_generate_high_fraction_distribution():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=40, skew=1.0, seed=4))
    items = w.generate(4000, high_fraction=0.25)
    n_high = sum(1 for it in items if it.priority == Priority.HIGH)
    frac = n_high / len(items)
    # Roughly a quarter high-priority (sampling noise tolerated).
    assert 0.20 < frac < 0.30


def test_generate_default_is_all_default_priority():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=10, skew=1.0, seed=8))
    items = w.generate(200)  # high_fraction defaults to 0
    assert all(it.priority == Priority.DEFAULT for it in items)


def test_request_ids_are_unique_and_sequential():
    w = ZipfianPrefixWorkload(WorkloadParams(pool_size=10, skew=1.0, seed=9))
    items = w.generate(100)
    ids = [it.request_id for it in items]
    assert len(set(ids)) == 100
    assert ids[0] == "req-00000001" and ids[-1] == "req-00000100"


def test_as_dict_roundtrips_params():
    p = WorkloadParams(pool_size=123, skew=1.3, prefix_chars=512, suffix_chars=32, seed=17)
    d = p.as_dict()
    assert d == {
        "pool_size": 123, "skew": 1.3, "prefix_chars": 512,
        "suffix_chars": 32, "seed": 17,
    }
