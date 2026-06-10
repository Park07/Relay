"""Tests for CacheAwareMockEngine (services/worker/engines/cache_aware_mock.py).

This engine is what the routing result is *measured* on, so its cache mechanics
and latency law must be exact: cold prefixes miss, resident prefixes hit, the
prefill cost scales with the number of distinct missed prefixes, and the LRU
cache evicts past capacity. With jitter disabled the latency is a deterministic
function we can check on paper.
"""

from __future__ import annotations

import math

from relay_core.types import InferItem
from services.worker.engines.cache_aware_mock import CacheAwareMockEngine


def _item(prefix: str, rid: str = "r") -> InferItem:
    return InferItem(request_id=rid, input=f"body-{rid}", prefix_hash=prefix)


def _engine(**kw) -> CacheAwareMockEngine:
    # jitter off by default here so latency is exactly base + prefill.
    kw.setdefault("jitter_sigma", 0.0)
    return CacheAwareMockEngine(**kw)


def test_first_batch_all_miss():
    e = _engine(alpha_ms=18.0, beta_ms=7.5, prefill_ms=160.0)
    batch = [_item("p1", "a"), _item("p2", "b")]
    latency, hits = e.run_batch(batch)
    assert hits == [False, False]
    assert e.cache_misses == 2 and e.cache_hits == 0
    # base = 18 + 7.5*2 = 33 ; prefill = 160 * 2 distinct missed = 320
    assert math.isclose(latency, 33.0 + 320.0, rel_tol=1e-9)


def test_repeat_prefix_hits_second_time():
    e = _engine(prefill_ms=160.0)
    e.run_batch([_item("p1", "a")])  # cold miss, now resident
    latency, hits = e.run_batch([_item("p1", "b")])  # same prefix → hit
    assert hits == [True]
    # base = 18 + 7.5*1 = 25.5 ; no prefill on a hit
    assert math.isclose(latency, 25.5, rel_tol=1e-9)
    assert e.cache_hits == 1


def test_distinct_missed_charged_once_per_prefix():
    # Two items share one new prefix + one item has another new prefix.
    e = _engine(alpha_ms=10.0, beta_ms=1.0, prefill_ms=100.0)
    batch = [_item("shared", "a"), _item("shared", "b"), _item("other", "c")]
    latency, hits = e.run_batch(batch)
    assert hits == [False, False, False]
    # base = 10 + 1*3 = 13 ; distinct missed = {shared, other} = 2 → prefill 200
    assert math.isclose(latency, 13.0 + 200.0, rel_tol=1e-9)
    # All three items counted as misses for the ratio, but prefill paid twice.
    assert e.cache_misses == 3


def test_within_batch_same_new_prefix_resident_after():
    # After a batch, every prefix it touched is resident (warm for next time).
    e = _engine(prefill_ms=100.0)
    e.run_batch([_item("p", "a"), _item("p", "b")])
    _, hits = e.run_batch([_item("p", "c")])
    assert hits == [True]


def test_lru_eviction_past_capacity():
    e = _engine(cache_capacity=2, prefill_ms=100.0)
    e.run_batch([_item("A", "1")])  # cache: [A]
    e.run_batch([_item("B", "2")])  # cache: [A, B]
    e.run_batch([_item("C", "3")])  # inserting C evicts A (LRU) → [B, C]
    # A should now miss again (evicted); B and C should hit.
    _, hit_a = e.run_batch([_item("A", "4")])
    assert hit_a == [False]


def test_lru_touch_keeps_recently_used():
    e = _engine(cache_capacity=2, prefill_ms=100.0)
    e.run_batch([_item("A", "1")])  # [A]
    e.run_batch([_item("B", "2")])  # [A, B]
    e.run_batch([_item("A", "3")])  # touch A → A most-recent → [B, A]
    e.run_batch([_item("C", "4")])  # evicts B (now LRU) → [A, C]
    _, hit_a = e.run_batch([_item("A", "5")])
    _, hit_b = e.run_batch([_item("B", "6")])
    assert hit_a == [True]   # A survived because it was touched
    assert hit_b == [False]  # B was evicted


def test_empty_batch_is_zero_latency_no_state_change():
    e = _engine()
    latency, hits = e.run_batch([])
    assert latency == 0.0 and hits == []
    assert e.cache_hits == 0 and e.cache_misses == 0


def test_stats_reports_hit_ratio():
    e = _engine(prefill_ms=100.0)
    e.run_batch([_item("p", "a")])  # miss
    e.run_batch([_item("p", "b")])  # hit
    st = e.stats()
    assert st.cache_hits == 1 and st.cache_misses == 1
    assert math.isclose(st.cache_hit_ratio, 0.5, rel_tol=1e-9)


def test_latency_deterministic_with_jitter_off():
    e1 = _engine(seed=1)
    e2 = _engine(seed=999)  # different seed must not matter when jitter is off
    b = [_item("p", "a"), _item("p", "b")]
    assert e1.run_batch(b)[0] == e2.run_batch(b)[0]


def test_jitter_on_is_positive_and_varies():
    e = CacheAwareMockEngine(jitter_sigma=0.3, seed=7, prefill_ms=100.0)
    lat1, _ = e.run_batch([_item("p", "a")])
    lat2, _ = e.run_batch([_item("q", "b")])
    assert lat1 > 0 and lat2 > 0
    # Same structural cost (1 item, 1 distinct miss) but jitter makes them differ.
    assert lat1 != lat2
