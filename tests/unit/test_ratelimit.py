"""Tests for the pure token-bucket limiter (services/gateway/ratelimit.py).

We test the pure ``token_bucket_step`` arithmetic — the same math the Redis Lua
script runs server-side. Refill is linear in elapsed time and clamped to the
burst ceiling; a request consumes one token; an empty bucket rejects and reports
a correct retry-after.
"""

from __future__ import annotations

import math

from services.gateway.ratelimit import BucketState, token_bucket_step


def test_fresh_key_starts_full_and_allows():
    r = token_bucket_step(None, now_s=100.0, rps=10.0, burst=10.0)
    assert r.allowed
    # Started full (10), consumed 1 → 9 left.
    assert math.isclose(r.state.tokens, 9.0, rel_tol=1e-9)
    assert r.retry_after_s == 0.0


def test_consume_until_empty_then_reject():
    rps, burst = 5.0, 5.0
    state = BucketState(tokens=5.0, ts=0.0)
    # Five requests at the same instant drain the bucket.
    for _ in range(5):
        r = token_bucket_step(state, now_s=0.0, rps=rps, burst=burst)
        assert r.allowed
        state = r.state
    # Sixth at the same instant is rejected.
    r = token_bucket_step(state, now_s=0.0, rps=rps, burst=burst)
    assert not r.allowed
    # One token refills in 1/rps = 0.2s.
    assert math.isclose(r.retry_after_s, 0.2, rel_tol=1e-9)


def test_refill_is_linear_in_elapsed_time():
    rps, burst = 10.0, 10.0
    state = BucketState(tokens=0.0, ts=0.0)
    # After 0.5s, 5 tokens have refilled; a request is allowed, leaving 4.
    r = token_bucket_step(state, now_s=0.5, rps=rps, burst=burst)
    assert r.allowed
    assert math.isclose(r.state.tokens, 4.0, rel_tol=1e-9)


def test_refill_clamped_to_burst():
    rps, burst = 10.0, 10.0
    state = BucketState(tokens=0.0, ts=0.0)
    # A long idle period cannot overfill beyond burst.
    r = token_bucket_step(state, now_s=10_000.0, rps=rps, burst=burst)
    assert r.allowed
    # Refilled to the ceiling (10) then consumed 1 → 9.
    assert math.isclose(r.state.tokens, 9.0, rel_tol=1e-9)


def test_cost_greater_than_one():
    rps, burst = 10.0, 10.0
    state = BucketState(tokens=3.0, ts=0.0)
    r = token_bucket_step(state, now_s=0.0, rps=rps, burst=burst, cost=5.0)
    assert not r.allowed  # only 3 tokens, need 5
    # Need 2 more tokens → 0.2s.
    assert math.isclose(r.retry_after_s, 0.2, rel_tol=1e-9)


def test_burst_defaults_to_rps_when_unset():
    # burst <= 0 means "one second of capacity".
    r = token_bucket_step(None, now_s=0.0, rps=7.0, burst=0.0)
    assert r.allowed
    assert math.isclose(r.state.tokens, 6.0, rel_tol=1e-9)


def test_negative_elapsed_clock_skew_does_not_add_tokens():
    rps, burst = 10.0, 10.0
    state = BucketState(tokens=2.0, ts=100.0)
    # now before ts (clock skew): elapsed clamped to 0, no phantom refill.
    r = token_bucket_step(state, now_s=90.0, rps=rps, burst=burst)
    assert r.allowed
    assert math.isclose(r.state.tokens, 1.0, rel_tol=1e-9)
