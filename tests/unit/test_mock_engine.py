"""Tests for MockEngine (services/worker/engines/mock.py).

The mock is the calibrated benchmarking backbone; its value is that you can
check it on paper. We pin the ``alpha + beta*b`` latency law (jitter off), the
closed-form throughput curve and its monotonicity/asymptote, and the jitter
behaviour.
"""

from __future__ import annotations

import math

from services.worker.engines.mock import MockEngine


def test_base_latency_law_jitter_off():
    e = MockEngine(alpha_ms=18.0, beta_ms=7.5, jitter_sigma=0.0)
    assert math.isclose(e.batch_latency_ms(1), 25.5, rel_tol=1e-9)  # 18 + 7.5
    assert math.isclose(e.batch_latency_ms(4), 48.0, rel_tol=1e-9)  # 18 + 30
    assert math.isclose(e.batch_latency_ms(16), 138.0, rel_tol=1e-9)  # 18 + 120


def test_zero_or_negative_batch_is_zero_latency():
    e = MockEngine(jitter_sigma=0.0)
    assert e.batch_latency_ms(0) == 0.0
    assert e.batch_latency_ms(-3) == 0.0
    assert e.throughput_analytic(0) == 0.0


def test_throughput_formula_and_monotonic_increasing():
    e = MockEngine(alpha_ms=18.0, beta_ms=7.5, jitter_sigma=0.0)
    # throughput(b) = 1000 * b / (alpha + beta*b)
    assert math.isclose(e.throughput_analytic(1), 1000.0 * 1 / 25.5, rel_tol=1e-9)
    vals = [e.throughput_analytic(b) for b in range(1, 33)]
    # Strictly increasing in b.
    assert all(b2 > b1 for b1, b2 in zip(vals, vals[1:], strict=False))


def test_throughput_asymptotes_to_inverse_beta():
    beta = 7.5
    e = MockEngine(alpha_ms=18.0, beta_ms=beta, jitter_sigma=0.0)
    # As b → large, throughput → 1000/beta req/s.
    big = e.throughput_analytic(100000)
    assert math.isclose(big, 1000.0 / beta, rel_tol=1e-3)
    # And it never exceeds the asymptote.
    assert all(e.throughput_analytic(b) < 1000.0 / beta for b in range(1, 50))


def test_per_item_latency_decreases_with_batch_size():
    e = MockEngine(alpha_ms=18.0, beta_ms=7.5, jitter_sigma=0.0)
    per_item = [e.batch_latency_ms(b) / b for b in range(1, 33)]
    assert all(p2 < p1 for p1, p2 in zip(per_item, per_item[1:], strict=False))


def test_jitter_off_is_exact_and_repeatable():
    e1 = MockEngine(jitter_sigma=0.0, seed=1)
    e2 = MockEngine(jitter_sigma=0.0, seed=2)
    assert e1.batch_latency_ms(8) == e2.batch_latency_ms(8)


def test_jitter_on_varies_but_stays_positive():
    e = MockEngine(alpha_ms=18.0, beta_ms=7.5, jitter_sigma=0.2, seed=42)
    samples = [e.batch_latency_ms(8) for _ in range(200)]
    assert all(s > 0 for s in samples)
    # With jitter on, not all samples are identical.
    assert len(set(samples)) > 1
    # Median-1 lognormal: mean should sit near the base (48ms) within ~15%.
    base = 18.0 + 7.5 * 8
    assert abs(sum(samples) / len(samples) - base) / base < 0.15
