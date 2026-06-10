"""Tests for admission control / backpressure (services/scheduler/admission.py)."""

from __future__ import annotations

import pytest

from services.scheduler.admission import Admission, Watermarks, classify, should_admit


def test_accept_below_low():
    m = Watermarks(low=2000, high=10000)
    assert classify(0, m) is Admission.ACCEPT
    assert classify(1999, m) is Admission.ACCEPT
    assert should_admit(1999, m)


def test_degraded_between_marks():
    m = Watermarks(low=2000, high=10000)
    assert classify(2000, m) is Admission.DEGRADED
    assert classify(9999, m) is Admission.DEGRADED
    # Degraded is still admitted (soft limit).
    assert should_admit(9999, m)


def test_shed_at_or_above_high():
    m = Watermarks(low=2000, high=10000)
    assert classify(10000, m) is Admission.SHED
    assert classify(50000, m) is Admission.SHED
    assert not should_admit(10000, m)


def test_invalid_watermarks_rejected():
    with pytest.raises(ValueError):
        Watermarks(low=5000, high=5000)  # high must exceed low
    with pytest.raises(ValueError):
        Watermarks(low=-1, high=10)
