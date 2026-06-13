"""Tests for the deadline-based batch former (services/scheduler/batch_former.py).

The former's decision logic is split into pure helpers so the live loop, the
simulator, and these tests share one implementation. We pin: slack arithmetic,
most-urgent selection, the dispatch trigger (full batch OR overdue), high-first
batch assembly under a cap, the per-worker cap formula, and the no-starvation
property that a default item past its budget triggers its own dispatch.
"""

from __future__ import annotations

from relay_core.queue import LocalQueue
from relay_core.types import InferItem, Priority, WorkerState
from services.scheduler.batch_former import (
    BUDGET_MS,
    Head,
    assemble_batch,
    most_urgent,
    should_dispatch,
    slack_ms,
    worker_batch_cap,
)


def _item(rid: str, enqueue_ts: float, priority: Priority = Priority.DEFAULT) -> InferItem:
    # Give an explicit prefix_hash so construction does not depend on hashing.
    return InferItem(
        request_id=rid,
        input=f"x-{rid}",
        prefix_hash=f"ph-{rid}",
        enqueue_ts=enqueue_ts,
        priority=priority,
    )


# -- slack_ms --------------------------------------------------------------- #
def test_slack_ms_positive_then_negative_across_budget():
    h = Head(Priority.DEFAULT, _item("a", enqueue_ts=100.0))
    budget = BUDGET_MS[Priority.DEFAULT]  # 50ms
    # Just enqueued: nearly the full budget of slack.
    assert slack_ms(h, now_ms=100.0, budgets=BUDGET_MS) == budget
    # Exactly at the deadline: zero slack.
    assert slack_ms(h, now_ms=100.0 + budget, budgets=BUDGET_MS) == 0.0
    # Past the deadline: negative.
    assert slack_ms(h, now_ms=100.0 + budget + 10, budgets=BUDGET_MS) < 0.0


def test_high_priority_has_tighter_budget():
    hi = Head(Priority.HIGH, _item("h", 0.0, Priority.HIGH))
    lo = Head(Priority.DEFAULT, _item("l", 0.0, Priority.DEFAULT))
    # At the same now, the high-priority item has less slack (tighter budget).
    assert slack_ms(hi, 1.0, BUDGET_MS) < slack_ms(lo, 1.0, BUDGET_MS)


# -- most_urgent ------------------------------------------------------------ #
def test_most_urgent_picks_smallest_slack():
    heads = [
        Head(Priority.DEFAULT, _item("old", enqueue_ts=0.0)),
        Head(Priority.DEFAULT, _item("new", enqueue_ts=40.0)),
    ]
    # At now=45, the older item has the least slack.
    u = most_urgent(heads, now_ms=45.0, budgets=BUDGET_MS)
    assert u is not None and u.item.request_id == "old"


def test_most_urgent_empty_is_none():
    assert most_urgent([], now_ms=0.0, budgets=BUDGET_MS) is None


def test_most_urgent_prefers_high_priority_when_contemporaneous():
    heads = [
        Head(Priority.DEFAULT, _item("d", 0.0, Priority.DEFAULT)),
        Head(Priority.HIGH, _item("h", 0.0, Priority.HIGH)),
    ]
    u = most_urgent(heads, now_ms=2.0, budgets=BUDGET_MS)
    assert u is not None and u.item.request_id == "h"


# -- should_dispatch -------------------------------------------------------- #
def test_should_dispatch_on_full_batch_even_if_not_overdue():
    urgent = Head(Priority.DEFAULT, _item("a", enqueue_ts=1000.0))  # tons of slack
    assert should_dispatch(
        total_queued=16, urgent=urgent, now_ms=1000.0, max_batch=16, budgets=BUDGET_MS
    )


def test_should_dispatch_on_overdue_even_if_not_full():
    urgent = Head(Priority.DEFAULT, _item("a", enqueue_ts=0.0))
    # now well past the 50ms budget, only 1 item queued
    assert should_dispatch(
        total_queued=1, urgent=urgent, now_ms=100.0, max_batch=16, budgets=BUDGET_MS
    )


def test_should_not_dispatch_when_partial_and_not_overdue():
    urgent = Head(Priority.DEFAULT, _item("a", enqueue_ts=100.0))
    assert not should_dispatch(
        total_queued=3, urgent=urgent, now_ms=110.0, max_batch=16, budgets=BUDGET_MS
    )


def test_should_not_dispatch_when_empty():
    assert not should_dispatch(
        total_queued=0, urgent=None, now_ms=10.0, max_batch=16, budgets=BUDGET_MS
    )


# -- assemble_batch --------------------------------------------------------- #
def _queues(n_high: int, n_default: int) -> dict[Priority, LocalQueue]:
    qh, qd = LocalQueue(), LocalQueue()
    for i in range(n_high):
        qh.push(_item(f"h{i}", 0.0, Priority.HIGH))
    for i in range(n_default):
        qd.push(_item(f"d{i}", 0.0, Priority.DEFAULT))
    return {Priority.HIGH: qh, Priority.DEFAULT: qd}


def test_assemble_batch_high_priority_first():
    qs = _queues(n_high=2, n_default=10)
    batch = assemble_batch(qs, cap=4)
    assert len(batch) == 4
    # First two must be the high-priority items, then defaults fill the rest.
    assert [it.request_id for it in batch[:2]] == ["h0", "h1"]
    assert all(it.priority == Priority.DEFAULT for it in batch[2:])


def test_assemble_batch_respects_cap():
    qs = _queues(n_high=0, n_default=100)
    batch = assemble_batch(qs, cap=5)
    assert len(batch) == 5


def test_assemble_batch_cap_zero_or_negative_is_empty():
    qs = _queues(n_high=3, n_default=3)
    assert assemble_batch(qs, cap=0) == []
    assert assemble_batch(qs, cap=-2) == []


def test_assemble_batch_drains_then_stops():
    qs = _queues(n_high=1, n_default=1)
    batch = assemble_batch(qs, cap=16)  # more capacity than items
    assert len(batch) == 2  # only what's available


# -- worker_batch_cap ------------------------------------------------------- #
def test_worker_batch_cap_limited_by_free_slots():
    # 1 free slot * max_batch(16) = 16, capped by max_batch arg of 16 => 16.
    w = WorkerState("w0", "mock", ("m",), max_batch=16, max_concurrent_batches=1)
    assert worker_batch_cap(w, max_batch=16) == 16


def test_worker_batch_cap_zero_when_no_free_slots():
    w = WorkerState("w0", "mock", ("m",), max_batch=16, max_concurrent_batches=2, inflight=2)
    assert w.free_slots == 0
    assert worker_batch_cap(w, max_batch=16) == 0


def test_worker_batch_cap_takes_min_of_max_batch_and_capacity():
    # 2 free slots * max_batch 4 = 8, but the caller cap (max_batch arg) is 5.
    w = WorkerState("w0", "mock", ("m",), max_batch=4, max_concurrent_batches=2)
    assert worker_batch_cap(w, max_batch=5) == 5
    # And when capacity is the binding constraint:
    w2 = WorkerState("w1", "mock", ("m",), max_batch=2, max_concurrent_batches=1)
    assert worker_batch_cap(w2, max_batch=16) == 2


# -- no-starvation ---------------------------------------------------------- #
def test_no_starvation_overdue_default_triggers_dispatch_under_high_load():
    # A default item sits while high-priority traffic streams in. Once it goes
    # overdue it must itself trigger a dispatch (the no-starvation guarantee),
    # regardless of how many high-priority items are queued.
    qd = LocalQueue()
    old_default = _item("d-old", enqueue_ts=0.0, priority=Priority.DEFAULT)
    qd.push(old_default)
    qh = LocalQueue()
    for i in range(3):  # high-priority churn, but never a full batch
        qh.push(_item(f"h{i}", enqueue_ts=90.0, priority=Priority.HIGH))
    queues = {Priority.HIGH: qh, Priority.DEFAULT: qd}

    now = 100.0  # past the default 50ms budget
    heads = [Head(p, q.peek()) for p, q in queues.items() if q.size()]
    urgent = most_urgent(heads, now, BUDGET_MS)
    total = sum(q.size() for q in queues.values())
    # The overdue default is the most urgent and forces a dispatch.
    assert urgent is not None and urgent.item.request_id == "d-old"
    assert should_dispatch(total, urgent, now, max_batch=16, budgets=BUDGET_MS)
