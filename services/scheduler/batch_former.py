"""Deadline-based batch former (DESIGN.md §8.1).

Priority is **not** a second mechanism — it is a tighter latency budget. The
former dispatches when *either* a full batch is available *or* the most-urgent
item across all priority queues has blown its budget. High-priority items fill a
batch first; a batch is then topped up with default-priority items. This yields a
provable no-starvation bound:

    A ``default`` item enqueued at t is dispatched by roughly
        t + BUDGET_MS["default"] + (one batch service time),
    because once it goes overdue it *triggers* a dispatch on the next tick, and
    it can be displaced from at most one batch's worth of higher-priority items.

So the worst-case wait is bounded by the budget plus a single batch service time,
independent of how much high-priority traffic arrives — no starvation.

The decision is split into pure helpers (``most_urgent``, ``should_dispatch``,
``assemble_batch``) so the live async loop, the virtual-time simulator, and the
unit tests all exercise the *same* logic rather than three lookalikes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from relay_core.queue import Queue
from relay_core.types import BatchAssignment, InferItem, Priority, WorkerState

# priority == latency budget, in milliseconds (DESIGN.md §8.1)
BUDGET_MS: dict[Priority, float] = {Priority.HIGH: 5.0, Priority.DEFAULT: 50.0}
MAX_BATCH = 16
TICK_MS = 1.0


@dataclass(slots=True)
class Head:
    priority: Priority
    item: InferItem


def slack_ms(head: Head, now_ms: float, budgets: Mapping[Priority, float]) -> float:
    """Time until this item is overdue. ``<= 0`` means past its budget."""
    return (head.item.enqueue_ts + budgets[head.priority]) - now_ms


def most_urgent(
    heads: list[Head], now_ms: float, budgets: Mapping[Priority, float]
) -> Optional[Head]:
    """The head item closest to (or past) its deadline across all queues."""
    if not heads:
        return None
    return min(heads, key=lambda h: slack_ms(h, now_ms, budgets))


def should_dispatch(
    total_queued: int,
    urgent: Optional[Head],
    now_ms: float,
    max_batch: int,
    budgets: Mapping[Priority, float],
) -> bool:
    """Dispatch iff a full batch is ready OR the most-urgent item is overdue."""
    if urgent is None:
        return False
    if total_queued >= max_batch:
        return True
    return slack_ms(urgent, now_ms, budgets) <= 0.0


def assemble_batch(
    queues: Mapping[Priority, Queue], cap: int
) -> list[InferItem]:
    """Fill up to ``cap`` items, high priority first, then top up with default."""
    if cap <= 0:
        return []
    batch: list[InferItem] = []
    for prio in (Priority.HIGH, Priority.DEFAULT):
        q = queues.get(prio)
        if q is None:
            continue
        want = cap - len(batch)
        if want <= 0:
            break
        batch += q.pop(want)
    return batch


def worker_batch_cap(worker: WorkerState, max_batch: int) -> int:
    """How many items this worker can take right now (DESIGN.md §8.1)."""
    return min(max_batch, worker.free_slots * worker.max_batch)


# --------------------------------------------------------------------------- #
# Live async loop. The simulator (bench/simulate.py) reimplements this control
# flow over a virtual clock using the *same* helpers above.
# --------------------------------------------------------------------------- #
async def batch_former(
    model: str,
    queues: Mapping[Priority, Queue],
    pick_worker: Callable[[str, str], Optional[WorkerState]],
    dispatch: Callable[[WorkerState, BatchAssignment], "asyncio.Future | None"],
    now_ms: Callable[[], float],
    make_batch_id: Callable[[], str],
    *,
    max_batch: int = MAX_BATCH,
    budgets: Mapping[Priority, float] = BUDGET_MS,
    tick_ms: float = TICK_MS,
    running: Callable[[], bool] = lambda: True,
) -> None:
    while running():
        now = now_ms()
        heads = [
            Head(prio, q.peek())  # type: ignore[arg-type]
            for prio, q in queues.items()
            if q.size() > 0 and q.peek() is not None
        ]
        urgent = most_urgent(heads, now, budgets)
        total = sum(q.size() for q in queues.values())

        if not should_dispatch(total, urgent, now, max_batch, budgets):
            await asyncio.sleep(tick_ms / 1000.0)
            continue

        assert urgent is not None
        worker = pick_worker(model, urgent.item.prefix_hash)
        if worker is None:  # backpressure: all workers capped/full
            await asyncio.sleep(tick_ms / 1000.0)
            continue

        cap = worker_batch_cap(worker, max_batch)
        items = assemble_batch(queues, cap)
        if not items:
            await asyncio.sleep(tick_ms / 1000.0)
            continue

        assignment = BatchAssignment(make_batch_id(), model, items)
        maybe = dispatch(worker, assignment)
        if maybe is not None:
            await maybe
