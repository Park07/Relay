"""Dispatch — push a formed BatchAssignment to a worker over its gRPC lease
stream (DESIGN.md §5.2 "Dispatch", §7.2, ADR-2).

The scheduler is the gRPC server. Each registered worker holds an open bidi
``Lease`` stream; ``LeaseManager`` tracks one outbound queue per worker and the
worker's advertised ``free_slots``. The batch former calls ``dispatch(worker,
assignment)``, which drops the assignment on that worker's queue; the gRPC Lease
handler drains the queue into the stream as the worker pulls. Pull-based leasing
is what gives backpressure for free — we never push more than ``free_slots``.

NEEDS the generated stubs (``make proto`` → services/_gen/relay/v1/...). This
module converts between relay_core dataclasses and the generated protobuf
messages at the edge; the conversion helpers are written against the proto in
DESIGN.md §7.2 so they are correct the moment the stubs exist.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from relay_core.metrics import BATCH_SIZE, WORKER_INFLIGHT
from relay_core.types import BatchAssignment, WorkerState


@dataclass
class WorkerChannel:
    state: WorkerState
    outbox: "asyncio.Queue[BatchAssignment]" = field(default_factory=asyncio.Queue)


class LeaseManager:
    """Server-side registry of connected workers and their outbound queues."""

    def __init__(self) -> None:
        self._channels: dict[str, WorkerChannel] = {}

    def register(self, state: WorkerState) -> WorkerChannel:
        ch = WorkerChannel(state=state)
        self._channels[state.worker_id] = ch
        return ch

    def deregister(self, worker_id: str) -> None:
        self._channels.pop(worker_id, None)

    def update_free_slots(self, worker_id: str, free_slots: int) -> None:
        ch = self._channels.get(worker_id)
        if ch is not None:
            # inflight is derived from advertised free_slots (ADR-2 backpressure).
            ch.state.inflight = max(0, ch.state.max_concurrent_batches - free_slots)
            WORKER_INFLIGHT.labels(worker_id).set(ch.state.inflight)

    def channel(self, worker_id: str) -> WorkerChannel | None:
        return self._channels.get(worker_id)

    def workers(self) -> list[WorkerState]:
        return [ch.state for ch in self._channels.values()]

    async def dispatch(self, worker: WorkerState, assignment: BatchAssignment) -> None:
        ch = self._channels.get(worker.worker_id)
        if ch is None:
            raise KeyError(f"worker {worker.worker_id} not connected")
        await ch.outbox.put(assignment)
        ch.state.inflight += 1  # optimistic; reconciled by the next lease tick
        BATCH_SIZE.labels(assignment.model).observe(len(assignment.items))
        WORKER_INFLIGHT.labels(worker.worker_id).set(ch.state.inflight)


# -- proto edge conversions (valid against DESIGN.md §7.2 once stubs exist) -- #
def to_proto_assignment(assignment: BatchAssignment):  # pragma: no cover - needs stubs
    from services._gen.relay.v1 import worker_pb2 as pb  # type: ignore

    return pb.BatchAssignment(
        batch_id=assignment.batch_id,
        model=assignment.model,
        items=[
            pb.InferItem(
                request_id=it.request_id,
                input=it.input,
                prefix_hash=it.prefix_hash,
                params=pb.InferParams(
                    max_tokens=it.params.max_tokens,
                    temperature=it.params.temperature,
                    top_p=it.params.top_p,
                    stream=it.params.stream,
                ),
            )
            for it in assignment.items
        ],
    )
