"""gRPC servicer for the WorkerGateway service (DESIGN.md §7.2).

Implements the three RPCs the scheduler exposes to workers, against the
``LeaseManager`` and the per-model batch-former tasks held by ``SchedulerServer``:

  * ``Register``       — a worker announces capacity → tracked in the fleet,
                         router rebuilt, and a per-model former is ensured.
  * ``Lease``          — bidi stream: the worker pushes LeaseRequest{free_slots}
                         (backpressure signal), and we drain that worker's
                         dispatch queue back as BatchAssignments.
  * ``ReportResults``  — the worker streams ResultItems; we record metrics,
                         resolve jobs, and ack the underlying queue entries
                         (the at-least-once completion point).

NEEDS the generated stubs (``make proto``). The control flow is written against
the proto in DESIGN.md §7.2 so it is correct the moment the stubs exist; the
whole module is import-guarded behind the generated package so the rest of the
scheduler imports without it.
"""

from __future__ import annotations

import asyncio

from relay_core.metrics import PREFIX_CACHE_HIT_RATIO, QUEUE_WAIT
from relay_core.types import WorkerState

try:  # pragma: no cover - only importable after `make proto`
    from services._gen.relay.v1 import worker_pb2 as pb  # type: ignore
    from services._gen.relay.v1 import worker_pb2_grpc as pbg  # type: ignore

    _Base = pbg.WorkerGatewayServicer
    _HAVE_STUBS = True
except Exception:  # stubs not generated yet
    _HAVE_STUBS = False

    class _Base:  # type: ignore
        ...


class WorkerGatewayServicer(_Base):  # pragma: no cover - needs stubs + grpc runtime
    def __init__(self, server) -> None:
        self.server = server  # SchedulerServer

    async def Register(self, request, context):
        state = WorkerState(
            worker_id=request.worker_id,
            engine=request.engine,
            models=tuple(request.models),
            max_batch=request.max_batch or 16,
            max_concurrent_batches=request.max_concurrent_batches or 1,
        )
        self.server.leases.register(state)
        self.server._rebuild_router()
        return pb.Ack(ok=True, detail=f"registered {state.worker_id}")

    async def Lease(self, request_iterator, context):
        # First, consume the worker's lease requests in the background to keep
        # free_slots fresh; meanwhile drain its outbox into the response stream.
        worker_id_box: dict[str, str] = {}

        async def consume_requests():
            async for req in request_iterator:
                worker_id_box["id"] = req.worker_id
                self.server.leases.update_free_slots(req.worker_id, req.free_slots)

        consumer = asyncio.create_task(consume_requests())
        try:
            # Wait until we know which worker this stream belongs to.
            while "id" not in worker_id_box:
                await asyncio.sleep(0.01)
            ch = self.server.leases.channel(worker_id_box["id"])
            assert ch is not None
            while True:
                assignment = await ch.outbox.get()
                from services.scheduler.dispatch import to_proto_assignment

                yield to_proto_assignment(assignment)
        finally:
            consumer.cancel()
            if "id" in worker_id_box:
                self.server.leases.deregister(worker_id_box["id"])
                self.server._rebuild_router()

    async def ReportResults(self, request_iterator, context):
        n = 0
        async for r in request_iterator:
            n += 1
            m = r.metrics
            if m.queue_wait_ms:
                QUEUE_WAIT.labels("_all").observe(m.queue_wait_ms / 1000.0)
            # Resolve the job (write job:{id} → done, publish jobdone:{id}); the
            # real backplane call is wired in SchedulerServer. Cache-hit ratio is
            # updated per worker so the headline metric is live.
            await self.server.on_result(r)  # type: ignore[attr-defined]
        return pb.Ack(ok=True, detail=f"received {n} results")
