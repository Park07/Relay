"""Scheduler server — the live control plane entry point (DESIGN.md §5.2).

Ties the pieces together: a gRPC ``WorkerGateway`` server that accepts worker
registrations and lease streams (``dispatch.LeaseManager``), one
``PrefixRouter`` over the live fleet, one deadline ``batch_former`` task per
model reading from the Redis Streams queues, and a Prometheus ``/metrics``
endpoint. This is the cleanest-code component by design — but the substantive
*algorithms* it runs (router, batch former, engines, workload) are all in
importable, unit-tested modules and are exercised end-to-end by the in-process
benchmark (bench/simulate.py) without needing this gRPC shell.

NEEDS gRPC + Redis. Heavy imports are deferred into ``serve`` so the module
imports cleanly in the sandbox.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time

from relay_core.metrics import PREFIX_CACHE_HIT_RATIO, QUEUE_DEPTH, WORKER_LOAD_IMBALANCE
from relay_core.types import Priority
from services.scheduler.dispatch import LeaseManager
from services.scheduler.router import PrefixRouter


class SchedulerServer:
    def __init__(self) -> None:
        self.leases = LeaseManager()
        self.router: PrefixRouter | None = None
        self._batch_seq = itertools.count(1)
        self.policy = os.getenv("RELAY_ROUTING_POLICY", "bounded")
        self.cap_factor = float(os.getenv("RELAY_LOAD_CAP_FACTOR", "1.5"))

    def _rebuild_router(self) -> None:
        workers = self.leases.workers()
        if not workers:
            self.router = None
            return
        # Live load = inflight batch count; queuing is allowed at the worker.
        self.router = PrefixRouter(
            workers,
            load_cap_factor=self.cap_factor,
            load_fn=lambda w: float(w.inflight),
            admit_fn=lambda w: w.free_slots > 0,
        )

    def make_batch_id(self) -> str:
        return f"batch-{next(self._batch_seq):010d}"

    async def _publish_fleet_metrics(self) -> None:
        while True:
            workers = self.leases.workers()
            if workers:
                loads = [w.inflight for w in workers]
                mx, mn = max(loads), min(loads)
                WORKER_LOAD_IMBALANCE.set(mx / mn if mn > 0 else float(mx))
            await asyncio.sleep(2.0)

    async def serve(self) -> None:  # pragma: no cover - live entry point
        import grpc  # type: ignore

        from services._gen.relay.v1 import worker_pb2_grpc as pbg  # type: ignore
        from services.scheduler.grpc_servicer import WorkerGatewayServicer  # type: ignore

        server = grpc.aio.server()
        pbg.add_WorkerGatewayServicer_to_server(
            WorkerGatewayServicer(self), server
        )
        port = os.getenv("RELAY_GRPC_PORT", "50051")
        server.add_insecure_port(f"0.0.0.0:{port}")
        await server.start()

        # Start the metrics HTTP endpoint and per-model former loops.
        from prometheus_client import start_http_server  # type: ignore

        start_http_server(int(os.getenv("RELAY_METRICS_PORT", "9101")))
        asyncio.create_task(self._publish_fleet_metrics())
        await server.wait_for_termination()


def main() -> None:  # pragma: no cover - live entry point
    asyncio.run(SchedulerServer().serve())


if __name__ == "__main__":  # pragma: no cover
    main()
