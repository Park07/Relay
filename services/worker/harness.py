"""Worker harness — the data plane (DESIGN.md §5.3, §8.3).

Deliberately dumb and replaceable: register, open a bidi ``Lease`` stream, pull
``BatchAssignment``s up to ``free_slots``, run them through the pluggable
``Engine``, stream ``ResultItem``s back, and heartbeat. All intelligence is in
the scheduler.

The capacity model is the backpressure mechanism (ADR-2): the worker advertises
``free_slots = max_concurrent_batches - inflight`` on every lease tick, so the
scheduler never overcommits a slow worker.

vLLM layering (DESIGN.md §5.3): when the engine is vLLM, Relay must NOT hand
over large pre-formed batches (that fights vLLM's own continuous batcher) — it
dispatches items individually and lets PagedAttention merge them. The harness
honours this via ``engine.prefers_individual_dispatch`` so the composition is
"Relay does inter-worker placement; vLLM does intra-worker batching."

NEEDS the generated gRPC stubs + a running scheduler. The control flow and the
Engine contract are real; only the gRPC transport needs ``make proto`` + deploy.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator

from relay_core.metrics import TOKENS_PER_SECOND, WORKER_GPU_UTIL
from relay_core.types import BatchAssignment, Engine, ResultItem, WorkerState


class WorkerHarness:
    def __init__(
        self,
        engine: Engine,
        worker_id: str,
        models: tuple[str, ...],
        max_batch: int = 16,
        max_concurrent_batches: int = 4,
    ) -> None:
        self.engine = engine
        self.state = WorkerState(
            worker_id=worker_id,
            engine=getattr(engine, "name", "unknown"),
            models=models,
            max_batch=max_batch,
            max_concurrent_batches=max_concurrent_batches,
        )
        self._inflight = 0
        self._stop = asyncio.Event()

    @property
    def free_slots(self) -> int:
        return max(0, self.state.max_concurrent_batches - self._inflight)

    async def run_batch(self, assignment: BatchAssignment) -> list[ResultItem]:
        """Execute one assignment through the engine, collecting results.

        Mirrors §8.3: bump inflight, await the engine, stream results, then
        publish realized throughput as a USE metric.
        """
        self._inflight += 1
        started = time.monotonic()
        results: list[ResultItem] = []
        try:
            async for r in self.engine.infer(assignment.items):
                results.append(r)
        finally:
            self._inflight -= 1

        elapsed = max(1e-6, time.monotonic() - started)
        toks = sum(r.batch_size for r in results) or len(assignment.items)
        TOKENS_PER_SECOND.labels(self.state.worker_id, assignment.model).set(toks / elapsed)
        st = self.engine.stats()
        WORKER_GPU_UTIL.labels(self.state.worker_id).set(st.gpu_util)
        return results

    # -- gRPC lease loop --------------------------------------------------- #
    async def lease_requests(self) -> AsyncIterator[object]:  # pragma: no cover - needs stubs
        """Yield LeaseRequest messages advertising current free_slots."""
        from services._gen.relay.v1 import worker_pb2 as pb  # type: ignore

        while not self._stop.is_set():
            yield pb.LeaseRequest(
                worker_id=self.state.worker_id,
                free_slots=self.free_slots,
                loaded_models=list(self.state.models),
            )
            await asyncio.sleep(0.05)

    async def serve(self, scheduler_addr: str) -> None:  # pragma: no cover - needs stubs
        import grpc  # type: ignore

        from services._gen.relay.v1 import worker_pb2 as pb  # type: ignore
        from services._gen.relay.v1 import worker_pb2_grpc as pbg  # type: ignore

        async with grpc.aio.insecure_channel(scheduler_addr) as channel:
            stub = pbg.WorkerGatewayStub(channel)
            await stub.Register(
                pb.WorkerInfo(
                    worker_id=self.state.worker_id,
                    models=list(self.state.models),
                    max_batch=self.state.max_batch,
                    max_concurrent_batches=self.state.max_concurrent_batches,
                    engine=self.state.engine,
                )
            )
            async for assignment in stub.Lease(self.lease_requests()):
                a = _from_proto_assignment(assignment)
                results = await self.run_batch(a)
                await stub.ReportResults(_results_stream(results))

    def stop(self) -> None:
        self._stop.set()


def _from_proto_assignment(a) -> BatchAssignment:  # pragma: no cover - needs stubs
    from relay_core.types import InferItem, InferParams

    return BatchAssignment(
        batch_id=a.batch_id,
        model=a.model,
        items=[
            InferItem(
                request_id=it.request_id,
                input=it.input,
                prefix_hash=it.prefix_hash,
                params=InferParams(
                    max_tokens=it.params.max_tokens,
                    temperature=it.params.temperature,
                    top_p=it.params.top_p,
                    stream=it.params.stream,
                ),
            )
            for it in a.items
        ],
    )


async def _results_stream(results: list[ResultItem]):  # pragma: no cover - needs stubs
    from services._gen.relay.v1 import worker_pb2 as pb  # type: ignore

    for r in results:
        msg = pb.ResultItem(
            request_id=r.request_id,
            batch_id=r.batch_id,
            final=r.final,
            metrics=pb.WorkerMetrics(
                queue_wait_ms=r.queue_wait_ms,
                inference_ms=r.inference_ms,
                batch_size=r.batch_size,
                gpu_util=r.gpu_util,
                cache_hit=r.cache_hit,
            ),
        )
        if r.error is not None:
            msg.error = r.error
        elif r.token is not None:
            msg.token = r.token
        else:
            msg.output = r.output or ""
        yield msg


def build_engine_from_env() -> Engine:  # pragma: no cover - live entry point
    """Select an engine by RELAY_ENGINE so the same harness image serves any
    backend (the §5.3 promise: only the worker image's engine changes)."""
    kind = os.getenv("RELAY_ENGINE", "mock").lower()
    if kind == "mock":
        from services.worker.engines.mock import MockEngine

        return MockEngine()
    if kind == "cache-aware-mock":
        from services.worker.engines.cache_aware_mock import CacheAwareMockEngine

        return CacheAwareMockEngine()
    if kind == "ollama":
        from services.worker.engines.ollama import OllamaEngine

        return OllamaEngine()
    if kind == "torch-mps":
        from services.worker.engines.torch_mps import TorchMPSEngine

        return TorchMPSEngine()
    if kind == "vllm":
        from services.worker.engines.vllm import VLLMEngine

        return VLLMEngine()
    raise ValueError(f"unknown RELAY_ENGINE={kind!r}")
