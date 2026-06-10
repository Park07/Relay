"""Worker process entry point (DESIGN.md §5.3).

Builds the engine selected by ``RELAY_ENGINE`` and runs the harness lease loop
against the scheduler. Exposes a Prometheus ``/metrics`` endpoint for the USE
signals. Heavy imports (gRPC, the chosen engine's deps) are deferred so this
module imports cleanly in the sandbox; the actual process needs the scheduler
and the engine's runtime.
"""

from __future__ import annotations

import asyncio
import os
import uuid


def main() -> None:  # pragma: no cover - live entry point
    from prometheus_client import start_http_server  # type: ignore

    from services.worker.harness import WorkerHarness, build_engine_from_env

    start_http_server(int(os.getenv("RELAY_METRICS_PORT", "9102")))

    engine = build_engine_from_env()
    models = tuple(
        m for m in os.getenv("RELAY_WORKER_MODELS", "qwen2.5:0.5b").split(",") if m
    )
    harness = WorkerHarness(
        engine=engine,
        worker_id=os.getenv("RELAY_WORKER_ID", f"w-{uuid.uuid4().hex[:8]}"),
        models=models,
        max_batch=int(os.getenv("MAX_BATCH", "16")),
        max_concurrent_batches=int(os.getenv("MAX_CONCURRENT_BATCHES", "4")),
    )

    async def run() -> None:
        await engine.load(models[0])
        await harness.serve(os.getenv("SCHEDULER_ADDR", "localhost:50051"))

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
