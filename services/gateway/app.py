"""Relay Gateway — the only public surface (DESIGN.md §5.1, §7.1).

Stateless and horizontally scalable: auth, validation, rate limiting, request
intake, and result delivery. All intelligence is downstream in the scheduler;
this layer is intentionally thin.

  REST surface (DESIGN.md §7.1):
    POST /v1/infer            sync→200 / async→202 {job_id}
    GET  /v1/jobs/{id}        job status + result
    GET  /v1/jobs/{id}/stream SSE token stream
    GET  /v1/models           models available / loaded per worker
    GET  /healthz             liveness
    GET  /readyz              readiness (Redis reachable, ≥1 worker registered)
    GET  /metrics             Prometheus exposition

  Auth: `Authorization: Bearer <api_key>`; key rows live in Postgres, cached in
  Redis (DESIGN.md §5.4). Rate limit: atomic Redis Lua token bucket (ratelimit.py).
  Idempotency: optional `Idempotency-Key` → `idem:{key}` → job_id (TTL 24h).

RUNS LIVE only with Redis + the scheduler reachable (see deploy/compose). The
request-handling control flow, schema validation, rate-limit/idempotency wiring,
and metrics are all real; the Redis/Postgres/scheduler handles are injected so
the surface is unit-testable without those services standing up.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional, Protocol

from relay_core.metrics import REQUEST_DURATION, REQUEST_TOTAL
from relay_core.types import InferItem, InferParams, Priority, prefix_hash_of
from services.gateway.ratelimit import RATE_LIMIT_LUA
from services.gateway.schemas import (
    InferAcceptedResponse,
    InferRequest,
    InferSyncResponse,
    JobStatus,
    Mode,
)

HIGH_WATERMARK = int(os.getenv("RELAY_QUEUE_HIGH_WATERMARK", "10000"))


# --------------------------------------------------------------------------- #
# Backplane ports — injected so the gateway is testable without real services.
# The production wiring (redis.asyncio + the scheduler's enqueue) implements
# these Protocols; tests pass fakes.
# --------------------------------------------------------------------------- #
class RatePolicy(Protocol):
    async def check(self, api_key: str, rps: float, burst: float) -> tuple[bool, float]: ...


class Backplane(Protocol):
    async def ping(self) -> bool: ...
    async def active_workers(self) -> int: ...
    async def lookup_api_key(self, token: str) -> Optional[dict]: ...
    async def idem_get(self, key: str) -> Optional[str]: ...
    async def idem_put(self, key: str, job_id: str, ttl_s: int = 86_400) -> None: ...
    async def queue_depth(self, model: str) -> int: ...
    async def enqueue(self, item: InferItem, model: str, mode: str) -> None: ...
    async def get_job(self, job_id: str) -> Optional[dict]: ...
    async def wait_job(self, job_id: str, timeout_s: float) -> Optional[dict]: ...
    async def models(self) -> list[dict]: ...
    async def stream_tokens(self, job_id: str): ...  # -> AsyncIterator[str]


def _to_item(req: InferRequest, job_id: str) -> InferItem:
    return InferItem(
        request_id=job_id,
        input=req.input,
        params=InferParams(
            max_tokens=req.params.max_tokens,
            temperature=req.params.temperature,
            top_p=req.params.top_p,
            stream=req.params.stream,
        ),
        prefix_hash=prefix_hash_of(req.input),
        enqueue_ts=time.monotonic() * 1000.0,
        priority=Priority.HIGH if req.priority.value == "high" else Priority.DEFAULT,
    )


def create_app(backplane: Backplane, rate: RatePolicy):
    """Build the FastAPI app around injected service ports.

    Importing FastAPI is deferred to here so the module imports cleanly in the
    bench-only sandbox (where FastAPI/uvicorn aren't installed); the live entry
    point in ``__main__`` constructs the real backplane and calls this.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse

    app = FastAPI(title="Relay Gateway", version="0.1.0")

    async def auth(authorization: str = Header(default="")) -> dict:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        key = await backplane.lookup_api_key(token)
        if key is None:
            raise HTTPException(401, "invalid api key")
        return key

    @app.post("/v1/infer")
    async def infer(
        req: InferRequest,
        request: Request,
        key: dict = Depends(auth),
        idempotency_key: str = Header(default=""),
    ):
        # Rate limit (atomic token bucket, keyed on the API key).
        rps = float(key.get("rps_limit", 10))
        allowed, retry_after = await rate.check(key["key_hash"], rps, rps)
        if not allowed:
            REQUEST_TOTAL.labels(req.model, "429").inc()
            raise HTTPException(429, "rate limited", headers={"Retry-After": f"{retry_after:.2f}"})

        # Idempotency: a repeated key returns the original job rather than
        # enqueuing a duplicate (DESIGN.md §5.1).
        if idempotency_key:
            prior = await backplane.idem_get(idempotency_key)
            if prior is not None:
                return InferAcceptedResponse(job_id=prior, status="duplicate")

        # Admission control / backpressure: bounded queues (DESIGN.md §5.2).
        if await backplane.queue_depth(req.model) >= HIGH_WATERMARK:
            REQUEST_TOTAL.labels(req.model, "429").inc()
            raise HTTPException(429, "server overloaded, retry later")

        job_id = str(uuid.uuid4())
        if idempotency_key:
            await backplane.idem_put(idempotency_key, job_id)

        item = _to_item(req, job_id)
        await backplane.enqueue(item, req.model, req.mode.value)

        if req.mode == Mode.async_:
            REQUEST_TOTAL.labels(req.model, "202").inc()
            return InferAcceptedResponse(job_id=job_id)

        # Sync: wait for the result (bounded), then return it.
        started = time.monotonic()
        result = await backplane.wait_job(job_id, timeout_s=60.0)
        REQUEST_DURATION.labels(req.model, "sync").observe(time.monotonic() - started)
        if result is None:
            REQUEST_TOTAL.labels(req.model, "504").inc()
            raise HTTPException(504, "inference timed out")
        REQUEST_TOTAL.labels(req.model, "200").inc()
        return InferSyncResponse(
            job_id=job_id,
            model=req.model,
            output=result.get("output"),
            cache_hit=bool(result.get("cache_hit", False)),
            queue_wait_ms=float(result.get("queue_wait_ms", 0.0)),
            inference_ms=float(result.get("inference_ms", 0.0)),
            total_ms=float(result.get("total_ms", 0.0)),
        )

    @app.get("/v1/jobs/{job_id}")
    async def job(job_id: str, key: dict = Depends(auth)) -> JobStatus:
        rec = await backplane.get_job(job_id)
        if rec is None:
            raise HTTPException(404, "unknown job")
        return JobStatus(job_id=job_id, **rec)

    @app.get("/v1/jobs/{job_id}/stream")
    async def stream(job_id: str, key: dict = Depends(auth)):
        rec = await backplane.get_job(job_id)
        if rec is None:
            raise HTTPException(404, "unknown job")

        async def event_source():
            async for token in backplane.stream_tokens(job_id):
                yield f"data: {token}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models(key: dict = Depends(auth)):
        return {"models": await backplane.models()}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        # Ready iff Redis is reachable AND at least one worker has registered.
        if not await backplane.ping():
            raise HTTPException(503, "redis unreachable")
        if await backplane.active_workers() < 1:
            raise HTTPException(503, "no workers registered")
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    # Expose the Lua source on the app so the live backplane can SCRIPT LOAD it
    # at startup (one EVALSHA per request thereafter).
    app.state.rate_limit_lua = RATE_LIMIT_LUA
    return app


if __name__ == "__main__":  # pragma: no cover - live entry point
    # Real wiring lives in services/gateway/backplane_redis.py (needs Redis +
    # scheduler). Kept out of import path so the module loads in the sandbox.
    import uvicorn

    from services.gateway.backplane_redis import RedisBackplane, RedisRatePolicy

    bp = RedisBackplane.from_env()
    app = create_app(bp, RedisRatePolicy(bp))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
