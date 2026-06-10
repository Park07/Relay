"""Live backplane for the gateway: Redis (coordination) + Postgres (durable).

Implements the ``Backplane`` and ``RatePolicy`` ports declared in app.py against
the Redis key layout in DESIGN.md §5.4:

    queue:{model}:{prio}  Stream   pending requests (consumer group → at-least-once)
    worker:{id}           Hash     status/models/capacity/heartbeat (TTL 15s)
    workers:active        Set      live worker ids
    job:{id}              Hash     status/result/timestamps (TTL 1h)
    ratelimit:{key}       String   token-bucket state (via the Lua script)
    idem:{key}            String   job_id (TTL 24h)

NEEDS SERVICES: this module imports ``redis.asyncio`` and ``asyncpg`` and only
runs when Redis + Postgres are reachable (deploy/compose brings them up). It is
intentionally excluded from the default import path (app.py imports it lazily in
``__main__``) so the rest of the package imports cleanly in the bench sandbox.
"""

from __future__ import annotations

import json
import os
import time
from typing import AsyncIterator, Optional

from relay_core.types import InferItem
from services.gateway.ratelimit import RATE_LIMIT_LUA


class RedisBackplane:
    def __init__(self, redis, pg_pool=None) -> None:
        self.redis = redis
        self.pg = pg_pool
        self._rl_sha: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RedisBackplane":
        import redis.asyncio as aioredis  # type: ignore

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return cls(aioredis.from_url(url, decode_responses=True))

    async def rate_limit_sha(self) -> str:
        if self._rl_sha is None:
            self._rl_sha = await self.redis.script_load(RATE_LIMIT_LUA)
        return self._rl_sha

    # -- readiness --------------------------------------------------------- #
    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:
            return False

    async def active_workers(self) -> int:
        return int(await self.redis.scard("workers:active"))

    # -- auth (Postgres source of truth, Redis cache) ---------------------- #
    async def lookup_api_key(self, token: str) -> Optional[dict]:
        import hashlib

        key_hash = hashlib.sha256(token.encode()).hexdigest()
        cached = await self.redis.get(f"apikey:{key_hash}")
        if cached:
            return json.loads(cached)
        if self.pg is None:
            return None
        row = await self.pg.fetchrow(
            "SELECT key_hash, tenant_id, rps_limit FROM api_keys WHERE key_hash=$1",
            key_hash,
        )
        if row is None:
            return None
        rec = dict(row)
        await self.redis.set(f"apikey:{key_hash}", json.dumps(rec), ex=300)
        return rec

    # -- idempotency ------------------------------------------------------- #
    async def idem_get(self, key: str) -> Optional[str]:
        return await self.redis.get(f"idem:{key}")

    async def idem_put(self, key: str, job_id: str, ttl_s: int = 86_400) -> None:
        await self.redis.set(f"idem:{key}", job_id, ex=ttl_s, nx=True)

    # -- queue ------------------------------------------------------------- #
    async def queue_depth(self, model: str) -> int:
        total = 0
        for prio in ("high", "default"):
            total += int(await self.redis.xlen(f"queue:{model}:{prio}"))
        return total

    async def enqueue(self, item: InferItem, model: str, mode: str) -> None:
        prio = item.priority.value
        await self.redis.xadd(
            f"queue:{model}:{prio}",
            {
                "request_id": item.request_id,
                "input": item.input,
                "prefix_hash": item.prefix_hash,
                "max_tokens": item.params.max_tokens,
                "enqueue_ts": item.enqueue_ts,
                "mode": mode,
            },
        )
        await self.redis.hset(f"job:{item.request_id}", mapping={"status": "queued"})
        await self.redis.expire(f"job:{item.request_id}", 3600)

    # -- jobs -------------------------------------------------------------- #
    async def get_job(self, job_id: str) -> Optional[dict]:
        rec = await self.redis.hgetall(f"job:{job_id}")
        if not rec:
            return None
        for k in ("queue_wait_ms", "inference_ms", "total_ms"):
            if k in rec:
                rec[k] = float(rec[k])
        if "cache_hit" in rec:
            rec["cache_hit"] = rec["cache_hit"] in ("1", "true", "True")
        return rec

    async def wait_job(self, job_id: str, timeout_s: float) -> Optional[dict]:
        # Subscribe to a per-job completion channel the scheduler publishes to;
        # fall back to polling the hash so a missed publish still resolves.
        deadline = time.monotonic() + timeout_s
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"jobdone:{job_id}")
        try:
            while time.monotonic() < deadline:
                rec = await self.get_job(job_id)
                if rec and rec.get("status") in ("done", "error"):
                    return rec
                await pubsub.get_message(timeout=0.25)
        finally:
            await pubsub.unsubscribe(f"jobdone:{job_id}")
        return None

    async def stream_tokens(self, job_id: str) -> AsyncIterator[str]:
        # Workers push tokens to a per-job stream; relay them as SSE events.
        last_id = "0"
        while True:
            resp = await self.redis.xread({f"tokens:{job_id}": last_id}, block=2000, count=32)
            if not resp:
                rec = await self.get_job(job_id)
                if rec and rec.get("status") in ("done", "error"):
                    return
                continue
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    last_id = entry_id
                    if fields.get("final") in ("1", "true", "True"):
                        return
                    yield fields.get("token", "")

    async def models(self) -> list[dict]:
        ids = await self.redis.smembers("workers:active")
        by_model: dict[str, list[str]] = {}
        for wid in ids:
            info = await self.redis.hgetall(f"worker:{wid}")
            for m in (info.get("models", "")).split(",") if info else []:
                if m:
                    by_model.setdefault(m, []).append(wid)
        return [{"model": m, "loaded_on": ws} for m, ws in sorted(by_model.items())]


class RedisRatePolicy:
    """RatePolicy backed by the atomic Lua token bucket."""

    def __init__(self, backplane: RedisBackplane) -> None:
        self.bp = backplane

    async def check(self, api_key: str, rps: float, burst: float) -> tuple[bool, float]:
        sha = await self.bp.rate_limit_sha()
        now = time.time()
        allowed, _tokens, retry = await self.bp.redis.evalsha(
            sha, 1, f"ratelimit:{api_key}", now, rps, burst, 1.0
        )
        return bool(int(allowed)), float(retry)
