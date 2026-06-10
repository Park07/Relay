"""Token-bucket rate limiting (DESIGN.md §5.1).

Two surfaces, one algorithm:

  * ``token_bucket_step`` — a **pure** function: given the prior bucket state and
    the current time, return the new state and whether the request is allowed. It
    has no I/O, so it is trivially unit-testable and is the single source of truth
    for the refill-then-consume math.
  * ``RATE_LIMIT_LUA`` — the same arithmetic as a Redis Lua script. In production
    the gateway runs this server-side so refill+consume happen in **one atomic
    round trip**. Doing it as read-modify-write across two calls would race under
    concurrent requests for the same key (DESIGN.md §5.1 calls this out
    explicitly), so the Lua script is not an optimization — it is correctness.

State is two numbers: ``tokens`` (current fill) and ``ts`` (last refill time, in
seconds). A bucket of capacity ``burst`` refills at ``rps`` tokens/sec and a
request costs one token.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BucketState:
    tokens: float
    ts: float  # seconds


@dataclass(slots=True)
class BucketResult:
    allowed: bool
    state: BucketState
    retry_after_s: float  # 0.0 when allowed; else time until one token refills


def token_bucket_step(
    state: BucketState | None,
    now_s: float,
    rps: float,
    burst: float,
    cost: float = 1.0,
) -> BucketResult:
    """Refill since ``state.ts`` then try to consume ``cost`` tokens.

    A fresh key (``state is None``) starts full, which is the usual, friendly
    choice: a client's first request is never rejected.
    """
    if burst <= 0:
        burst = rps  # default the bucket depth to one second of capacity
    if state is None:
        state = BucketState(tokens=burst, ts=now_s)

    # Refill: add rps tokens per elapsed second, clamped to the burst ceiling.
    elapsed = max(0.0, now_s - state.ts)
    tokens = min(burst, state.tokens + elapsed * rps)

    if tokens >= cost:
        return BucketResult(True, BucketState(tokens - cost, now_s), 0.0)

    # Not enough tokens: reject and tell the caller when to retry.
    deficit = cost - tokens
    retry_after = deficit / rps if rps > 0 else float("inf")
    return BucketResult(False, BucketState(tokens, now_s), retry_after)


# The Lua mirror of token_bucket_step. KEYS[1] = ratelimit:{key};
# ARGV = now_s, rps, burst, cost. Returns {allowed(0|1), tokens, retry_after_s}.
# Stored as a hash {tokens, ts} with a rolling TTL so idle keys evict themselves.
RATE_LIMIT_LUA = r"""
local key   = KEYS[1]
local now   = tonumber(ARGV[1])
local rps   = tonumber(ARGV[2])
local burst = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])
if burst <= 0 then burst = rps end

local data   = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts     = tonumber(data[2])
if tokens == nil then tokens = burst; ts = now end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(burst, tokens + elapsed * rps)

local allowed = 0
local retry = 0.0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
else
  retry = (cost - tokens) / rps
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
-- Rolling TTL: time to refill a full bucket, so abandoned keys disappear.
local ttl = math.ceil(burst / rps) + 1
redis.call('EXPIRE', key, ttl)
return {allowed, tostring(tokens), tostring(retry)}
"""
