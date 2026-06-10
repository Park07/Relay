# Redis keyspace (DESIGN.md §5.4)

Redis is the **coordination / ephemeral** half of the state store (Postgres is the
durable, queryable half — see `deploy/postgres/init.sql` and ADR-5). Two stores
because they do different jobs.

| Key | Type | Contents | TTL |
|---|---|---|---|
| `queue:{model}:{prio}` | Stream | Pending requests. Consumer-group reads (`XREADGROUP`) give **at-least-once** delivery; `XACK` on completion; `XAUTOCLAIM` reclaims a dead worker's unacked entries after the lease timeout, reprocessed oldest-first by stream ID. Implemented in `services/scheduler/redis_stream_queue.py`. | — |
| `worker:{id}` | Hash | `status`, `models`, `capacity`, `last_heartbeat`. Refreshed by the worker heartbeat. | 15s |
| `workers:active` | Set | Live worker ids. `/readyz` checks `SCARD ≥ 1`. | — |
| `job:{id}` | Hash | `status`, `result`, `ts_queued`, `ts_done`. Read by `GET /v1/jobs/{id}`. | 1h |
| `ratelimit:{key}` | Hash | Token-bucket state (`tokens`, `ts`), mutated atomically by the Lua script in `services/gateway/ratelimit.py`. | rolling (≈ burst/rps + 1s) |
| `idem:{key}` | String | `job_id` for an `Idempotency-Key`, so client retries don't double-enqueue. | 24h |
| `apikey:{key_hash}` | String | JSON cache of a Postgres `api_keys` row (auth fast path). | 300s |
| `tokens:{job_id}` | Stream | Per-job token stream for SSE (`GET /v1/jobs/{id}/stream`). | short |
| `jobdone:{job_id}` | Pub/Sub | Completion notification so the sync path wakes immediately instead of polling. | — |

Notes:

- **At-least-once, not exactly-once.** Duplicates are made client-safe by the
  idempotency key, not prevented in the queue (DESIGN.md §5.2).
- **Bounded queues.** Admission control (`services/scheduler/admission.py`) sheds
  with `429` once total `XLEN` across priorities crosses the high-water mark;
  `relay_queue_depth` (the same depth) is what the HPA scales on (ADR-9).
- The rate-limit refill+consume **must** be one atomic Lua round trip — a
  read-modify-write across two calls races under concurrent requests for the
  same key. See `services/gateway/ratelimit.py`.
