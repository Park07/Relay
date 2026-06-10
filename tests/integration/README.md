# Integration tests (need the live stack)

These exercise the wired system rather than pure logic, so they require the
docker-compose stack up (`make compose-up`): Redis, Postgres, the scheduler's
gRPC server, the gateway, and at least one worker.

Planned coverage (DESIGN.md §12):
- `POST /v1/infer` sync + async round-trips through gateway → scheduler →
  mock worker → result, asserting the response shape and that `requests`/`batches`
  rows land in Postgres.
- Rate-limit 429s under burst, and idempotency-key dedupe across retries.
- `XAUTOCLAIM` recovery: kill a worker mid-batch and assert its unacked stream
  entries are reclaimed and reprocessed (at-least-once).
- `/readyz` flips to 503 when Redis is down or zero workers are registered.

They are kept out of the default `make test` (the fast, dependency-free unit
suite) and run in a separate CI job once the stack is available.
