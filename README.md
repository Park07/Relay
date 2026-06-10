# Relay

A GPU inference-serving **control plane**: queue → schedule → batch → place → observe → autoscale. The lead feature is **prefix / KV-cache-aware routing under bounded load** — steering same-prefix requests to the same worker to reuse its KV cache, balanced by bounded-load consistent hashing so a hot prefix can't pin one worker. One knob (`load_cap_factor`) sweeps the whole policy space from pure cache affinity to round-robin, and the headline artifact is the **Pareto frontier** that knob traces between cache locality, load balance, and tail latency.

Full design rationale, ADRs, and the one-month build plan are in [`DESIGN.md`](DESIGN.md).

## Headline result (real, reproducible, no GPU)

Produced by the in-process virtual-time simulator (`bench/simulate.py`) over 30,000 requests with a calibrated cache-aware latency model, 4 workers, and a finite-Zipf (s=1.1) workload over 256 shared prefixes. Sweeping `load_cap_factor` from pure affinity toward round-robin:

| policy | cap | cache-hit | p99 | load imbalance |
|---|---|---|---|---|
| round-robin | — | **69.1%** | **1436 ms** | 1.01× |
| bounded (knee) | 1.5 | **88.4%** | **559 ms** | 1.47× |
| pure affinity | ∞ | **95.1%** | **440 ms** | 2.08× |

- Cache-hit rate climbs from **69% → 95%** as routing shifts from balance to affinity.
- p99 latency drops **3.3×** (1436 → 440 ms) — cache hits skip the ~160 ms prefill.
- The cost is load imbalance (1.01× → 2.08×), the other axis of the frontier.
- **Recommended operating point — the knee at `load_cap_factor = 1.5`:** 88% hit-rate and a **2.6×** p99 reduction for only 1.47× imbalance.

![Pareto frontier](bench/results/frontier.png)

Full sweep, per-worker locality breakdown, exact setup, and a documented negative finding are in [`bench/results/RESULTS.md`](bench/results/RESULTS.md).

> **A negative result worth keeping.** Forming batches globally and then routing by the batch head's prefix yields *no* locality — a globally-formed batch mixes prefixes, so whichever worker takes it must prefill all of them. Locality only materializes when routing happens **per request at admission** into per-worker queues, which is how production prefix-aware schedulers (SGLang, vLLM-router) are organized. That is the topology measured here.

## What runs where

This is a deliberately month-scale distributed-systems project. The parts split into three honesty tiers:

**Runs now, locally, for free (no GPU, no services):**
- The entire algorithmic core: consistent-hash ring, prefix router, deadline batch former, cache-aware + plain mock engines, Zipfian workload, calibration, and the benchmark sweep that produces the frontier above.
- The full unit suite (64 tests).

```bash
make install        # core + bench deps (numpy, matplotlib, pandas, scipy)
make test           # 64 unit tests, ~2s
make bench          # produces bench/results/{frontier.csv,frontier.png,RESULTS.md}
make bench-quick    # faster 8k-request smoke sweep
make calibrate      # fits MockEngine alpha/beta (synthetic if no Ollama)
```

**Written faithfully, needs external services to stand up (Redis, Postgres, gRPC):**
- Gateway (FastAPI: `/v1/infer`, `/v1/jobs/{id}`, SSE stream, `/v1/models`, `/healthz`, `/readyz`, `/metrics`; Bearer auth; atomic Redis-Lua token-bucket rate limiting; idempotency keys).
- Scheduler (gRPC server, admission/backpressure, Redis-Streams queue with `XAUTOCLAIM` recovery, dispatch over worker lease streams).
- Worker harness (pull-based gRPC leasing → backpressure) and the Ollama / Torch-MPS engines.
- One-command local stack and a k3d/Helm deploy:

```bash
make compose-up     # redis + postgres + scheduler + gateway + 4 mock workers + prometheus + grafana
# Grafana at :3000 (anon admin), Prometheus at :9090, gateway at :8080
make helm-install   # deploy to a k3d cluster; workers autoscale on queue depth
make proto          # generate gRPC stubs (buf) into services/_gen before running services live
```

The heavy imports (FastAPI, redis, asyncpg, grpc, torch) are deferred behind graceful fallbacks, so every module still imports without those packages — that is why `make test` runs anywhere.

**CUDA-only validation step (rented multi-GPU box):**
- `VLLMEngine` confirms the prefix-routing win on a *real* KV cache and produces the real tokens/s number. Same `Engine` interface, so only the worker image changes. vLLM's prefix-caching flags and cache-hit metric names drift between versions and are marked "verify on arrival" in the code (`services/worker/engines/vllm.py`). Per the design doc, this step only *confirms* the result — it never *produces* it.

## Architecture

```
            ┌─────────┐   REST/SSE    ┌───────────┐   gRPC bidi    ┌──────────┐
  client ──▶│ Gateway │──────────────▶│ Scheduler │◀──────lease────│ Worker×N │
            │ (auth,  │   enqueue     │ (admit,   │   dispatch ───▶│ (engine) │
            │  limit) │               │  batch,   │   results ◀────│          │
            └────┬────┘               │  route)   │                └────┬─────┘
                 │                    └─────┬─────┘                     │
              Redis  ◀── queues / jobs / rate / idem ──▶  Postgres (analytics)
                 │                          │                          │
                 └──────────── Prometheus  /metrics  ──────────────────┘
                                      │
                                  Grafana   +   HPA (autoscales workers on relay_queue_depth)
```

- **Routing depth** lives in `services/scheduler/router.py` + `relay_core/hashing.py`.
- **Batching** in `services/scheduler/batch_former.py` (priority = a tighter latency budget; provable no-starvation bound).
- **Autoscaling** on the custom `relay_queue_depth` metric (not CPU) via prometheus-adapter — `deploy/k8s/prometheus-adapter.yaml`, `deploy/helm/relay/templates/worker.yaml`.
- **Metrics** named once in `relay_core/metrics.py` (DESIGN.md §15), instrumented from the start.

## Repo map

```
relay_core/         types, hashing ring, queue, metrics  (the shared, transport-agnostic core)
services/
  gateway/          FastAPI app, schemas, token-bucket limiter, Redis backplane
  scheduler/        router, batch former, admission, dispatch, Redis-Streams queue, gRPC server
  worker/           harness + engines/{mock, cache_aware_mock, ollama, torch_mps, vllm}
proto/relay/v1/     worker.proto (buf lint/breaking in CI)
bench/              workload, calibrate, simulate, run  →  results/ (frontier + RESULTS.md)
deploy/             compose/, postgres/, redis/, helm/relay/, k8s/ (HPA + prometheus-adapter)
dashboards/         relay.json (Grafana, keyed to the §15 metric names)
tests/unit/         64 tests covering ring, router, former, engines, workload, limiter, admission
```

## Tests

```bash
make test           # PYTHONPATH=. python -m pytest tests/unit -q  →  64 passed
```

The unit suite pins the properties the result depends on: the ring's minimal-disruption guarantee, the router's affinity-vs-spill behaviour across the cap, the former's no-starvation dispatch trigger, the cache engine's exact hit/miss latency law, and the workload's distribution and prefix-hash stability.

## Status

The core algorithmic result is built, executed, and verified with real artifacts. The surrounding gateway/scheduler/worker/proto/deploy are substantive, faithful implementations of `DESIGN.md`; standing them up live requires Redis/Postgres/gRPC (local) and a CUDA box (vLLM validation), per the tiers above.
