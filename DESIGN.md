# Relay — GPU Inference Serving Platform

Design document v1.1 · single-author project · target platform: MacBook Air M2 (dev) + cloud GPU (validation)
## Note: this is the original design spec; actual results are in the README and differ from the placeholder numbers; the soak/chaos tests are specced but not built
This is the original design specification (v1.1), written before implementation. It records design intent and rationale. Actual measured results — including the ~530-token crossover, the 69%→95% frontier, and the 2× A40 vLLM validation — are in README.md and differ from the illustrative placeholder numbers used here. Phases 0–2 and the depth feature (§5, §8) shipped; the live-stack soak and chaos tests (§12) are specified but not yet run
---

## Changelog (v1.0 → v1.1)

This revision sharpens v1.0 from "a broad serving control plane" into "a control plane with one genuinely deep, measured feature." The bones are unchanged; the focus is not.

- **Promoted prefix / KV-cache-aware routing to the single lead depth feature** (it was one of several Phase-5 options). It is the one play that directly rebuts the project's central weakness — "you're orchestrating the hard part, not building it" — because it forces engagement with engine-internal state (the KV cache) rather than treating the worker as a black box.
- **Added a calibrated `MockEngine`** (and a cache-aware variant) so the core algorithmic results are produced **locally, free, and reproducibly**. Cloud GPU now *confirms* the headline on real hardware rather than being its sole source — the result is no longer hostage to a rented afternoon.
- **Reframed reporting around a Pareto frontier** (cache locality vs load imbalance vs p99), not a single number. The frontier is the artifact that proves you understood the tradeoff rather than just enabling a flag.
- **Demoted the GPU-utilization headline.** On Apple Silicon, utilization is coarse (scraped from `powermetrics`/`asitop`). Throughput, p99, and prefix-cache hit-rate now lead; utilization is a CUDA-validated bonus.
- **Corrected reliability mechanics.** At-least-once delivery is achieved via Redis Streams **`XAUTOCLAIM`** reclaiming entries unacked past a lease timeout (reprocessed oldest-first by stream ID) — not "re-enqueue to the front of the queue," which is not a real Streams operation.
- **Folded request priority into the deadline-based batch former** (priority = a tighter latency budget) with a stated, provable no-starvation bound. The vestigial second mechanism is gone.
- **Clarified vLLM layering.** Relay does *inter-worker* batching, placement, and fleet autoscaling; vLLM does *intra-worker* continuous batching. They compose. Relay's own batching is demonstrated on the Mock/MPS engines, where there is no competing batcher to muddy the result.
- **Added a focused one-month execution plan (§17)** for the depth feature, ordered so a compressed final week still ships a complete result.

---

## 0. Purpose of this document

This is the spec I'd hand a junior engineer joining the project — it states *what* we're building, *why* each decision was made, and *how* success is measured. It is opinionated on purpose: a portfolio project that hedges every decision demonstrates nothing. Where there's a real fork in the road, the alternative is named and the tradeoff is stated (ADR section).

**Who this is for / why it exists.** The goal is not "a working demo." The goal is an artifact that reads, to a hiring engineer at a Korean AI-infra company (Lablup, FuriosaAI's platform team, Rebellions' AI-server team, Moreh), as *"this person has already done a smaller version of our job."* The serving-control-plane shape — queue, schedule, batch, place, observe, autoscale — is the most common backend shape at those companies. Every design choice below is justified against that goal, not against production-readiness at scale.

**What "depth" means here.** To a reviewer, depth is four things: you engaged the hard substrate, you built a non-trivial algorithm with a real tradeoff, you measured why it works, and you can explain it. Breadth (more endpoints, more features) is the trap that keeps a project at "another gateway." This document spends its ambition on **one** hard thing measured ruthlessly.

---

## 1. Problem statement

GPUs are expensive and idle GPUs are pure waste. A naive `POST /predict → model.forward()` server runs one request at a time, leaving the accelerator under 20% utilized and giving terrible throughput under concurrency. Real inference systems solve this with a **control plane** that decouples request ingestion from execution: requests are queued, **batched** to fill the device, scheduled across a fleet of workers, and the whole thing is observable and horizontally scalable.

**The one-sentence problem:** keep expensive GPUs busy under concurrent load — instead of ~20% idle — by batching inference requests, routing them intelligently across a worker fleet, and scaling that fleet with traffic.

**The one-sentence depth problem (the lead feature):** when many requests share a long prefix (the same system prompt, few-shot block, or RAG document), naive routing scatters them across workers so every worker wastefully recomputes that prefix and the KV-cache hit rate collapses; **route same-prefix requests to the same worker to reuse its cache, while balancing so a single hot prefix doesn't overload one worker.**

**Relay** is a minimal but real implementation of that control plane. It accepts inference requests, batches and routes them across a pool of GPU workers, exposes the results over REST and gRPC, and ships full observability. It is engine-agnostic: a worker can be backed by a calibrated mock (for benchmarking), PyTorch (Apple MPS), Ollama (Metal), or vLLM (CUDA), behind a single interface.

---

## 2. Goals and non-goals

### Goals

1. **Maximize accelerator utilization** under concurrent load via dynamic (continuous) batching.
2. **Decouple ingestion from execution** with an explicit control-plane / data-plane split.
3. **Be measurable end to end** — every latency, every queue, every batch, every cache hit is a metric from day one.
4. **Run fully on an M2 locally** for the control plane, the mock/CPU/MPS workers, and the *entire core algorithmic result*; require a cloud GPU only to validate the one CUDA-specific phase.
5. **Be Kubernetes-native** — deployable to a local k3d cluster with autoscaling driven by a custom metric.
6. **Demonstrate one deep feature** — prefix/KV-cache-aware routing with a measured locality↔load tradeoff (a Pareto frontier), the single thing this project is *about*.

### Non-goals

1. **Not** a production system. No multi-region, no HA control plane, no exactly-once semantics, no auth provider integration beyond API keys.
2. **Not** a new inference *engine*. We orchestrate vLLM/Ollama/PyTorch; we do not write CUDA kernels here (that's a separate portfolio project for the chip-runtime lane).
3. **Not** a model registry / training platform. Models are assumed pre-existing.
4. **Not** trying to beat vLLM's scheduler. We're demonstrating that we understand the *same class* of problem — and that we can build a routing layer that makes a fleet of vLLM workers more cache-efficient than naive round-robin would.

> **Scoping principle (repeated throughout):** a deep, measured Phase 0–2 plus one ruthlessly-measured deep feature beats a half-broken Phase 5. Build down before building out.

---

## 3. Success criteria

The project is "done enough to ship to a résumé" when:

- It serves real inference across **≥3 concurrent workers** with dynamic batching.
- There is a **benchmark report** showing throughput and p99 latency as functions of batch window and worker count, with before/after numbers for the headline optimization.
- The **lead depth feature is measured as a Pareto frontier**: prefix-cache hit-rate vs load imbalance vs p99, swept over the routing policy's spillover/replication knob, with naive round-robin and pure-affinity as the two reference endpoints.
- It **deploys to k3d** with one command and autoscales workers on queue depth.
- The README's first screen contains an architecture diagram and a results table/plot. (If a reader has to dig to find the numbers, the numbers don't exist.)

Note the deliberate ordering: the **prefix-cache hit-rate and the latency delta vs round-robin are the headline**, produced locally on the calibrated mock. GPU utilization is reported only as a CUDA-validated bonus, because on Apple Silicon it is coarse and not credible as a headline.

---

## 4. System architecture

```
                         ┌───────────────────────────────────────────┐
        REST / gRPC      │                  RELAY                     │
  clients ──────────────▶│                                            │
                         │   ┌──────────┐      ┌──────────────────┐   │
                         │   │ Gateway  │─────▶│    Scheduler     │   │
                         │   │ (FastAPI │      │  (control plane) │   │
                         │   │  + gRPC) │◀─────│  - admit         │   │
                         │   └────┬─────┘      │  - batch former  │   │
                         │        │            │  - PREFIX ROUTER │   │
                         │        │            │  - dispatch      │   │
                         │        │            └───────┬──────────┘   │
                         │        │                    │ gRPC stream  │
                         │   ┌────▼─────┐         ┌─────▼──────┐       │
                         │   │  Redis   │◀───────▶│  Workers   │       │
                         │   │ streams/ │         │  (N pods)  │       │
                         │   │ registry │         │  engine:   │       │
                         │   │ jobs     │         │  mock/mps/ │       │
                         │   └────┬─────┘         │  ollama/   │       │
                         │        │               │  vllm      │       │
                         │   ┌────▼─────┐          └─────┬──────┘       │
                         │   │ Postgres │                │ metrics     │
                         │   │ (durable │          ┌─────▼──────┐      │
                         │   │  logs)   │          │ Prometheus │      │
                         │   └──────────┘          │ + Grafana  │      │
                         └─────────────────────────┴────────────┴──────┘
```

**Request lifecycle (async path):**

1. Client `POST /v1/infer` → Gateway validates, authenticates, applies rate limit, writes a `request` row (status=`queued`), enqueues onto `queue:{model}:{priority}` in Redis, returns `202 {job_id}`.
2. Scheduler's batch-former, per model, pops items when *either* `MAX_BATCH` reached *or* the oldest item ages past its latency budget, then the **prefix router** picks a worker: same-prefix requests are steered to the same worker to maximize KV-cache reuse, subject to a bounded-load constraint. The batch is pushed over the worker's open gRPC lease stream.
3. Worker runs the batch through its engine, streams `ResultItem`s back; scheduler writes results to `job:{id}` and updates Postgres (status=`done`, `latency_ms`, `batch_id`, `cache_hit`).
4. Client polls `GET /v1/jobs/{id}` or subscribes to `GET /v1/jobs/{id}/stream` (SSE) for token streaming.

**Sync path:** same pipeline, but the Gateway holds the connection and long-polls `job:{id}` until terminal, with a timeout. Sync is a convenience wrapper over async, never a separate code path.

---

## 5. Component specifications

### 5.1 Gateway (API)

**Responsibility:** the only public surface. Auth, validation, rate limiting, request intake, result delivery. Stateless; horizontally scalable.

- **Stack:** Python 3.12, FastAPI, `uvicorn` (or `granian`), `pydantic v2` for schema validation, `grpclib`/`grpcio` for the gRPC edge.
- **Endpoints (REST):**

  | Method | Path | Purpose |
  |---|---|---|
  | `POST` | `/v1/infer` | Submit inference. `mode=sync\|async`, `priority`, `model`, `input`, `params`. Returns `200` (sync) or `202 {job_id}`. |
  | `GET` | `/v1/jobs/{id}` | Job status + result. |
  | `GET` | `/v1/jobs/{id}/stream` | SSE token stream (LLM workers). |
  | `GET` | `/v1/models` | Models available / currently loaded per worker. |
  | `GET` | `/healthz` | Liveness (process up). |
  | `GET` | `/readyz` | Readiness (Redis reachable, ≥1 worker registered). |
  | `GET` | `/metrics` | Prometheus exposition. |

- **Auth:** `Authorization: Bearer <api_key>`; keys live in Postgres, cached in Redis. No JWT/OAuth — out of scope (non-goal #1).
- **Rate limiting:** token-bucket per API key, implemented as an **atomic Redis Lua script** (refill + consume in one round trip — never read-modify-write across two calls, that races).
- **Idempotency:** optional `Idempotency-Key` header → `idem:{key}` maps to a `job_id` (TTL 24h) so client retries don't double-enqueue.

### 5.2 Scheduler (control plane) — *the core of the project*

**Responsibility:** turn a stream of heterogeneous requests into full batches on the *right* worker — least-loaded among capable, and cache-local when prefixes are shared — while respecting latency budgets and limited worker memory. This is the component a reviewer will read first; it must be the cleanest code in the repo.

- **Stack:** Python 3.12 + `asyncio` (baseline). Optionally one hot sub-component re-implemented in Rust/Go (ADR-8) — lower priority than the routing depth.
- **Sub-systems:**
  - **Admission control / backpressure:** if total queue depth exceeds a high-water mark, the Gateway is told to shed (`429`) or the request is rejected before enqueue. Unbounded queues are how systems die; bound them explicitly.
  - **Batch former:** per-model deadline loop (see §8 for the algorithm), with priority folded in as a tighter latency budget.
  - **Prefix router / placement (the depth):** hash the request prefix; steer same-prefix requests to the same worker via **bounded-load consistent hashing**; spill over (or replicate a hot prefix) when the target worker is saturated. Policy: maximize cache locality subject to a per-worker load cap. See ADR-12 and §8.
  - **Dispatch:** push `BatchAssignment` over the worker's gRPC lease stream.
  - **Reliability:** at-least-once. If a worker's lease stream drops mid-batch, its in-flight stream entries go unacked and are reclaimed by **`XAUTOCLAIM`** after a lease timeout (reprocessed oldest-first by stream ID); idempotency keys make client-visible duplicates safe.

### 5.3 Worker (data plane)

**Responsibility:** run inference. Pull work, execute, stream results, advertise capacity, heartbeat. Designed to be *dumb and replaceable* — all intelligence is in the scheduler.

- **Stack:** Python 3.12 worker harness wrapping a pluggable **`Engine` interface**:

  ```python
  class Engine(Protocol):
      name: str                       # "mock" | "torch-mps" | "ollama" | "vllm"
      async def load(self, model: str) -> None: ...
      async def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]: ...
      def stats(self) -> EngineStats: ...   # gpu_util, mem, loaded models, cache state
  ```

- **Engine implementations:**
  - **`MockEngine` — the benchmarking backbone.** Returns synthetic results with a *latency model that reproduces real accelerator behavior*: a **fixed per-batch cost** `alpha` (kernel launch, dispatch) plus a **marginal per-item cost** `beta`, with lognormal jitter for a real tail. Because throughput is `b / (alpha + beta·b)` — rising in batch size `b`, asymptoting to `1/beta` — sweeping `MAX_BATCH` against the mock produces the exact textbook latency↔throughput knee, and it is *analytically checkable*. **Calibration:** run a real model (e.g. `qwen2.5:0.5b` on Ollama) a handful of times at `b ∈ {1,4,8,16}`, fit `alpha` and `beta`, and bake those constants in. The mock is then a faithful, disclosed simulation of measured hardware. It also makes scaling sweeps and eviction demoable on an 8GB Air (mock workers cost nothing).
  - **`CacheAwareMockEngine` — for the prefix-routing result.** Holds a per-worker prefix set; returns *low* latency on a prefix hit and *full prefill cost* on a miss. This is what lets the entire prefix-routing algorithm — and its Pareto frontier — be developed and measured locally, with vLLM as later confirmation.
  - `TorchMPSEngine` — PyTorch with `device="mps"`; for vision/embedding/small transformer models on the M2 GPU.
  - `OllamaEngine` — talks to a local Ollama daemon (OpenAI-compatible HTTP); Metal-accelerated LLM inference on Mac. Easiest path to streaming tokens. Note: Ollama **hides the KV cache**, so it cannot be used to *observe or exploit* prefix-cache state — it's for general serving and latency calibration, not the cache-locality measurement.
  - `VLLMEngine` — used **only on a CUDA cloud GPU** (validation phase); gives real continuous batching, automatic prefix caching, and scrapeable per-worker cache-hit counters. Verify current prefix-caching flags/metric names when you get there, since they evolve.

- **vLLM layering (important):** vLLM does its *own* intra-worker continuous batching and per-request streaming. When the engine is vLLM, Relay should **not** form large batches and hand them over as a unit (that fights vLLM's scheduler); dispatch items individually and let vLLM's PagedAttention scheduler merge them. The clean framing: *Relay does inter-worker batching, placement, and fleet autoscaling; vLLM does intra-worker continuous batching — they compose.* Relay's **own** batching value is therefore demonstrated on the Mock/MPS engines, where there is no competing batcher.

- **Capacity model:** worker advertises `max_batch` and `max_concurrent_batches`; the lease loop sends `free_slots = capacity - inflight` so the scheduler never overcommits a slow worker (this is the backpressure mechanism, ADR-2).

### 5.4 State store

**Redis** (coordination, ephemeral) + **Postgres** (durable, queryable). Two stores because they do different jobs — see ADR-5.

- **Redis key layout:**

  | Key | Type | Contents | TTL |
  |---|---|---|---|
  | `queue:{model}:{prio}` | Stream | pending requests (consumer-group acks → at-least-once) | — |
  | `worker:{id}` | Hash | status, models, capacity, last_heartbeat | 15s (refreshed by heartbeat) |
  | `workers:active` | Set | live worker ids | — |
  | `job:{id}` | Hash | status, result, ts_queued, ts_done | 1h |
  | `ratelimit:{key}` | String | token-bucket state (via Lua) | rolling |
  | `idem:{key}` | String | job_id | 24h |

- **Postgres schema (DDL sketch):**

  ```sql
  CREATE TABLE api_keys (
    key_hash    TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    rps_limit   INT  NOT NULL DEFAULT 10,
    created_at  TIMESTAMPTZ DEFAULT now()
  );

  CREATE TABLE requests (
    id            UUID PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    model         TEXT NOT NULL,
    params        JSONB NOT NULL,
    status        TEXT NOT NULL,          -- queued|running|done|error
    batch_id      UUID,
    worker_id     TEXT,
    prefix_hash   TEXT,                   -- for cache-locality analysis
    cache_hit     BOOLEAN,                -- did the worker reuse a prefix?
    queue_wait_ms INT,
    inference_ms  INT,
    total_ms      INT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    completed_at  TIMESTAMPTZ
  );
  CREATE INDEX ON requests (model, created_at);
  CREATE INDEX ON requests (status);
  CREATE INDEX ON requests (prefix_hash);

  CREATE TABLE batches (
    id          UUID PRIMARY KEY,
    model       TEXT NOT NULL,
    size        INT  NOT NULL,
    worker_id   TEXT NOT NULL,
    inference_ms INT,
    created_at  TIMESTAMPTZ DEFAULT now()
  );
  ```

  Postgres is the source of truth for *analytics* ("p99 by model over the last hour", "average batch size", "cache-hit rate by routing policy") — exactly the queries that feed the benchmark report and the Pareto frontier.

### 5.5 Observability

Metrics from commit #1, not bolted on later (ADR-10). Framing: **RED** (Rate/Errors/Duration) for the gateway, **USE** (Utilization/Saturation/Errors) for workers.

- **Stack:** `prometheus-client` exposition on every service, Prometheus scraping, Grafana dashboards committed as JSON in `dashboards/`, structured logging via `structlog` → JSON, optional OpenTelemetry traces (gateway → scheduler → worker) via OTLP.
- **Core metrics:**

  | Metric | Type | Labels | Why it matters |
  |---|---|---|---|
  | `relay_request_total` | counter | model, status | request rate + error rate (RED) |
  | `relay_request_duration_seconds` | histogram | model, mode | end-to-end latency → p50/p95/p99 |
  | `relay_queue_wait_seconds` | histogram | model | how long batching makes you wait |
  | `relay_queue_depth` | gauge | model, priority | **drives autoscaling (HPA)** |
  | `relay_batch_size` | histogram | model | are we actually filling batches? |
  | `relay_prefix_cache_hit_ratio` | gauge | worker_id, policy | **the depth-feature headline** — locality achieved |
  | `relay_worker_load_imbalance` | gauge | — | spread across workers; the other axis of the frontier |
  | `relay_worker_inflight` | gauge | worker_id | saturation (USE) |
  | `relay_worker_gpu_util` | gauge | worker_id | utilization (USE) — *bonus*, credible only on CUDA |
  | `relay_tokens_per_second` | gauge | worker_id, model | LLM throughput |

---

## 6. Architecture Decision Records (ADRs)

The reason this document exists. Each: context → decision → rationale → alternatives → tradeoff.

**ADR-1 — Custom thin scheduler, not Celery/RQ/Ray Serve.**
*Decision:* hand-build the control plane. *Rationale:* the scheduler (batching, placement, backpressure, prefix routing) *is* the thing being demonstrated; a task queue hides exactly the logic worth showing. *Alternatives:* Celery/RQ (hide batching), Ray Serve / KServe (hide everything — you'd be a config author, not an engineer). *Tradeoff:* you re-implement some reliability primitives (retries, acks); acceptable because Redis Streams give you consumer-group acks for free.

**ADR-2 — Pull-based leasing, not push assignment.**
*Decision:* workers open a lease stream and pull batches sized to their free slots. *Rationale:* natural backpressure (a slow/overloaded worker simply stops asking), trivial failure handling (dead stream → reclaim), no central overcommit. *Alternative:* scheduler pushes to workers — simpler to start, but you must track per-worker load perfectly or you overload the slow ones. *Tradeoff:* a poll-interval latency floor; mitigated by a *streaming* lease (gRPC bidi) so assignments arrive the instant capacity frees.

**ADR-3 — Dynamic batching with `(MAX_BATCH, MAX_DELAY_MS)` window.**
*Decision:* form a batch when the queue hits `MAX_BATCH` or the oldest item ages past its latency budget. *Rationale:* this is the single highest-leverage throughput lever and the central latency↔throughput tradeoff in all inference serving. *Alternative:* fixed-size batching (bad tail latency at low load), no batching (terrible utilization). *Tradeoff:* the two knobs are *the* thing to benchmark — there's no universal right value, which is precisely why exposing and measuring them is the point. (Demonstrated on Mock/MPS engines; with vLLM, intra-worker batching is vLLM's job — see ADR-7.)

**ADR-4 — gRPC internally, REST + gRPC at the edge.**
*Decision:* scheduler↔worker over gRPC streaming; clients get both REST and gRPC. *Rationale:* protobuf schema discipline, bidi streaming for lease + result push, low overhead; REST at the edge for accessibility/curl-ability. *Alternative:* REST everywhere (no clean streaming, more overhead), message bus for worker comms (heavier). *Tradeoff:* a proto build step and more moving parts — worth it; "knows more than REST" is a real backend signal.

**ADR-5 — Redis (Streams + structures) for coordination; Postgres for durability.**
*Decision:* two stores. *Rationale:* Redis gives sub-ms queues, consumer-group acks, TTL'd worker registry; Postgres gives durable, *queryable* history for the benchmark report and frontier. *Alternative:* Kafka/RabbitMQ (more durable/throughput than needed, heavier on an 8GB laptop), one store for both (each is wrong at the other's job). *Tradeoff:* Redis Streams are less durable than Kafka — fine at this scale; the README notes the threshold where you'd switch.

**ADR-6 — At-least-once + idempotency, not exactly-once.**
*Decision:* accept at-least-once delivery; make it safe with idempotency keys + a dedup cache. *Mechanism:* in-flight stream entries are claimed via a consumer group; on worker death, entries unacked past a lease timeout are reclaimed with **`XAUTOCLAIM`** (oldest-first by stream ID) and reprocessed. *Rationale:* exactly-once across a network is effectively a myth and very expensive; the pragmatic industry answer is idempotency. *Tradeoff:* clients that want dedup must send a key — documented in the API contract.

**ADR-7 — Pluggable `Engine` interface (mock / mps / ollama / vllm).**
*Decision:* one interface, four backends, with the calibrated mock as the benchmarking backbone. *Rationale:* lets the entire control plane develop and run on the M2, lets the *core algorithmic result be produced locally and reproducibly* (mock), and lets it validate unchanged on a CUDA cloud GPU (vLLM) — develop-local, validate-on-target. *Tradeoff:* the interface must abstract over streaming vs non-streaming engines (handled by always yielding `ResultItem`s) and over engines that do their own batching (vLLM): when the engine batches internally, Relay dispatches items individually and lets the engine merge them.

**ADR-8 — One hot path in Rust/Go alongside the Python baseline (optional, lower priority).**
*Decision:* *after* the routing depth is done and measured, optionally re-implement the dispatch loop or gateway proxy in Rust/Go and benchmark the delta. *Rationale:* demonstrates performance engineering and polyglot range. *Honest caveat:* in an LLM serving system the model forward pass dominates by orders of magnitude, so the dispatch loop is rarely the bottleneck. A "scheduler-induced p99 45ms → 12ms" claim is only honest in a narrow regime — many tiny, fast requests where the Python event loop actually saturates — and is **defensible only if** the benchmark is deliberately constructed to isolate that regime (constant fast Mock engine, swept offered RPS, with the explicit statement that the win is pointless for LLM workloads). *Tradeoff:* extra build/CI complexity for a narrower, more arguable number than the prefix-routing frontier. Do this only if time remains.

**ADR-9 — Kubernetes-native with custom-metric autoscaling.**
*Decision:* deploy to k3d; HPA scales workers on `relay_queue_depth` (custom metric via prometheus-adapter), not CPU. *Rationale:* "autoscale stateless GPU workers on a business metric" is the K8s skill these JDs actually want. *Alternative:* docker-compose only (simpler, but misses the headline skill), CPU-based HPA (meaningless for GPU work). *Tradeoff:* prometheus-adapter wiring is fiddly; that fiddliness is itself the learning.

**ADR-10 — Observability is a Phase-0 requirement, not a later phase.**
*Decision:* every service exposes `/metrics` from its first commit. *Rationale:* the entire résumé value is *measurement*; you cannot retrofit a credible benchmark onto a system that wasn't instrumented from the start. *Tradeoff:* slightly slower start; pays for itself the first time you need a number.

**ADR-11 — A calibrated `MockEngine` is the benchmarking backbone.**
*Decision:* build a mock whose latency follows `alpha + beta·b` (fixed per-batch + marginal per-item), calibrated to real hardware, and run the core sweeps against it. *Rationale:* (1) it fits an 8GB Air where the full real-model stack does not; (2) it isolates the control plane from model variance, so batching/scaling/routing sweeps measure *the scheduler*, not noise; (3) the throughput curve `b/(alpha+beta·b)` is analytically checkable, which is a stronger rigor signal than any single real-model run; (4) it decouples "prove the algorithm" from "prove it on silicon," so the headline is never hostage to a rented GPU window. *Alternative:* benchmark only on real models (expensive, noisy, RAM-bound, non-reproducible). *Tradeoff:* you must disclose the calibration and confirm on real hardware — which is exactly the honest, senior framing.

**ADR-12 — Prefix / KV-cache-aware routing is the single lead depth feature.**
*Decision:* the one deep extension is prefix-aware routing with bounded-load balancing, not a grab-bag of Phase-5 options. *Rationale:* it is the only feature that rebuts the project's central critique ("you orchestrated the hard part") because it depends on engine-internal state (the KV cache). It is a current frontier topic (vLLM automatic prefix caching, SGLang RadixAttention), so reviewers recognize it as real systems work. The *genuinely hard part is the tradeoff*: pure affinity maximizes cache locality but overloads a hot worker; pure load-balancing scatters and destroys locality. Bounded-load consistent hashing (with hot-prefix spillover/replication) navigates that tension, and the **Pareto frontier** of locality vs imbalance vs p99 is the artifact that proves understanding. *Alternative:* weighted-fair scheduling (WFQ/DRF) across tenants — fully local, but touches the inference substrate *zero* (DRF over tenants is the same whether scheduling GPUs or print jobs), so it gives up the property that makes the pick valuable. *Tradeoff:* the real-hardware confirmation needs ≥2 vLLM workers (cloud spend); mitigated by ADR-11's `CacheAwareMockEngine`, which produces the full algorithmic result locally, with vLLM as confirmation rather than sole source.

---

## 7. API & wire contracts

### 7.1 REST request (illustrative)

```http
POST /v1/infer
Authorization: Bearer rl_live_xxx
Idempotency-Key: 8f3a...            # optional
Content-Type: application/json

{ "model": "qwen2.5:0.5b",
  "input": "Summarize: ...",
  "params": { "max_tokens": 128, "temperature": 0.7, "stream": true },
  "mode": "async",
  "priority": "default" }            # default | high
```
→ `202 { "job_id": "f1e2...", "status": "queued" }`

### 7.2 Internal gRPC (`proto/relay/v1/worker.proto`)

```proto
syntax = "proto3";
package relay.v1;

service WorkerGateway {
  rpc Lease(stream LeaseRequest) returns (stream BatchAssignment); // bidi
  rpc ReportResults(stream ResultItem) returns (Ack);
  rpc Register(WorkerInfo) returns (Ack);
}

message WorkerInfo {
  string worker_id = 1; repeated string models = 2;
  int32 max_batch = 3; int32 max_concurrent_batches = 4; string engine = 5;
}
message LeaseRequest { string worker_id = 1; int32 free_slots = 2; repeated string loaded_models = 3; }
message BatchAssignment { string batch_id = 1; string model = 2; repeated InferItem items = 3; }
message InferItem { string request_id = 1; string input = 2; InferParams params = 3; string prefix_hash = 4; }
message InferParams { int32 max_tokens = 1; float temperature = 2; float top_p = 3; bool stream = 4; }
message ResultItem {
  string request_id = 1; string batch_id = 2;
  oneof payload { string output = 3; string token = 4; string error = 5; }
  bool final = 6; WorkerMetrics metrics = 7;
}
message WorkerMetrics { float queue_wait_ms = 1; float inference_ms = 2; int32 batch_size = 3; float gpu_util = 4; bool cache_hit = 5; }
message Ack { bool ok = 1; string detail = 2; }
```

---

## 8. The batching, routing & scheduling algorithm

The heart of the system. Keep it readable; comment the *why*.

### 8.1 Deadline-based batch former (priority folded in)

Priority is *not* a second mechanism — it is a tighter latency budget. The batch former picks across priority queues by deadline, fills high-priority first, but is *triggered* by whichever item across all queues has blown its budget. This gives a provable bound: a `default` item is dispatched within roughly its budget plus one batch service time, so there is **no starvation**.

```python
BUDGET_MS = {"high": 5, "default": 50}   # priority == latency budget

async def batch_former(model: str):
    qs = {p: queues[(model, p)] for p in ("high", "default")}
    while running:
        now = now_ms()
        heads = [(p, q.peek()) for p, q in qs.items() if q.size() > 0]
        if not heads:
            await wait_for_arrival_or(TICK_MS); continue

        slack = lambda p, h: (h.enqueue_ts + BUDGET_MS[p]) - now   # <=0 means overdue
        p_urgent, head = min(heads, key=lambda ph: slack(*ph))
        total = sum(q.size() for q in qs.values())

        # Dispatch when a full batch is available OR the most-urgent item is overdue.
        if not (total >= MAX_BATCH or slack(p_urgent, head) <= 0):
            await wait_for_arrival_or(TICK_MS); continue

        # Prefix-aware placement (see 8.2). Returns None under backpressure.
        worker = router.pick(model, head.prefix_hash)
        if worker is None:
            await asyncio.sleep(TICK_MS); continue

        cap = min(MAX_BATCH, worker.free_slots * worker.max_batch)
        batch = await qs["high"].pop(min(cap, qs["high"].size()))     # high first
        if len(batch) < cap:
            batch += await qs["default"].pop(cap - len(batch))         # top up with default
        await dispatch(worker, make_batch(model, batch))
```

### 8.2 Prefix-aware routing with bounded load (the depth)

```python
# Maximize KV-cache reuse: send same-prefix requests to the same worker.
# But cap each worker's load so a hot prefix can't overload one worker.
class PrefixRouter:
    def __init__(self, workers, load_cap_factor=1.25):
        self.ring = BoundedLoadConsistentHashRing(workers)
        self.cap_factor = load_cap_factor          # <- the Pareto knob

    def pick(self, model, prefix_hash):
        avg_load = mean(w.inflight for w in self.capable(model))
        cap = self.cap_factor * max(avg_load, 1)   # bounded-load constraint
        # Walk the ring from the prefix's hash position; take the first
        # capable worker under its load cap. Pure affinity = cap_factor→∞;
        # pure round-robin = cap_factor→1. The interesting policies live between.
        for w in self.ring.walk(prefix_hash):
            if w.has_model(model) and w.inflight < cap and w.free_slots > 0:
                return w
        return None                                # backpressure: all capped
```

The single knob `load_cap_factor` sweeps the **entire policy space**: at `→∞` it is pure cache affinity (best locality, worst balance); at `→1` it is essentially round-robin (best balance, worst locality). The §13 frontier is a sweep over exactly this knob.

### 8.3 Worker harness

```python
await stub.Register(my_info)
async for assignment in stub.Lease(self.lease_requests()):   # bidi stream
    self.inflight += 1
    async for result in self.engine.infer(assignment.items): # mock / MPS / Ollama / vLLM
        await stub.ReportResults(result)
    self.inflight -= 1
# lease_requests() also emits a LeaseRequest every HEARTBEAT_MS, doubling as liveness.
```

**Knobs to expose (and benchmark):** `MAX_BATCH`, per-priority `BUDGET_MS`, `TICK_MS`, worker `max_concurrent_batches`, and — the headline — `load_cap_factor`.

---

## 9. Local development environment (MacBook Air M2)

Everything except the CUDA/vLLM *validation* runs here natively (Apple Silicon, arm64) — including the **entire prefix-routing result**, via the calibrated `CacheAwareMockEngine`.

**Toolchain:**

```bash
# package mgmt
brew install colima docker kubectl k3d helm ollama
brew install asitop                 # nvidia-smi-style GPU metrics for Apple Silicon
pipx install uv                     # fast Python env/deps  (or poetry)
rustup-init                         # only if doing the optional Rust hot path (ADR-8)

# container runtime (no Docker Desktop license needed)
colima start --cpu 4 --memory 8 --disk 40   # tune to your RAM; see note below

# local kubernetes
k3d cluster create relay --servers 1 --agents 2 \
  --port "8080:80@loadbalancer"
```

**Mac worker engines:** use `MockEngine`/`CacheAwareMockEngine` for all sweeps (free, reproducible, RAM-light). Use Ollama (`ollama pull qwen2.5:0.5b` / `llama3.2:1b`) for latency *calibration* and for a "yes it serves real inference" demo. Use `TorchMPSEngine` for vision/embedding models.

**GPU utilization metric on Mac:** no `nvidia-smi`; scrape `asitop`/`powermetrics` into `relay_worker_gpu_util`. This is coarse — treat tokens/s, p99, and prefix-cache hit-rate as the story; util is a bonus.

**RAM is the real constraint, not the GPU.**
- **8GB Air:** be disciplined. Run the control plane plus *mock* workers (which cost almost nothing) for all benchmarking. Don't run k3d + Postgres + Grafana + a real model simultaneously. Treat the full k3d+observability stack as something you spin up to *capture results*, then tear down.
- **16GB+ Air:** the whole stack runs comfortably; you can keep Grafana up while load testing.

**The cloud-GPU boundary (validation only):** to *confirm* the prefix-routing win on a real KV cache (and to produce the vLLM tokens/s number), rent a multi-GPU spot box (RunPod / Lambda / vast.ai) for a focused day or two. Same `Engine` interface, so only the worker image changes. Verify vLLM's current prefix-caching flags and the exact cache-hit metric names when you arrive.

---

## 10. Repository layout

```
relay/
├── README.md                 # arch diagram + Pareto frontier + numbers on first screen
├── DESIGN.md                 # this document
├── Makefile                  # make dev / test / bench / deploy
├── proto/relay/v1/worker.proto
├── services/
│   ├── gateway/              # FastAPI + gRPC edge
│   ├── scheduler/            # control plane (the showpiece)
│   │   ├── batch_former.py
│   │   ├── router.py         # prefix-aware bounded-load routing (the depth)
│   │   ├── dispatch.py
│   │   └── admission.py
│   └── worker/
│       ├── harness.py
│       └── engines/{mock,cache_aware_mock,torch_mps,ollama,vllm}.py
├── proxy-rs/                 # optional: Rust hot path (tokio + tonic)   [ADR-8]
├── deploy/
│   ├── compose/              # docker-compose for laptop dev (8GB-friendly)
│   ├── helm/relay/           # Helm chart
│   └── k8s/                  # HPA + prometheus-adapter rules
├── dashboards/               # Grafana JSON (committed)
├── bench/                    # workload generator + scenarios + analysis notebooks
│   ├── workload.py           # Zipfian-over-shared-prefixes generator (see §13)
│   ├── calibrate.py          # fit alpha/beta from real Ollama runs
│   ├── scenarios/
│   └── results/              # CSVs + plots that feed the README (incl. the frontier)
├── tests/{unit,integration,chaos}/
└── .github/workflows/ci.yml
```

---

## 11. Phased delivery plan

Each phase has an **exit criterion** — don't start the next until it's met. (For the focused one-month execution of the depth feature, see §17.)

| Phase | Scope | Deliverable | Exit criterion | Rough effort |
|---|---|---|---|---|
| **0 — MVP** | FastAPI → one model → result, Dockerized, `/metrics` live | single-container service | `curl` returns a real prediction; one metric scrapes; **you watch the GPU sit at ~20%** | weekend |
| **1 — Make it a system** | Split gateway / scheduler / worker; Redis stream; async submit + poll | 3 processes via compose | a request flows through the queue and back; status transitions persist | ~1 week |
| **2 — The headline** | Dynamic batching, multiple workers, pull-based leasing, heartbeats, gRPC, **calibrated MockEngine** | batching control plane | **measured throughput gain from batching** on the mock (the money number) | ~1–2 weeks |
| **3 — Kubernetes** | Containerize all; Helm chart; deploy to k3d; HPA on queue depth | one-command k3d deploy | workers autoscale up under load and back down | ~1 week |
| **4 — Observability + load** | Grafana dashboards; k6/locust; latency/throughput sweeps | benchmark report | README has results table + plots vs batch window & worker count | ~3–5 days |
| **5 — The depth (lead feature)** | **Prefix / KV-cache-aware routing + bounded-load balancing + Pareto frontier**; confirm on cloud vLLM | the deep extension | a frontier (locality vs imbalance vs p99) over `load_cap_factor`, local on the mock and confirmed on vLLM | ~1 month focused (see §17) |

---

## 12. Testing strategy

- **Unit (pytest):** batch-former edge cases (empty queue, single item ages out, exactly `MAX_BATCH`), priority/deadline arbitration and the no-starvation bound, token-bucket math, router behavior at the `load_cap_factor` extremes (affinity vs round-robin).
- **Integration:** `docker compose up` the full stack in CI, fire requests, assert end-to-end correctness and status transitions.
- **Contract:** `buf lint` / `buf breaking` on the proto so the wire format can't silently drift.
- **Chaos (the senior touch):** kill a worker mid-batch → assert its unacked stream entries are reclaimed via `XAUTOCLAIM` and the client still gets exactly one result (idempotency holds). Network-partition Redis briefly → assert graceful degradation, not corruption.
- **Load (bench/):** k6 or Locust scenarios at increasing concurrency, driven by the §13 workload generator; this *is* the data source for the report and frontier.

---

## 13. Benchmarking & results methodology

This section is what converts the repo into an interview conversation. Almost no new grad does it — which is exactly why it differentiates.

### 13.1 The workload generator (most of the intellectual work)

Prefix routing does **nothing** on uncorrelated random prompts — you'd get a beautiful null result. The benchmark must model real prefix-sharing. The generator (`bench/workload.py`) draws prefixes from a **Zipfian distribution over a pool of shared prefixes** (system prompts, few-shot blocks, RAG documents), because that is what real traffic looks like: a few prefixes are very hot, a long tail are rare. Parameters to expose and report: pool size, Zipf skew `s`, prefix length, and unique-suffix length. Designing and disclosing this skew honestly is the difference between a real result and self-flattery — state it explicitly so a reviewer can judge it.

### 13.2 Experiments to run

1. **Batching sweep (on MockEngine):** fix worker count; vary `MAX_DELAY_MS` ∈ {0, 5, 10, 25, 50} and `MAX_BATCH` ∈ {1, 4, 8, 16}. Plot throughput (req/s) and p99 latency. Show the knee of the latency↔throughput curve, and overlay the analytic `b/(alpha+beta·b)` prediction to show the system tracks it.
2. **Scaling sweep (on MockEngine):** fix batching; vary worker count 1→N. Plot throughput scaling and where it goes sub-linear (and explain why — scheduler, Redis, or model).
3. **The headline — prefix-routing Pareto frontier (on CacheAwareMockEngine):** drive the skewed workload; sweep `load_cap_factor` from ≈1 (round-robin) to ∞ (pure affinity). Plot **prefix-cache hit-rate vs worker load imbalance vs p99**. The frontier — not a single number — is the artifact. Naive round-robin and pure affinity are the two reference endpoints; your bounded-load policy is the interesting middle.
4. **Real-hardware confirmation (on cloud vLLM):** swap in ≥2 vLLM workers, scrape per-worker prefix-cache hit counters, and confirm the simulated locality win holds on a real KV cache. This *validates*, it does not *source*, the headline.

### 13.3 How to report (README, first screen)

> *"Prefix-aware routing under realistic (Zipf s=1.1) prefix sharing raised KV-cache hit-rate from 18% (round-robin) to 71% and cut p99 by 2.4× at a load-imbalance cost of only 1.3×; the full locality↔balance frontier is below. Confirmed on 2× vLLM workers on an A10G."*

Numbers with units and conditions. A plot beats a paragraph; a table beats a plot for skim-readers — include both. Lead with cache-hit-rate and p99; report utilization only as a CUDA-validated bonus.

---

## 14. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 8GB RAM can't run full stack | high (8GB) / low (16GB+) | mock workers for all benchmarking, compose for dev, spin up k3d only to capture results |
| **Null result — prefix routing does nothing** | **high if ignored** | **model realistic prefix sharing (Zipf over a shared pool); designing the skew is the work, not the hashing** |
| **Headline hostage to a rented GPU window** | **high if ignored** | **produce the full algorithmic result locally on `CacheAwareMockEngine`; vLLM only confirms** |
| Can't observe cache state to prove the win | medium | mock exposes hit/miss directly; vLLM exposes scrapeable counters (verify metric names on arrival) |
| vLLM not usable locally (CUDA-first) | high | confine vLLM to the cloud validation phase; develop against the mock |
| Scope creep → nothing finished | high | strict phase exit criteria; affinity-first then balancing (§17), so a compressed week still ships a result |
| prometheus-adapter / custom-metric HPA fiddliness | medium | timebox; fall back to a documented manual-scale demo if Phase 3 stalls |
| Over-engineering the scheduler | medium | the scheduler should be *clear*, not clever; readability is the deliverable |
| Rust hot path optimizes the wrong thing | medium | only attempt with the isolation methodology in ADR-8; it's optional, after the routing depth |

---

## 15. Tech stack summary

| Layer | Choice | One-line rationale |
|---|---|---|
| Gateway | FastAPI + pydantic v2 + uvicorn | fast to write, async-native, Python is the infra lingua franca |
| Internal RPC | gRPC + protobuf (buf) | streaming + schema discipline; signals beyond-REST competence |
| Control plane | Python asyncio | clarity first; the depth is the routing algorithm, not a rewrite |
| Routing | bounded-load consistent hashing on prefix hash | cache locality vs load balance — the measured tradeoff |
| Coordination | Redis (Streams, hashes, sorted sets, Lua) | sub-ms queues, consumer-group acks + `XAUTOCLAIM`, TTL registry, atomic rate limits |
| Durable store | Postgres (JSONB) | queryable history → feeds the report and frontier |
| Inference engine | Mock (benchmarking) · Ollama / PyTorch-MPS (Mac) · vLLM (cloud) | reproducible results local, validate-on-target CUDA |
| Orchestration | Docker + Kubernetes (k3d) + Helm | k8s-native + autoscaling is the headline JD skill |
| Autoscaling | HPA on custom metric (queue depth) via prometheus-adapter | scale on a business metric, not CPU |
| Observability | Prometheus + Grafana + structlog + OTel | RED/USE; measurement is the whole point |
| Load testing | k6 or Locust + Zipfian workload generator | reproducible, *realistic* numbers for the report |
| CI | GitHub Actions (lint, test, buf, build) | basic engineering hygiene, visible on the repo |

---

## 16. The interview narrative this produces

When asked "tell me about a project," the spine of the answer:

> *"I built a model-serving control plane — gateway, scheduler, GPU workers — that decouples ingestion from execution. The interesting part is the scheduler's prefix-aware routing: many LLM requests share a long prefix (system prompt, RAG document), and naive round-robin scatters them so every worker recomputes the prefix and the KV-cache hit rate collapses. I route same-prefix requests to the same worker with bounded-load consistent hashing — so a hot prefix can't overload one worker — and I measured the whole locality-vs-balance tradeoff as a Pareto frontier. Under realistic Zipfian prefix sharing it took cache-hit rate from ~18% to ~71% and cut p99 2.4×. I produced that result locally against a calibrated mock engine so it's reproducible, then confirmed it on two vLLM workers on a rented GPU. It's Kubernetes-native — workers autoscale on queue depth — and the pull-based leasing gives backpressure for free."*

That paragraph hits: distributed systems, scheduling, backpressure, **the actual hard substrate of inference serving (the KV cache)**, a real algorithmic tradeoff measured as a frontier, gRPC, Kubernetes, autoscaling, observability, and rigorous reproducible benchmarking — i.e. the union of what Lablup, Furiosa's platform team, Rebellions' AI-server team, and Moreh actually hire for. (Lablup is the bullseye — Backend.AI is exactly this class of system. The silicon companies' platform teams are relevant but adjacent; their deepest value is compiler/runtime, which a separate chip-runtime project addresses.)

**Résumé bullets (fill in your real numbers):**
- Built a GPU inference-serving control plane (FastAPI/gRPC/Redis/Postgres/K8s) with dynamic batching and **prefix/KV-cache-aware routing**.
- Designed bounded-load consistent-hashing routing balancing cache locality against worker load; measured the tradeoff as a Pareto frontier — **cache-hit ~18%→~71%, p99 −2.4×** under realistic Zipfian prefix sharing.
- Produced all core results reproducibly against a calibrated mock engine; confirmed on 2× vLLM workers on a rented GPU.
- Pull-based worker leasing for backpressure + at-least-once delivery (`XAUTOCLAIM` reclaim) with idempotency; chaos-tested worker failure mid-batch.
- Kubernetes-native with HPA autoscaling workers on a custom queue-depth metric; one-command k3d deploy.

---

## 17. The focused one-month execution plan (the depth feature)

"Add depth" has a precise meaning and a precise failure mode. The entire month goes into **one** hard thing measured ruthlessly; the discipline is resisting breadth. The plan is ordered so that the difficulty is front-loaded where it actually lives (workload realism and measurability), and so that a compressed final week still ships a complete result.

**Where the difficulty actually is.** The routing algorithm (hash prefix → ring → bounded-load fallback) is ~200 lines and takes about two days. The project succeeds or fails on two things: (1) **workload realism** — without modeled prefix sharing you get a null result; (2) **measurability** — without observable cache-hit-rate you can't claim anything. Spend the time there.

### Week 1 — Foundation (all local, all free)
- Build the calibrated `MockEngine` (`alpha + beta·b`), fitting `alpha`/`beta` from real Ollama runs (`bench/calibrate.py`).
- Build the `CacheAwareMockEngine` (per-worker prefix set; hit = low latency, miss = full prefill).
- Build the Zipfian-over-shared-prefixes workload generator (`bench/workload.py`).
- Wire the harness that measures cache-hit-rate and p99 per policy.
- **Exit:** a local affinity-vs-round-robin result on the mock under skewed load — i.e. the measurement pipeline works end to end.

### Week 2 — The router (the core result, still local)
- Implement prefix hashing + a consistent-hash ring + affinity placement; integrate into scheduler placement.
- **Exit:** routing demonstrably lifts cache-hit-rate vs round-robin on the skewed workload. This is the core result and it is already a complete, strong artifact on its own.

### Week 3 — The depth (the stretch)
- Add the bounded-load constraint (hot-prefix spillover / replication) via `load_cap_factor`.
- Sweep `load_cap_factor` from ≈1 to ∞ and produce the **Pareto frontier** (locality vs imbalance vs p99).
- **Exit:** the frontier plot. (If this week compresses, you still have Week 2's locality win and you describe load-balancing as future work — you are never left with nothing.)

### Week 4 — Validate + present
- Rent a multi-GPU box; swap in ≥2 vLLM workers; verify current prefix-caching flags; scrape real per-worker cache-hit counters.
- Confirm the simulated locality win holds on a real KV cache.
- Write the README first screen: architecture diagram + the frontier + the headline numbers.
- **Exit:** README with the artifact, reproducible locally with `make bench`, confirmed on real hardware.

**Order of operations is the safety net:** affinity-first (Week 2), balancing-second (Week 3), validation-last (Week 4). The local result is never hostage to the GPU window.

**Local-only fallback (if you won't spend on GPU, or want lower risk):** weighted-fair scheduling (WFQ or DRF) across tenants — fully local on the mock/Ollama, lower ceiling but still a strong "I understand scheduling as a discipline" signal. The month structure is identical; only Weeks 2–3 swap content. Note the explicit cost: WFQ touches the inference substrate *zero*, so it gives up the property that makes prefix routing the better pick. Prefer prefix routing with the mock-first approach, which keeps it almost entirely local anyway.

**One pre-build factual check worth doing before committing the GPU day:** confirm whether your target vLLM version exposes *per-worker* prefix-cache hit counters you can scrape externally (vs only aggregate). A wrong assumption there costs a paid afternoon.
