"""Virtual-time discrete-event simulator for the routing experiments
(DESIGN.md §13.2).

This is how the headline result is produced *locally, free, and reproducibly*.
It runs the real control-plane logic — the §8.1 deadline batch former and the
§8.2 bounded-load router, imported unchanged — against the CacheAwareMockEngine,
over a **virtual clock**. No sockets, no sleeps: the engine reports how long a
batch *would* take and the clock jumps to the next event, so a 30k-request sweep
finishes in seconds.

------------------------------------------------------------------------------
Where the locality actually comes from (a design decision worth stating)
------------------------------------------------------------------------------
The first cut routed *formed batches* by the prefix of the batch's head item
(literally §8.2 applied to the output of a single global §8.1 former). That
produces a **null result**: the global former fills a batch FIFO across *all*
prefixes, so a batch is prefix-heterogeneous, and whichever worker it lands on
must prefill every distinct prefix in it. Cache-hit rate was then identical
across every policy — routing the mixed batch by its head changes *which* worker
pays, not *whether* prefill is paid. That negative finding is reproducible and is
called out in RESULTS.md.

The faithful realization of the doc's actual goal ("route same-prefix requests
to the same worker so its KV cache is reused", §1/§16) is to route **per request
at admission** into **per-worker queues**, and let each worker run the §8.1
deadline former over its *own* queue. Then a worker's batches are prefix-coherent
(same hot prefix repeated), its cache stays warm, and the locality is real. This
is exactly how production prefix-aware schedulers (SGLang's radix router,
vLLM-router) are organised, so the model matches the systems it is standing in
for. The §8.1/§8.2 code is unchanged; only the *topology* (one queue per worker,
router at admission) differs from the naive cut, and the router is the same
class — see ``PrefixRouter``'s ``load_fn`` / ``admit_fn`` seam.

Load metric for the cap in this topology is ``queued_items + inflight_items`` per
worker (a high-resolution proxy for "how backed up is this worker"), and
admission always succeeds (``admit_fn = lambda w: True``) because requests queue
at the worker rather than being rejected for lack of a free batch slot.

------------------------------------------------------------------------------
Experiment design
------------------------------------------------------------------------------
Open-system model: Poisson arrivals at a fixed offered rate (req/s) so queueing
is real and p99 is meaningful. Every policy run uses the *same* prefix sequence
and the *same* arrival times (fixed seeds) and starts with cold per-worker caches,
so comparisons are apples-to-apples. Steady-state metrics exclude a warmup prefix
of the request stream so the cold-cache transient doesn't pollute the hit-rate.

Two reference policies bracket the bounded-load sweep:
  * ``round_robin`` — least-loaded placement, ignores the prefix (best balance,
    worst locality);
  * ``prefix`` with ``cap_factor = inf`` — pure affinity (best locality, worst
    balance under skew).
The bounded-load sweep over finite ``cap_factor`` is the interesting middle and
is what traces the Pareto frontier.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from bench.workload import WorkloadParams, ZipfianPrefixWorkload
from relay_core.queue import LocalQueue
from relay_core.types import InferItem, Priority, WorkerState
from services.scheduler import batch_former as bf
from services.scheduler.router import PrefixRouter
from services.worker.engines.cache_aware_mock import CacheAwareMockEngine

MODEL = "qwen2.5:0.5b"


# --------------------------------------------------------------------------- #
# Scenario + result
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    n_requests: int = 30_000
    warmup_requests: int = 3_000  # excluded from steady-state metrics
    n_workers: int = 4
    max_concurrent_batches: int = 4
    max_batch: int = 16
    offered_rps: float = 240.0
    # engine (alpha/beta calibrated; prefill is the cache-saved cost; capacity
    # bounds resident prefixes per worker). cache_capacity ≈ pool_size/n_workers
    # is the interesting regime: affinity nearly fits a worker's owned prefix set
    # in cache while round-robin (which sees the whole pool per worker) cannot.
    alpha_ms: float = 18.0
    beta_ms: float = 7.5
    prefill_ms: float = 160.0
    cache_capacity: int = 48
    jitter_sigma: float = 0.15
    # workload — Zipf s=1.1 over a pool of 256 shared prefixes (DESIGN.md §13.1)
    workload: WorkloadParams = field(
        default_factory=lambda: WorkloadParams(pool_size=256, skew=1.1)
    )
    # scheduling budgets (priority == budget)
    budget_high_ms: float = 5.0
    budget_default_ms: float = 50.0
    vnodes: int = 160
    # seeds
    arrival_seed: int = 11
    engine_seed: int = 23


@dataclass
class RunResult:
    policy: str
    cap_factor: float | None
    offered_rps: float
    n_completed: int
    cache_hit_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    load_imbalance: float  # max/mean items processed per worker
    busy_imbalance: float  # max/mean busy time per worker
    mean_queue_wait_ms: float
    mean_batch_size: float
    utilization: float
    per_worker_hit: dict[str, float]
    per_worker_items: dict[str, int]

    def row(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "policy",
                "cap_factor",
                "offered_rps",
                "n_completed",
                "cache_hit_rate",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "throughput_rps",
                "load_imbalance",
                "busy_imbalance",
                "mean_queue_wait_ms",
                "mean_batch_size",
                "utilization",
            )
        }


# --------------------------------------------------------------------------- #
# Round-robin reference policy (balance endpoint).
# Least-loaded placement (load == queued + inflight items), prefix-agnostic,
# with deterministic rotation among ties. This is the strong *balance* baseline
# that destroys locality (DESIGN.md §13.2).
# --------------------------------------------------------------------------- #
class RoundRobinRouter:
    def __init__(
        self,
        workers: list[WorkerState],
        load_fn: Callable[[WorkerState], float],
    ) -> None:
        self.workers = {w.worker_id: w for w in workers}
        self.load_fn = load_fn
        self._cycle = itertools.cycle(sorted(self.workers))

    def pick(self, model: str, prefix_hash: str) -> WorkerState | None:
        capable = [w for w in self.workers.values() if w.has_model(model)]
        if not capable:
            return None
        m = min(self.load_fn(w) for w in capable)
        least = [w for w in capable if self.load_fn(w) == m]
        if len(least) == 1:
            return least[0]
        ids = {w.worker_id for w in least}
        for _ in range(len(self.workers)):
            wid = next(self._cycle)
            if wid in ids:
                return self.workers[wid]
        return least[0]


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
class Simulator:
    def __init__(self, scenario: Scenario, policy: str, cap_factor: float | None):
        self.s = scenario
        self.policy = policy
        self.cap_factor = cap_factor

        # Workers + their private cache-aware engines (cold caches).
        self.workers: list[WorkerState] = [
            WorkerState(
                worker_id=f"w{i}",
                engine="cache-aware-mock",
                models=(MODEL,),
                max_batch=scenario.max_batch,
                max_concurrent_batches=scenario.max_concurrent_batches,
            )
            for i in range(scenario.n_workers)
        ]
        self._wby: dict[str, WorkerState] = {w.worker_id: w for w in self.workers}
        self.engines: dict[str, CacheAwareMockEngine] = {
            w.worker_id: CacheAwareMockEngine(
                alpha_ms=scenario.alpha_ms,
                beta_ms=scenario.beta_ms,
                prefill_ms=scenario.prefill_ms,
                cache_capacity=scenario.cache_capacity,
                jitter_sigma=scenario.jitter_sigma,
                seed=scenario.engine_seed + i,
            )
            for i, w in enumerate(self.workers)
        }

        # One queue-set per worker; routing happens at admission (see docstring).
        self.wq: dict[str, dict[Priority, LocalQueue]] = {
            w.worker_id: {Priority.HIGH: LocalQueue(), Priority.DEFAULT: LocalQueue()}
            for w in self.workers
        }
        self.inflight_items: dict[str, int] = {w.worker_id: 0 for w in self.workers}
        self.budgets = {
            Priority.HIGH: scenario.budget_high_ms,
            Priority.DEFAULT: scenario.budget_default_ms,
        }

        if policy == "round_robin":
            self.router = RoundRobinRouter(self.workers, load_fn=self._load)
        elif policy == "prefix":
            assert cap_factor is not None
            self.router = PrefixRouter(
                self.workers,
                load_cap_factor=cap_factor,
                vnodes=scenario.vnodes,
                load_fn=self._load,
                admit_fn=lambda w: True,
            )
        else:
            raise ValueError(policy)

        # Workload (content) + arrivals (timing), both from fixed seeds.
        wl = ZipfianPrefixWorkload(scenario.workload)
        self.workload = wl
        self.items: list[InferItem] = wl.generate(scenario.n_requests)
        rng = np.random.default_rng(scenario.arrival_seed)
        gaps = rng.exponential(1000.0 / scenario.offered_rps, size=scenario.n_requests)
        self.arrivals = np.cumsum(gaps)

        # Event heap: (time_ms, seq, kind, payload)
        self._heap: list[tuple[float, int, str, object]] = []
        self._seq = itertools.count()
        self.clock = 0.0
        self._batch_ids = itertools.count()
        self._pending_dl: set[tuple[str, float]] = set()

        # Bookkeeping (per request, indexed like self.items)
        n = scenario.n_requests
        self.completion_ms = np.full(n, np.nan)
        self.latency_ms = np.full(n, np.nan)
        self.queue_wait_ms = np.full(n, np.nan)
        self.cache_hit = np.zeros(n, dtype=bool)
        self.item_worker = np.full(n, -1, dtype=np.int64)
        self._idx_of: dict[str, int] = {it.request_id: i for i, it in enumerate(self.items)}
        self._wid_index = {w.worker_id: i for i, w in enumerate(self.workers)}
        # Per-worker busy time (full run; secondary metric)
        self.busy_ms = {w.worker_id: 0.0 for w in self.workers}
        self.batch_sizes: list[int] = []

    # -- load metric the router caps on (queued + inflight items) ---------- #
    def _load(self, w: WorkerState) -> float:
        wid = w.worker_id
        q = self.wq[wid]
        return float(
            q[Priority.HIGH].size() + q[Priority.DEFAULT].size() + self.inflight_items[wid]
        )

    # -- event helpers ----------------------------------------------------- #
    def _push(self, t: float, kind: str, payload: object) -> None:
        heapq.heappush(self._heap, (t, next(self._seq), kind, payload))

    def _push_deadline(self, wid: str, dl: float) -> None:
        key = (wid, dl)
        if dl > self.clock and key not in self._pending_dl:
            self._pending_dl.add(key)
            self._push(dl, "deadline", wid)

    def _start_batch(self, w: WorkerState, items: list[InferItem]) -> None:
        engine = self.engines[w.worker_id]
        latency, per_item_hit = engine.run_batch(items)
        w.inflight += 1
        self.inflight_items[w.worker_id] += len(items)
        self.batch_sizes.append(len(items))
        self.busy_ms[w.worker_id] += latency
        widx = self._wid_index[w.worker_id]
        for it, hit in zip(items, per_item_hit, strict=False):
            idx = self._idx_of[it.request_id]
            self.cache_hit[idx] = hit
            self.queue_wait_ms[idx] = self.clock - it.enqueue_ts
            self.item_worker[idx] = widx
        self._push(
            self.clock + latency, "done", (w.worker_id, len(items), [it.request_id for it in items])
        )

    def _try_form(self, wid: str) -> None:
        """Run the §8.1 former over *this worker's* queues, starting as many
        batches as it has free concurrent slots for."""
        w = self._wby[wid]
        queues = self.wq[wid]
        while w.free_slots > 0:
            heads = [
                bf.Head(p, q.peek())  # type: ignore[arg-type]
                for p, q in queues.items()
                if q.size() > 0 and q.peek() is not None
            ]
            urgent = bf.most_urgent(heads, self.clock, self.budgets)
            total = sum(q.size() for q in queues.values())
            if not bf.should_dispatch(total, urgent, self.clock, self.s.max_batch, self.budgets):
                if urgent is not None:
                    dl = urgent.item.enqueue_ts + self.budgets[urgent.priority]
                    self._push_deadline(wid, dl)
                return
            assert urgent is not None
            cap = bf.worker_batch_cap(w, self.s.max_batch)  # one batch's worth
            items = bf.assemble_batch(queues, cap)
            if not items:
                return
            self._start_batch(w, items)

    # -- main loop --------------------------------------------------------- #
    def run(self) -> RunResult:
        for i in range(self.s.n_requests):
            self._push(float(self.arrivals[i]), "arrival", i)

        while self._heap:
            t, _, kind, payload = heapq.heappop(self._heap)
            self.clock = t
            if kind == "arrival":
                i = payload  # type: ignore[assignment]
                it = self.items[i]
                it.enqueue_ts = self.clock
                w = self.router.pick(MODEL, it.prefix_hash)
                if w is None:  # no capable worker under cap; retry shortly
                    self._push(self.clock + 1.0, "arrival", i)
                    continue
                self.wq[w.worker_id][it.priority].push(it)
                self._try_form(w.worker_id)
            elif kind == "done":
                wid, nitems, req_ids = payload  # type: ignore[assignment]
                self._wby[wid].inflight -= 1
                self.inflight_items[wid] -= nitems
                for rid in req_ids:
                    idx = self._idx_of[rid]
                    self.completion_ms[idx] = self.clock
                    self.latency_ms[idx] = self.clock - self.items[idx].enqueue_ts
                self._try_form(wid)
            elif kind == "deadline":
                wid = payload  # type: ignore[assignment]
                self._pending_dl.discard((wid, t))
                self._try_form(wid)

        return self._summarize()

    # -- metrics ----------------------------------------------------------- #
    def _summarize(self) -> RunResult:
        n = self.s.n_requests
        idx = np.arange(n)
        completed = ~np.isnan(self.latency_ms)
        steady = completed & (idx >= self.s.warmup_requests)
        n_steady = int(steady.sum())

        lat = self.latency_ms[steady]
        # Throughput over the steady-state window (completions / wall span).
        if n_steady:
            comp = self.completion_ms[steady]
            enq = np.array([self.items[i].enqueue_ts for i in idx[steady]])
            span_s = max((comp.max() - enq.min()) / 1000.0, 1e-9)
        else:
            span_s = 1e-9

        # Per-worker steady-state item counts + hit rates.
        per_worker_items: dict[str, int] = {}
        per_worker_hit: dict[str, float] = {}
        for w in self.workers:
            wi = self._wid_index[w.worker_id]
            mask = steady & (self.item_worker == wi)
            cnt = int(mask.sum())
            per_worker_items[w.worker_id] = cnt
            per_worker_hit[w.worker_id] = float(self.cache_hit[mask].mean()) if cnt else 0.0

        items_arr = np.array(list(per_worker_items.values()), dtype=np.float64)
        mean_items = items_arr.mean() if items_arr.size and items_arr.mean() else 1.0
        busy_arr = np.array(list(self.busy_ms.values()), dtype=np.float64)
        mean_busy = busy_arr.mean() if busy_arr.size and busy_arr.mean() else 1.0

        # Utilization uses full-run busy time over full-run span (secondary).
        full_done = ~np.isnan(self.completion_ms)
        if full_done.any():
            full_span_s = max(
                (float(np.nanmax(self.completion_ms)) - float(self.arrivals[0])) / 1000.0,
                1e-9,
            )
        else:
            full_span_s = 1e-9
        util = float(
            busy_arr.sum()
            / (self.s.n_workers * self.s.max_concurrent_batches * full_span_s * 1000.0)
        )

        return RunResult(
            policy=self.policy,
            cap_factor=self.cap_factor,
            offered_rps=self.s.offered_rps,
            n_completed=n_steady,
            cache_hit_rate=float(self.cache_hit[steady].mean()) if n_steady else 0.0,
            p50_ms=float(np.percentile(lat, 50)) if n_steady else float("nan"),
            p95_ms=float(np.percentile(lat, 95)) if n_steady else float("nan"),
            p99_ms=float(np.percentile(lat, 99)) if n_steady else float("nan"),
            throughput_rps=n_steady / span_s,
            load_imbalance=float(items_arr.max() / mean_items) if items_arr.size else 1.0,
            busy_imbalance=float(busy_arr.max() / mean_busy) if busy_arr.size else 1.0,
            mean_queue_wait_ms=float(np.nanmean(self.queue_wait_ms[steady]))
            if n_steady
            else float("nan"),
            mean_batch_size=float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
            utilization=util,
            per_worker_hit=per_worker_hit,
            per_worker_items=per_worker_items,
        )


def run_one(scenario: Scenario, policy: str, cap_factor: float | None) -> RunResult:
    return Simulator(scenario, policy, cap_factor).run()
