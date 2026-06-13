"""Does longest-prefix (radix) routing actually beat single-block-hash routing?
(DESIGN.md §8.2 radix follow-up — the finding that justifies the data structure.)

The original `PrefixRouter` keys on a hash of the *first* block. That is fine when
each distinct prefix is its own island, but it has a sharp failure mode in the
regime prefix-caching exists for: a **broadly shared stem**. If one system prompt
(or RAG preamble) is block 0 of *almost all* traffic, single-first-block-hash maps
all of it to one ring position → one worker. Bounded load then forces the overflow
onto *cold* workers that must re-prefill from scratch, so the shared cache barely
helps and balance is poor.

Longest-prefix routing distributes that same traffic by its *deeper* shared
prefix: requests are co-located by the longest stem they share, so each worker
owns a subtree of the prefix space, keeps that subtree's blocks warm, and load
spreads across the branches instead of piling on one owner.

This harness drives the **real routers** (`PrefixRouter`, `RadixPrefixRouter`)
over a branching block-workload, with an inline per-worker block-level LRU cache
(prefill charged per *missing block* — partial reuse, as real engines do). It is
self-contained (it touches none of the simulator core) and reports, per policy:

  * block-reuse fraction  — (cached blocks) / (total blocks)  [the cache win]
  * load imbalance        — max worker load / mean              [the balance cost]
  * p99 service time      — under a block-level prefill cost model
  * distinct workers per stem — placement spread

    PYTHONPATH=. python bench/radix_compare.py
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
from collections import OrderedDict

import numpy as np

from relay_core.types import WorkerState
from services.scheduler.router import PURE_AFFINITY, PrefixRouter
from services.scheduler.radix_router import RadixPrefixRouter

MODEL = "m"
PREFILL_MS_PER_BLOCK = 40.0   # cost to build one block's KV on a miss
ALPHA_MS = 18.0
BETA_MS = 7.5                  # decode, per item in batch (single-item here)


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def branching_workload(n: int, n_branches: int, depth: int, leaf_fanout: int,
                       seed: int) -> list[tuple[str, ...]]:
    """Block sequences with ONE shared root block (block 0), branching into
    `n_branches` sub-stems, each extended to `depth` shared blocks, then a
    per-request divergent tail. So *all* traffic shares block 0; subsets share
    progressively deeper stems — the regime single-first-block-hash mishandles."""
    rng = np.random.default_rng(seed)
    root = _h("system-prompt-v1")               # shared by EVERY request
    # Zipfian popularity over branches (some stems much hotter than others).
    ranks = np.arange(1, n_branches + 1, dtype=np.float64)
    pmf = 1.0 / ranks ** 1.1
    pmf /= pmf.sum()
    seqs: list[tuple[str, ...]] = []
    for i in range(n):
        br = int(rng.choice(n_branches, p=pmf))
        blocks = [root] + [_h(f"branch{br}-d{d}") for d in range(depth)]
        # divergent tail block (unique-ish per request, shared within a small leaf)
        leaf = int(rng.integers(0, leaf_fanout))
        blocks.append(_h(f"branch{br}-leaf{leaf}-{i % leaf_fanout}"))
        seqs.append(tuple(blocks))
    return seqs


class BlockCache:
    """Per-worker LRU over *blocks*; reports how many of an item's blocks are
    resident (reuse depth) and inserts all of them (warming the worker)."""

    def __init__(self, capacity_blocks: int):
        self.cap = capacity_blocks
        self.lru: "OrderedDict[str, None]" = OrderedDict()

    def reused_and_warm(self, blocks: tuple[str, ...]) -> int:
        reused = 0
        # leading resident blocks count as reuse (prefix property)
        for b in blocks:
            if b in self.lru:
                reused += 1
                self.lru.move_to_end(b)
            else:
                break
        for b in blocks:                          # warm all blocks
            if b in self.lru:
                self.lru.move_to_end(b)
            else:
                self.lru[b] = None
                while len(self.lru) > self.cap:
                    self.lru.popitem(last=False)
        return reused


def run(policy: str, seqs, n_workers, cap_factor, offered_rps, cache_blocks,
        max_concurrent: int = 4):
    workers = [WorkerState(worker_id=f"w{i}", engine="mock", models=(MODEL,),
                           max_concurrent_batches=max_concurrent)
               for i in range(n_workers)]
    # Live load the cap reacts to = in-service + queued (mirrors simulate.py's
    # _load), so a worker with a deep queue is seen as loaded and spilled off.
    busy = {w.worker_id: 0 for w in workers}              # slots in service
    queued = {w.worker_id: 0 for w in workers}            # waiting in FIFO
    load_fn = lambda w: float(busy[w.worker_id] + queued[w.worker_id])

    if policy == "single_hash":
        router = PrefixRouter(workers, load_cap_factor=cap_factor,
                              load_fn=load_fn, admit_fn=lambda w: True)
        pick = lambda blocks: router.pick(MODEL, blocks[0])      # first block only
    elif policy == "radix":
        router = RadixPrefixRouter(workers, load_cap_factor=cap_factor,
                                   load_fn=load_fn, admit_fn=lambda w: True)
        pick = lambda blocks: router.pick(MODEL, blocks)
    else:
        raise ValueError(policy)

    caches = {w.worker_id: BlockCache(cache_blocks) for w in workers}
    fifo: dict[str, list[tuple[float, tuple]]] = {w.worker_id: [] for w in workers}
    rng = np.random.default_rng(7)
    gaps = rng.exponential(1000.0 / offered_rps, size=len(seqs))
    arrivals = np.cumsum(gaps)

    heap: list[tuple[float, int, str, object]] = []
    seq = itertools.count()
    for i, t in enumerate(arrivals):
        heapq.heappush(heap, (float(t), next(seq), "arr", i))

    clock = 0.0
    latency_ms: list[float] = []      # END-TO-END: queue wait + service
    reuse_frac: list[float] = []
    assign_count = {w.worker_id: 0 for w in workers}
    stem_workers: dict[tuple, set] = {}

    def _start(wid: str, arrival_ts: float, blocks: tuple) -> None:
        # Cache state is read at service *start* (it changed while queued).
        reused = caches[wid].reused_and_warm(blocks)
        missed = len(blocks) - reused
        svc = ALPHA_MS + BETA_MS + PREFILL_MS_PER_BLOCK * missed
        busy[wid] += 1
        reuse_frac.append(reused / len(blocks))
        latency_ms.append((clock - arrival_ts) + svc)     # wait + service
        heapq.heappush(heap, (clock + svc, next(seq), "done", wid))

    while heap:
        t, _, kind, payload = heapq.heappop(heap)
        clock = t
        if kind == "arr":
            i = payload
            blocks = seqs[i]
            w = pick(blocks)
            if w is None:                          # all capped → retry shortly
                heapq.heappush(heap, (clock + 1.0, next(seq), "arr", i))
                continue
            wid = w.worker_id
            assign_count[wid] += 1
            stem_workers.setdefault(blocks[:3], set()).add(wid)
            if busy[wid] < max_concurrent:         # free slot → serve now
                _start(wid, clock, blocks)
            else:                                  # busy → wait in this worker's queue
                fifo[wid].append((clock, blocks))
                queued[wid] += 1
        else:                                      # done: free slot, pull next queued
            wid = payload
            busy[wid] -= 1
            if fifo[wid]:
                arr_ts, blks = fifo[wid].pop(0)
                queued[wid] -= 1
                _start(wid, arr_ts, blks)

    counts = np.array(list(assign_count.values()), float)
    imbalance = float(counts.max() / counts.mean()) if counts.mean() else 0.0
    distinct = float(np.mean([len(s) for s in stem_workers.values()]))
    return {
        "policy": policy,
        "reuse_frac": round(float(np.mean(reuse_frac)), 3),
        "imbalance": round(imbalance, 3),
        "p99_ms": round(float(np.percentile(latency_ms, 99)), 1),
        "p50_ms": round(float(np.percentile(latency_ms, 50)), 1),
        "distinct_workers_per_stem": round(distinct, 2),
        "per_worker": dict(assign_count),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="single-block-hash vs radix routing.")
    ap.add_argument("--n-requests", type=int, default=6000)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--branches", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--leaf-fanout", type=int, default=6)
    ap.add_argument("--cap-factor", type=float, default=1.5)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cache-blocks", type=int, default=64)
    args = ap.parse_args()

    seqs = branching_workload(args.n_requests, args.branches, args.depth,
                              args.leaf_fanout, seed=1)
    avg_blocks = float(np.mean([len(s) for s in seqs]))
    print(f"[radix] branching workload: {args.n_requests} reqs, {args.branches} "
          f"branches × depth {args.depth}, ~{avg_blocks:.0f} blocks/req, ALL sharing "
          f"block 0 (a common system prompt).\n        {args.n_workers} workers × "
          f"{args.max_concurrent} concurrent, cap={args.cap_factor}.\n")

    # Routing quality is load-independent — report it once at light load.
    base = {p: run(p, seqs, args.n_workers, args.cap_factor, 40.0,
                   args.cache_blocks, args.max_concurrent)
            for p in ("single_hash", "radix")}
    sh0, rx0 = base["single_hash"], base["radix"]
    print("  routing quality (light load):")
    print(f"    single_hash  block-reuse={sh0['reuse_frac']:.3f}  "
          f"imbalance={sh0['imbalance']:.2f}  distinct_workers/stem={sh0['distinct_workers_per_stem']:.2f}")
    print(f"    radix        block-reuse={rx0['reuse_frac']:.3f}  "
          f"imbalance={rx0['imbalance']:.2f}  distinct_workers/stem={rx0['distinct_workers_per_stem']:.2f}")
    print(f"    → radix reuses +{100*(rx0['reuse_frac']-sh0['reuse_frac']):.0f}% more "
          f"cache and balances {sh0['imbalance']/rx0['imbalance']:.1f}× better.\n")

    # The finding: as offered load rises, single-hash's imbalance saturates its
    # hottest worker first → p99 diverges. Sweep to find where.
    print("  p99 latency (ms) vs offered load — where does imbalance bite?")
    print(f"    {'rps':>6}  {'single_hash':>12}  {'radix':>10}   ratio")
    for rps in (40, 70, 100, 130, 160, 200):
        sh = run("single_hash", seqs, args.n_workers, args.cap_factor, rps,
                 args.cache_blocks, args.max_concurrent)
        rx = run("radix", seqs, args.n_workers, args.cap_factor, rps,
                 args.cache_blocks, args.max_concurrent)
        ratio = sh["p99_ms"] / rx["p99_ms"] if rx["p99_ms"] else 0.0
        flag = "  ← single-hash hot worker saturating" if ratio >= 1.5 else ""
        print(f"    {rps:>6}  {sh['p99_ms']:>12.0f}  {rx['p99_ms']:>10.0f}   "
              f"{ratio:>4.1f}×{flag}")
    print("\n  Read: longest-prefix routing distributes the shared-stem traffic by "
          "deeper\n  prefix, so no single worker is pinned — it sustains higher load "
          "before p99\n  degrades, while also reusing more cache. (Service-time cost "
          "model; latency is\n  queue-wait + block-prefill. The crossover in *load*, "
          "not prefix length, is the\n  point here.)")


if __name__ == "__main__":
    main()