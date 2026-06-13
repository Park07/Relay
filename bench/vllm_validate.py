"""Real-hardware validation: does Relay's prefix router actually reuse KV cache
on real vLLM workers, and does the prefill-length advantage cross over as the
simulator predicts? (DESIGN.md §13.2, experiment 4 — the step that turns the
simulated crossover into a measured one.)

This driver runs on a box with >=2 GPUs each hosting a real vLLM server with
``--enable-prefix-caching --enable-prompt-tokens-details``. It imports Relay's
*actual* router (``services/scheduler/router.py``) and uses it to place each
request across the workers, then sends the request directly to the chosen
worker. So the placement decision is Relay's own code; the KV cache, eviction,
and batching are real. vLLM reports ``usage.prompt_tokens_details.cached_tokens``
per response — the ground-truth signal for "was this prefix already cached on
the worker it landed on".

What it measures, per (shared-prefix length, policy):
  * cache-hit fraction  = sum(cached_tokens) / sum(prompt_tokens)   [ground truth]
  * p50 / p99 latency
  * placement: mean distinct workers per prefix (affinity → ~1, round-robin → ~2)

Two policies bracket it: ``round_robin`` (balance, scatters prefixes) and
``affinity`` (Relay's PrefixRouter at cap=inf, concentrates prefixes). The
contrast IS the validation — if affinity doesn't raise cache-hit and concentrate
placement vs round-robin, the routing isn't doing what's claimed. Sweeping the
prefix length shows whether affinity's latency advantage grows with prefill cost
and crosses over, mirroring the simulated ~530-token crossover.

    # on the pod, from the Relay repo root, with two vLLM workers already up:
    PYTHONPATH=. python bench/vllm_validate.py \
        --workers http://localhost:8001 http://localhost:8002 \
        --model Qwen/Qwen2.5-0.5B-Instruct

Honest scope: this validates Relay's routing logic + the cache mechanism on real
hardware. It is a small-scale confirmation (two workers, a 0.5B model), not a
throughput benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from pathlib import Path

import numpy as np
import requests

from relay_core.types import WorkerState
from services.scheduler.router import PURE_AFFINITY, PrefixRouter, default_load
from relay_core.types import prefix_hash_of

RESULTS = Path(__file__).resolve().parent / "results"

_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
          "lima mike november oscar papa quebec romeo sierra tango").split()


def make_prefix(prefix_id: int, target_tokens: int, rng: np.random.Generator) -> str:
    """A deterministic shared block ~target_tokens long (≈1 word/token here; the
    server reports the true token count, so exactness doesn't matter)."""
    n_words = int(target_tokens * 1.3)
    seq = [f"doc{prefix_id}"] + [_WORDS[rng.integers(0, len(_WORDS))] for _ in range(n_words)]
    return "Context document: " + " ".join(seq) + ". Question: "


class LiveRouter:
    """Wraps Relay's PrefixRouter / a round-robin baseline with live inflight."""

    def __init__(self, worker_urls: list[str], policy: str, cap_factor: float):
        self.urls = {f"w{i}": u for i, u in enumerate(worker_urls)}
        self.model_tag = "m"
        self.states = [WorkerState(worker_id=f"w{i}", engine="vllm", models=(self.model_tag,),
                                   max_concurrent_batches=10_000)
                       for i in range(len(worker_urls))]
        self.inflight = {s.worker_id: 0 for s in self.states}
        self.lock = threading.Lock()
        self.policy = policy

        def load_fn(w: WorkerState) -> float:
            return float(self.inflight[w.worker_id])

        if policy == "round_robin":
            self._cycle = itertools.cycle(sorted(self.inflight))
            self._rr_load = load_fn
            self.router = None
        else:  # "affinity" or "bounded"
            self.router = PrefixRouter(
                self.states, load_cap_factor=cap_factor,
                load_fn=load_fn, admit_fn=lambda w: True)

    def pick(self, prefix_hash: str) -> str:
        with self.lock:
            if self.policy == "round_robin":
                m = min(self._rr_load(s) for s in self.states)
                least = [s for s in self.states if self._rr_load(s) == m]
                if len(least) == 1:
                    wid = least[0].worker_id
                else:
                    ids = {s.worker_id for s in least}
                    wid = next(w for w in self._cycle if w in ids)
            else:
                w = self.router.pick(self.model_tag, prefix_hash)
                wid = w.worker_id if w else self.states[0].worker_id
            self.inflight[wid] += 1
            return wid

    def done(self, wid: str) -> None:
        with self.lock:
            self.inflight[wid] -= 1


def run_policy(worker_urls, model, policy, cap_factor, prefix_tokens,
               pool_size, skew, n_requests, concurrency, max_tokens):
    rng = np.random.default_rng(0)
    prefixes = [make_prefix(i, prefix_tokens, rng) for i in range(pool_size)]
    # finite-Zipf sampling of prefix ids (same access pattern across policies)
    ranks = np.arange(1, pool_size + 1, dtype=np.float64)
    pmf = (1.0 / ranks ** skew); pmf /= pmf.sum()
    seq_rng = np.random.default_rng(12345)          # fixed: identical sequence per policy
    prefix_ids = seq_rng.choice(pool_size, size=n_requests, p=pmf)

    lr = LiveRouter(worker_urls, policy, cap_factor)
    results: list[dict] = []
    res_lock = threading.Lock()
    idx = itertools.count()
    counter = threading.Lock()
    next_i = [0]

    def take() -> int | None:
        with counter:
            i = next_i[0]
            if i >= n_requests:
                return None
            next_i[0] += 1
            return i

    def worker_loop():
        while True:
            i = take()
            if i is None:
                return
            pid = int(prefix_ids[i])
            prompt = prefixes[pid] + f" [req {i}] Answer in one word."
            ph = prefix_hash_of(prefixes[pid])
            wid = lr.pick(ph)
            url = lr.urls[wid]
            t0 = time.perf_counter()
            cached = ptok = -1
            ok = False
            try:
                r = requests.post(f"{url}/v1/completions", json={
                    "model": model, "prompt": prompt,
                    "max_tokens": max_tokens, "temperature": 0.0}, timeout=180)
                d = r.json()
                u = d.get("usage", {})
                ptok = int(u.get("prompt_tokens", 0))
                ptd = u.get("prompt_tokens_details") or {}
                cached = int(ptd.get("cached_tokens", 0) or 0)
                ok = "choices" in d
            except Exception as e:  # noqa: BLE001
                err = str(e)[:80]
            finally:
                lr.done(wid)
            lat = (time.perf_counter() - t0) * 1000.0
            with res_lock:
                results.append({"i": i, "prefix_id": pid, "worker": wid,
                                "prompt_tokens": ptok, "cached_tokens": cached,
                                "latency_ms": lat, "ok": ok})

    threads = [threading.Thread(target=worker_loop) for _ in range(concurrency)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    ok = [r for r in results if r["ok"] and r["prompt_tokens"] > 0]
    if not ok:
        return {"policy": policy, "prefix_tokens": prefix_tokens, "n_ok": 0,
                "error": "no successful responses"}
    lat = np.array([r["latency_ms"] for r in ok])
    tot_prompt = sum(r["prompt_tokens"] for r in ok)
    tot_cached = sum(r["cached_tokens"] for r in ok)
    # placement: distinct workers per prefix id
    by_pref: dict[int, set] = {}
    for r in ok:
        by_pref.setdefault(r["prefix_id"], set()).add(r["worker"])
    distinct = np.mean([len(s) for s in by_pref.values()])
    per_worker = {}
    for r in ok:
        per_worker[r["worker"]] = per_worker.get(r["worker"], 0) + 1
    return {
        "policy": policy, "prefix_tokens": prefix_tokens, "n_ok": len(ok),
        "cache_hit_frac": round(tot_cached / tot_prompt, 4) if tot_prompt else 0.0,
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
        "mean_distinct_workers_per_prefix": round(float(distinct), 3),
        "per_worker_counts": per_worker,
        "throughput_rps": round(len(ok) / wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Relay routing on real vLLM workers.")
    ap.add_argument("--workers", nargs="+", required=True,
                    help="vLLM base URLs, e.g. http://localhost:8001 http://localhost:8002")
    ap.add_argument("--model", required=True, help="model id served by the workers")
    ap.add_argument("--prefix-tokens", nargs="+", type=int, default=[256, 1024, 4096],
                    help="shared-prefix lengths to sweep")
    ap.add_argument("--policies", nargs="+", default=["round_robin", "affinity"],
                    choices=["round_robin", "affinity", "bounded"])
    ap.add_argument("--cap-factor", type=float, default=1.5, help="cap for the 'bounded' policy")
    ap.add_argument("--pool-size", type=int, default=64)
    ap.add_argument("--skew", type=float, default=1.1)
    ap.add_argument("--n-requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    print(f"[validate] workers={args.workers} model={args.model}")
    print(f"[validate] sweep prefix_tokens={args.prefix_tokens} policies={args.policies} "
          f"n={args.n_requests} concurrency={args.concurrency}\n")

    rows: list[dict] = []
    for L in args.prefix_tokens:
        for pol in args.policies:
            cap = PURE_AFFINITY if pol == "affinity" else args.cap_factor
            r = run_policy(args.workers, args.model, pol, cap, L,
                           args.pool_size, args.skew, args.n_requests,
                           args.concurrency, args.max_tokens)
            rows.append(r)
            if r.get("n_ok"):
                print(f"  L={L:>5} {pol:<11} hit={r['cache_hit_frac']:.3f} "
                      f"p50={r['p50_ms']:>7.1f} p99={r['p99_ms']:>7.1f} "
                      f"distinct_workers/prefix={r['mean_distinct_workers_per_prefix']:.2f} "
                      f"{r['per_worker_counts']}")
            else:
                print(f"  L={L:>5} {pol:<11} FAILED: {r.get('error')}")

    # crossover view: affinity vs round_robin p99 at each prefix length
    print("\n[validate] affinity speedup vs round-robin (p99 RR / p99 affinity):")
    by = {(r["prefix_tokens"], r["policy"]): r for r in rows if r.get("n_ok")}
    for L in args.prefix_tokens:
        rr, af = by.get((L, "round_robin")), by.get((L, "affinity"))
        if rr and af and af["p99_ms"] > 0:
            sp = rr["p99_ms"] / af["p99_ms"]
            tag = "affinity wins" if sp > 1 else "round-robin wins"
            print(f"  L={L:>5}: speedup {sp:.2f}×  ({tag})   "
                  f"cache-hit {rr['cache_hit_frac']:.2f} → {af['cache_hit_frac']:.2f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "vllm_validation.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\n[validate] wrote {out}")
    print("[validate] sanity checks: affinity should show higher cache-hit and "
          "~1 distinct worker/prefix; round-robin ~2 and lower cache-hit. If not, "
          "routing is not concentrating prefixes — investigate before trusting p99.")


if __name__ == "__main__":
    main()
