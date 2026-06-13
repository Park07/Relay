"""Sweep runner — trace the Pareto frontier (DESIGN.md §13.2) and write artifacts.

Runs the in-process virtual-time simulator (``bench/simulate.py``) across the two
reference policies plus a bounded-load sweep, and emits, under ``bench/results/``:

  * ``frontier.csv``  — one row per policy point (round_robin + prefix sweep);
  * ``frontier.png``  — the Pareto-frontier figure;
  * ``RESULTS.md``    — a human-readable summary populated with the *actual*
                        produced numbers (this is what the README quotes).

If ``bench/results/calibration.json`` exists (written by ``bench/calibrate.py``)
its fitted ``alpha_ms``/``beta_ms`` are used for the engine; otherwise the
calibrated defaults baked into :class:`~bench.simulate.Scenario` are used. Either
way the source is recorded in the output so the provenance is explicit.

Usage::

    PYTHONPATH=. python bench/run.py            # full 30k-request sweep
    PYTHONPATH=. python bench/run.py --quick     # fast 8k-request smoke sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from bench.simulate import RunResult, Scenario, run_one  # noqa: E402
from bench.workload import ZipfianPrefixWorkload  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

# Bounded-load knob sweep. The two endpoints (round_robin and prefix@inf) are
# added separately; these are the interesting middle (DESIGN.md §13.2).
CAP_FACTORS = [1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, float("inf")]


# --------------------------------------------------------------------------- #
def load_calibration() -> dict | None:
    f = RESULTS / "calibration.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def build_scenario(quick: bool) -> tuple[Scenario, str]:
    kw: dict = {}
    src = "Scenario defaults (calibrated constants)"
    cal = load_calibration()
    measured = False
    if cal:
        kw["alpha_ms"] = float(cal["alpha_ms"])
        kw["beta_ms"] = float(cal["beta_ms"])
        if "prefill_ms" in cal:  # newer calibrations carry it
            kw["prefill_ms"] = float(cal["prefill_ms"])
        measured = bool(cal.get("measured", False))
        src = f"calibration.json (source={cal.get('source', 'unknown')})"

    n = 8_000 if quick else 30_000
    s = Scenario(n_requests=n, warmup_requests=n // 10, **kw)

    # The default offered_rps (240) is sized for the fast synthetic engine. Real
    # measured latencies are far larger, so a fixed rate would overload the fleet
    # and p99 would diverge (an unstable queue, not a result). When the constants
    # are measured, size the offered load to a conservative fraction of fleet
    # capacity instead, so every policy runs in a stable regime and p99 is
    # meaningful. Conservative per-item service time = alpha + beta + prefill
    # (a b=1 cache *miss*: the slowest unit, ignoring batching/cache amortisation),
    # which under-counts throughput and therefore keeps the queue stable.
    if measured:
        unit_ms = s.alpha_ms + s.beta_ms + s.prefill_ms
        capacity_rps = s.n_workers * s.max_concurrent_batches * 1000.0 / unit_ms
        s.offered_rps = max(1.0, round(0.75 * capacity_rps, 2))

    return s, src


def run_sweep(s: Scenario) -> tuple[pd.DataFrame, dict[str, RunResult]]:
    results: list[RunResult] = [run_one(s, "round_robin", None)]
    for cf in CAP_FACTORS:
        results.append(run_one(s, "prefix", cf))
    df = pd.DataFrame([r.row() for r in results])

    by_label: dict[str, RunResult] = {"round_robin": results[0]}
    for r in results[1:]:
        if r.cap_factor == float("inf"):
            by_label["affinity"] = r
        elif r.cap_factor == 1.5:
            by_label["knee"] = r
    return df, by_label


# --------------------------------------------------------------------------- #
def plot_frontier(df: pd.DataFrame, s: Scenario, out: Path) -> None:
    rr = df[df.policy == "round_robin"].iloc[0]
    pf = df[df.policy == "prefix"].copy().reset_index(drop=True)
    finite = pf[np.isfinite(pf.cap_factor)]
    inf_row = pf[~np.isfinite(pf.cap_factor)].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(
        "Relay — prefix-aware bounded-load routing: Pareto frontier\n"
        f"(4 workers x {s.max_concurrent_batches} slots, Zipf s={s.workload.skew} "
        f"over {s.workload.pool_size} prefixes, offered {s.offered_rps:.0f} req/s, "
        f"cache={s.cache_capacity} prefixes/worker)",
        fontsize=11,
    )
    cmap = plt.cm.viridis
    caps = finite.cap_factor.to_numpy()
    norm = plt.Normalize(caps.min(), caps.max())

    # Panel 1: cache hit rate vs load imbalance (the core trade) ------------ #
    ax = axes[0]
    ax.scatter(
        finite.load_imbalance, finite.cache_hit_rate, c=caps, cmap=cmap, norm=norm, s=70, zorder=3
    )
    for _, r in finite.iterrows():
        ax.annotate(
            f"{r.cap_factor:g}",
            (r.load_imbalance, r.cache_hit_rate),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
        )
    ax.scatter(
        [inf_row.load_imbalance],
        [inf_row.cache_hit_rate],
        marker="*",
        s=240,
        color="crimson",
        zorder=4,
        label="prefix, cap=inf (pure affinity)",
    )
    ax.scatter(
        [rr.load_imbalance],
        [rr.cache_hit_rate],
        marker="s",
        s=90,
        color="black",
        zorder=4,
        label="round-robin (balance)",
    )
    ax.set_xlabel("load imbalance  (max/mean items per worker)")
    ax.set_ylabel("cache-hit rate")
    ax.set_title("hit-rate vs imbalance\n(label = load_cap_factor)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: p99 latency vs load imbalance -------------------------------- #
    ax = axes[1]
    ax.scatter(finite.load_imbalance, finite.p99_ms, c=caps, cmap=cmap, norm=norm, s=70, zorder=3)
    ax.scatter(
        [inf_row.load_imbalance], [inf_row.p99_ms], marker="*", s=240, color="crimson", zorder=4
    )
    ax.scatter([rr.load_imbalance], [rr.p99_ms], marker="s", s=90, color="black", zorder=4)
    ax.axhline(rr.p99_ms, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("load imbalance  (max/mean items per worker)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("p99 vs imbalance\n(higher locality -> less prefill -> lower p99)")
    ax.grid(alpha=0.3)

    # Panel 3: the knee — hit-rate and p99 vs the knob ---------------------- #
    ax = axes[2]
    labels = ["RR"] + [f"{c:g}" for c in finite.cap_factor] + ["inf"]
    hit = [rr.cache_hit_rate] + list(finite.cache_hit_rate) + [inf_row.cache_hit_rate]
    p99 = [rr.p99_ms] + list(finite.p99_ms) + [inf_row.p99_ms]
    xx = np.arange(len(labels))
    l1 = ax.plot(xx, hit, "o-", color="tab:blue", label="cache-hit rate")[0]
    ax.set_ylabel("cache-hit rate", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("policy:  round-robin / balance  <-  load_cap_factor  ->  affinity / locality")
    ax2 = ax.twinx()
    l2 = ax2.plot(xx, p99, "s--", color="tab:red", label="p99 latency (ms)")[0]
    ax2.set_ylabel("p99 latency (ms)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("hit-rate & p99 across the knob")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2], fontsize=8, loc="center right")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def write_results_md(
    df: pd.DataFrame, by: dict[str, RunResult], s: Scenario, cal_src: str, out: Path
) -> None:
    rr = by["round_robin"]
    af = by["affinity"]
    knee = by.get("knee", af)

    # Workload skew, realized.
    wl = ZipfianPrefixWorkload(s.workload)
    _ = wl.generate(min(s.n_requests, 30_000))
    top8 = wl.realized_topk_mass(8)

    def ratio(a: float, b: float) -> float:
        return a / b if b else float("nan")

    p99_gain = ratio(rr.p99_ms, knee.p99_ms)
    imb_cost = ratio(knee.load_imbalance, rr.load_imbalance)
    p99_gain_af = ratio(rr.p99_ms, af.p99_ms)

    lines = []
    lines.append("# Relay — routing benchmark results\n")
    lines.append(
        "*Generated by `bench/run.py` from the in-process virtual-time simulator "
        "(`bench/simulate.py`). These are real numbers produced on this machine, "
        "reproducibly, with no GPU.*\n"
    )
    lines.append("## Headline\n")
    lines.append(
        "Sweeping the router's single knob `load_cap_factor` from pure affinity "
        "(cap = inf) toward round-robin traces a clean Pareto frontier between "
        "**KV-cache locality** and **load balance**:\n"
    )
    lines.append(
        f"- **Cache-hit rate** rises from **{rr.cache_hit_rate:.1%}** (round-robin) "
        f"to **{af.cache_hit_rate:.1%}** (pure affinity).\n"
        f"- **p99 latency** falls from **{rr.p99_ms:.0f} ms** (round-robin) to "
        f"**{af.p99_ms:.0f} ms** (pure affinity) — a **{p99_gain_af:.1f}x** reduction "
        f"— because cache hits skip the {s.prefill_ms:.0f} ms prefill.\n"
        f"- The **cost** is load imbalance: from **{rr.load_imbalance:.2f}x** "
        f"(round-robin) to **{af.load_imbalance:.2f}x** (pure affinity, where the "
        f"hottest prefix pins its traffic to one worker).\n"
    )
    lines.append("### The knee (recommended operating point)\n")
    lines.append(
        f"At **`load_cap_factor = {knee.cap_factor:g}`** the bounded-load policy "
        f"captures most of the locality benefit for a modest imbalance cost:\n"
        f"- cache-hit **{knee.cache_hit_rate:.1%}**, p99 **{knee.p99_ms:.0f} ms** "
        f"(**{p99_gain:.1f}x** lower than round-robin), "
        f"imbalance **{knee.load_imbalance:.2f}x** "
        f"(**{imb_cost:.2f}x** the round-robin baseline).\n"
    )
    lines.append("![Pareto frontier](frontier.png)\n")

    lines.append("## Full sweep\n")
    show = df.copy()

    def cap_str(row) -> str:
        if row.policy == "round_robin":
            return "—"
        return "inf" if not np.isfinite(row.cap_factor) else f"{row.cap_factor:g}"

    show["cap_factor"] = show.apply(cap_str, axis=1)
    cols = [
        "policy",
        "cap_factor",
        "cache_hit_rate",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "throughput_rps",
        "load_imbalance",
        "mean_batch_size",
        "utilization",
    ]
    hdr = ["policy", "cap", "hit", "p50", "p95", "p99", "thrpt", "imbal", "batch", "util"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for _, r in show[cols].iterrows():
        lines.append(
            f"| {r.policy} | {r.cap_factor} | {r.cache_hit_rate:.3f} | "
            f"{r.p50_ms:.0f} | {r.p95_ms:.0f} | {r.p99_ms:.0f} | "
            f"{r.throughput_rps:.0f} | {r.load_imbalance:.2f} | "
            f"{r.mean_batch_size:.1f} | {r.utilization:.2f} |"
        )
    lines.append("")

    lines.append("## Per-worker locality (why imbalance is the cost)\n")
    lines.append(
        "Round-robin spreads every prefix across all workers, so each worker's "
        "cache-hit rate is identical and item counts are flat. Pure affinity makes "
        "one worker the home of the hottest prefix: its hit-rate is high but it "
        "processes far more items.\n"
    )
    lines.append("| worker | round-robin hit | round-robin items | affinity hit | affinity items |")
    lines.append("|---|---|---|---|---|")
    for wid in sorted(rr.per_worker_hit):
        lines.append(
            f"| {wid} | {rr.per_worker_hit[wid]:.3f} | "
            f"{rr.per_worker_items[wid]} | {af.per_worker_hit[wid]:.3f} | "
            f"{af.per_worker_items[wid]} |"
        )
    lines.append("")

    lines.append("## Setup (for reproducibility)\n")
    lines.append(
        f"- **Engine model:** CacheAwareMockEngine, latency "
        f"`(alpha + beta*b + prefill*distinct_missed_prefixes) * jitter` with "
        f"alpha={s.alpha_ms:.1f} ms, beta={s.beta_ms:.1f} ms, "
        f"prefill={s.prefill_ms:.0f} ms, lognormal jitter sigma={s.jitter_sigma:.2f}. "
        f"alpha/beta provenance: {cal_src}.\n"
        f"- **Fleet:** {s.n_workers} workers x {s.max_concurrent_batches} concurrent "
        f"batches, max batch {s.max_batch}, cache capacity {s.cache_capacity} "
        f"distinct prefixes per worker.\n"
        f"- **Workload:** finite Zipf, s={s.workload.skew}, pool={s.workload.pool_size} "
        f"shared prefixes ({s.workload.prefix_chars}-char shared block + "
        f"{s.workload.suffix_chars}-char unique suffix); realized top-8 prefix mass "
        f"= {top8:.1%}.\n"
        f"- **Arrivals:** Poisson, offered {s.offered_rps:.0f} req/s; "
        f"{s.n_requests:,} requests, first {s.warmup_requests:,} excluded from "
        f"steady-state metrics (cold-cache warmup).\n"
        f"- **Scheduling:** deadline batch former, default budget "
        f"{s.budget_default_ms:.0f} ms (DESIGN.md §8.1); routing per request at "
        f"admission into per-worker queues (DESIGN.md §8.2).\n"
        f"- **Seeds:** workload={s.workload.seed}, arrivals={s.arrival_seed}, "
        f"engine={s.engine_seed}. Every policy sees the identical request/arrival "
        f"stream.\n"
    )
    lines.append("## Method note (a negative result worth keeping)\n")
    lines.append(
        "Routing *formed* batches by the prefix of the batch head — the literal "
        "composition of a single global §8.1 former with §8.2 — yields **no "
        "locality** (hit-rate identical across policies): a globally-formed batch "
        "mixes prefixes, so whichever worker takes it must prefill all of them. "
        "Locality only materializes when routing happens **per request at "
        "admission** into per-worker queues, so each worker batches its own "
        "prefix-coherent traffic. That is the topology measured here, and it is how "
        "production prefix-aware schedulers (SGLang, vLLM-router) are organized.\n"
    )

    out.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast 8k-request smoke sweep")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    s, cal_src = build_scenario(args.quick)
    print(
        f"[run] scenario: {s.n_requests:,} reqs, offered {s.offered_rps:.0f} rps, "
        f"{s.n_workers}x{s.max_concurrent_batches} workers, cache={s.cache_capacity}; "
        f"alpha/beta from {cal_src}"
    )

    df, by = run_sweep(s)

    csv_path = RESULTS / "frontier.csv"
    df.to_csv(csv_path, index=False)
    print(f"[run] wrote {csv_path}")

    png_path = RESULTS / "frontier.png"
    plot_frontier(df, s, png_path)
    print(f"[run] wrote {png_path}")

    md_path = RESULTS / "RESULTS.md"
    write_results_md(df, by, s, cal_src, md_path)
    print(f"[run] wrote {md_path}")

    rr, af = by["round_robin"], by["affinity"]
    knee = by.get("knee", af)
    print(
        f"[run] DONE | RR hit={rr.cache_hit_rate:.3f} p99={rr.p99_ms:.0f}ms "
        f"imbal={rr.load_imbalance:.2f}  ->  "
        f"knee(cap={knee.cap_factor:g}) hit={knee.cache_hit_rate:.3f} "
        f"p99={knee.p99_ms:.0f}ms imbal={knee.load_imbalance:.2f}  ->  "
        f"affinity hit={af.cache_hit_rate:.3f} p99={af.p99_ms:.0f}ms "
        f"imbal={af.load_imbalance:.2f}"
    )


if __name__ == "__main__":
    main()
