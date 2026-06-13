"""Crossover experiment: *when* does prefix-aware routing beat round-robin?
(DESIGN.md §13 — the honest follow-up to the Pareto sweep.)

The Pareto sweep answers "what does the cache-hit/imbalance/p99 trade look like at
one operating point". This answers the sharper question a reviewer asks: *under
what conditions is the technique worth it at all?*

The physics: every request pays prefill (build the prompt's KV) once, then decodes
output tokens one by one. Prefix-affinity routing concentrates same-prefix traffic
on one worker so its KV cache is reused — it trades **load balance** for **saved
prefill**. So the technique wins exactly when prefill cost is large relative to the
decode cost every request pays regardless of where it lands. Hold the two per-token
costs FIXED at their measured values (read from ``calibration.json`` — the real M2
rates) and sweep the **shared-prefix length**: short prefixes → cheap prefill →
round-robin wins; long prefixes (RAG, big system prompts) → prefill dominates →
affinity wins. The crossover between them is the result.

Rigor: every point is run across ``--seeds`` independent replicates (different
arrival, engine, and workload seeds), and we report **mean ± 95% CI** for the
speedup at each prefix length AND a **95% CI on the crossover token value** itself
(interpolated per seed, then aggregated). A single-seed number is an anecdote; this
is the honest version. CIs use a Student-t multiplier (no scipy dependency).

Only the per-token rates come from the measurement (they are robust); the fixed
per-request overhead ``alpha`` is a small constant, NOT the noisy fitted intercept
(which on an 8 GB box under load is mostly thrashing artifact). Stated in output.

    PYTHONPATH=. python bench/crossover.py
    PYTHONPATH=. python bench/crossover.py --seeds 20 --output-tokens 16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench.simulate import Scenario, run_one  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
CALIB = RESULTS / "calibration.json"

PREFIX_TOKEN_GRID = (128, 256, 512, 1024, 2048, 4096, 8192)
PREFIX_CAPS = (1.25, 1.5, 2.0, float("inf"))
FALLBACK_DECODE_MS_PER_TOK = 30.0
FALLBACK_PREFILL_MS_PER_TOK = 1.3

# Student-t two-sided 95% multipliers by degrees of freedom (n-1); 1.96 beyond 30.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t95(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    return _T95.get(dof, 1.96)


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Return (mean, 95% CI half-width). Half-width is 0 for a single sample."""
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = a.size
    if n == 0:
        return float("nan"), float("nan")
    m = float(a.mean())
    if n == 1:
        return m, 0.0
    sd = float(a.std(ddof=1))
    return m, _t95(n - 1) * sd / np.sqrt(n)


def load_rates() -> tuple[float, float, str]:
    if CALIB.exists():
        d = json.loads(CALIB.read_text())
        dec, pre = d.get("decode_ms_per_token"), d.get("prefill_ms_per_token")
        if dec and pre:
            return float(dec), float(pre), f"measured ({d.get('source', '?')})"
    return (
        FALLBACK_DECODE_MS_PER_TOK,
        FALLBACK_PREFILL_MS_PER_TOK,
        "fallback (no measured rates in calibration.json)",
    )


def stable_offered_rps(s: Scenario) -> float:
    unit_ms = s.alpha_ms + s.beta_ms + s.prefill_ms
    cap = s.n_workers * s.max_concurrent_batches * 1000.0 / unit_ms
    return max(1.0, round(0.75 * cap, 3))


def run_point(
    prefix_tokens: int,
    decode_rate: float,
    prefill_rate: float,
    output_tokens: int,
    alpha_ms: float,
    n_requests: int,
    seed_index: int,
) -> dict:
    """One (prefix_length, seed) replicate. Distinct arrival/engine/workload seeds."""
    beta_ms = decode_rate * output_tokens
    prefill_ms = prefill_rate * prefix_tokens
    base_wl = Scenario().workload  # keep pool_size/skew, vary seed
    s = Scenario(
        n_requests=n_requests,
        warmup_requests=n_requests // 10,
        alpha_ms=alpha_ms,
        beta_ms=beta_ms,
        prefill_ms=prefill_ms,
        arrival_seed=11 + seed_index,
        engine_seed=23 + 100 * seed_index,
        workload=replace(base_wl, seed=(base_wl.seed or 0) + 1 + seed_index),
    )
    s.offered_rps = stable_offered_rps(s)

    rr = run_one(s, "round_robin", None)
    best = min((run_one(s, "prefix", cf) for cf in PREFIX_CAPS), key=lambda r: r.p99_ms)
    return {
        "prefix_tokens": prefix_tokens,
        "prefill_ms": prefill_ms,
        "beta_ms": beta_ms,
        "offered_rps": s.offered_rps,
        "rr_p99": rr.p99_ms,
        "prefix_p99": best.p99_ms,
        "rr_hit": rr.cache_hit_rate,
        "prefix_hit": best.cache_hit_rate,
        "rr_imbalance": rr.load_imbalance,
        "prefix_imbalance": best.load_imbalance,
        "prefix_best_cap": best.cap_factor,
        "speedup": rr.p99_ms / best.p99_ms,  # >1 ⇒ prefix wins
    }


def crossover_of(prefix_tokens: np.ndarray, speedup: np.ndarray) -> float | None:
    """Interpolate the prefix length where speedup first crosses 1.0."""
    for i in range(len(prefix_tokens) - 1):
        a, b = speedup[i] - 1.0, speedup[i + 1] - 1.0
        if a == 0.0:
            return float(prefix_tokens[i])
        if a * b < 0:
            t = (1.0 - speedup[i]) / (speedup[i + 1] - speedup[i])
            return float(prefix_tokens[i] + t * (prefix_tokens[i + 1] - prefix_tokens[i]))
    return None


def plot(
    per_len: pd.DataFrame,
    cx_mean: float | None,
    cx_ci: float,
    prov: str,
    decode_rate: float,
    prefill_rate: float,
    n_seeds: int,
    out: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Relay — when does prefix-aware routing beat round-robin?\n"
        f"rates: decode {decode_rate:.1f} ms/tok, prefill {prefill_rate:.2f} ms/tok "
        f"({prov}) · mean ± 95% CI over {n_seeds} seeds",
        fontsize=12,
    )
    x = per_len["prefix_tokens"].to_numpy()

    ax1.plot(x, per_len["rr_p99_mean"], "o-", color="#c44", label="round-robin")
    ax1.fill_between(
        x,
        per_len["rr_p99_mean"] - per_len["rr_p99_ci"],
        per_len["rr_p99_mean"] + per_len["rr_p99_ci"],
        color="#c44",
        alpha=0.15,
    )
    ax1.plot(x, per_len["prefix_p99_mean"], "s-", color="#268", label="prefix-affinity (best cap)")
    ax1.fill_between(
        x,
        per_len["prefix_p99_mean"] - per_len["prefix_p99_ci"],
        per_len["prefix_p99_mean"] + per_len["prefix_p99_ci"],
        color="#268",
        alpha=0.15,
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("shared-prefix length (tokens)")
    ax1.set_ylabel("p99 latency (ms)")
    ax1.set_title("p99 vs prefix length")
    ax1.legend()
    ax1.grid(alpha=0.3)

    sm, sci = per_len["speedup_mean"].to_numpy(), per_len["speedup_ci"].to_numpy()
    ax2.axhline(1.0, color="black", ls="--", lw=0.8)
    ax2.plot(x, sm, "o-", color="#3a3")
    ax2.fill_between(x, sm - sci, sm + sci, color="#3a3", alpha=0.2, label="speedup 95% CI")
    if cx_mean is not None:
        if np.isfinite(cx_ci) and cx_ci > 0:
            ax2.axvspan(cx_mean - cx_ci, cx_mean + cx_ci, color="#555", alpha=0.15)
        ax2.axvline(cx_mean, color="#555", ls=":", lw=1.2)
        lab = (
            f"crossover ≈ {cx_mean:.0f} ± {cx_ci:.0f} tok"
            if np.isfinite(cx_ci) and cx_ci > 0
            else f"crossover ≈ {cx_mean:.0f} tok"
        )
        ax2.annotate(lab, xy=(cx_mean, 1.0), xytext=(cx_mean, sm.max()), fontsize=9, ha="center")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("shared-prefix length (tokens)")
    ax2.set_ylabel("p99 speedup  (round-robin p99 / prefix p99)")
    ax2.set_title("prefix routing pays off above the crossover")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_md(
    per_len: pd.DataFrame,
    cx_mean: float | None,
    cx_ci: float,
    n_cross: int,
    n_seeds: int,
    prov: str,
    decode_rate: float,
    prefill_rate: float,
    alpha_ms: float,
    output_tokens: int,
    n_requests: int,
    out: Path,
) -> None:
    lo, hi = per_len.iloc[0], per_len.iloc[-1]
    if cx_mean is None:
        cx = "outside the swept range (no sign change)"
    elif np.isfinite(cx_ci) and cx_ci > 0:
        cx = f"**≈ {cx_mean:.0f} ± {cx_ci:.0f} tokens** (95% CI, {n_cross}/{n_seeds} seeds crossed)"
    else:
        cx = f"**≈ {cx_mean:.0f} tokens**"
    lines = [
        "# Crossover: when prefix-aware routing beats round-robin\n",
        f"Per-token costs held fixed at their {prov} values "
        f"(**decode {decode_rate:.1f} ms/tok**, **prefill {prefill_rate:.2f} ms/tok**); "
        f"fixed overhead alpha = {alpha_ms:.0f} ms; completion = {output_tokens} tokens; "
        f"{n_requests:,} requests/run; **{n_seeds} seeds** per point. Only the "
        "**shared-prefix length** varies.\n",
        f"**Crossover prefix length: {cx}.** Below it, decode dominates and "
        "round-robin's better balance wins; above it, prefill dominates and reusing "
        "the KV cache (prefix-affinity) wins.\n",
        f"- short prefix ({int(lo.prefix_tokens)} tok): speedup "
        f"**{lo.speedup_mean:.2f} ± {lo.speedup_ci:.2f}×** "
        f"(round-robin {'wins' if lo.speedup_mean < 1 else 'loses'}).",
        f"- long prefix ({int(hi.prefix_tokens)} tok): speedup "
        f"**{hi.speedup_mean:.2f} ± {hi.speedup_ci:.2f}×**.\n",
        "Every figure is mean ± 95% CI across independent seeds (Student-t). The "
        "takeaway is the *condition*, not a single number: KV-cache-aware routing "
        "is worth its load-imbalance cost only once the shared prefix is long enough "
        "that prefill is the bottleneck — the RAG / shared-system-prompt regime. On "
        "short chat prompts it is a net loss, and a router should load-balance there.\n",
        "Full per-length data with CIs: `crossover.csv`. Figure: `crossover.png`.\n",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed-replicated prefix-routing crossover sweep.")
    ap.add_argument("--seeds", type=int, default=10, help="independent replicates per point")
    ap.add_argument(
        "--output-tokens",
        type=int,
        default=64,
        help="nominal completion length (defines one item's decode)",
    )
    ap.add_argument(
        "--alpha-ms",
        type=float,
        default=30.0,
        help="fixed per-request overhead (NOT the noisy fitted intercept)",
    )
    ap.add_argument("--n-requests", type=int, default=10_000)
    args = ap.parse_args()

    decode_rate, prefill_rate, prov = load_rates()
    print(
        f"[crossover] rates: decode={decode_rate:.2f} ms/tok, "
        f"prefill={prefill_rate:.2f} ms/tok  [{prov}]"
    )
    print(
        f"[crossover] {args.seeds} seeds × {len(PREFIX_TOKEN_GRID)} prefix lengths "
        f"× {len(PREFIX_CAPS) + 1} policies, {args.n_requests:,} req/run ..."
    )

    # raw[(prefix_tokens)] -> list of per-seed dicts
    raw: dict[int, list[dict]] = {pt: [] for pt in PREFIX_TOKEN_GRID}
    per_seed_curls: list[float] = []
    for r in range(args.seeds):
        seed_rows = [
            run_point(
                pt, decode_rate, prefill_rate, args.output_tokens, args.alpha_ms, args.n_requests, r
            )
            for pt in PREFIX_TOKEN_GRID
        ]
        for row in seed_rows:
            raw[row["prefix_tokens"]].append(row)
        xs = np.array([row["prefix_tokens"] for row in seed_rows], float)
        ys = np.array([row["speedup"] for row in seed_rows], float)
        cx = crossover_of(xs, ys)
        if cx is not None:
            per_seed_curls.append(cx)
        print(
            f"  seed {r + 1}/{args.seeds} done"
            + (f" — crossover {cx:.0f} tok" if cx is not None else " — no crossing")
        )

    # aggregate per prefix length
    per_len_rows = []
    for pt in PREFIX_TOKEN_GRID:
        rows = raw[pt]
        sp_m, sp_ci = mean_ci([x["speedup"] for x in rows])
        rr_m, rr_ci = mean_ci([x["rr_p99"] for x in rows])
        px_m, px_ci = mean_ci([x["prefix_p99"] for x in rows])
        per_len_rows.append(
            {
                "prefix_tokens": pt,
                "prefill_ms": round(rows[0]["prefill_ms"], 1),
                "beta_ms": round(rows[0]["beta_ms"], 1),
                "offered_rps": rows[0]["offered_rps"],
                "rr_p99_mean": round(rr_m, 1),
                "rr_p99_ci": round(rr_ci, 1),
                "prefix_p99_mean": round(px_m, 1),
                "prefix_p99_ci": round(px_ci, 1),
                "speedup_mean": round(sp_m, 4),
                "speedup_ci": round(sp_ci, 4),
                "rr_hit_mean": round(np.mean([x["rr_hit"] for x in rows]), 3),
                "prefix_hit_mean": round(np.mean([x["prefix_hit"] for x in rows]), 3),
                "rr_imbalance_mean": round(np.mean([x["rr_imbalance"] for x in rows]), 3),
                "prefix_imbalance_mean": round(np.mean([x["prefix_imbalance"] for x in rows]), 3),
                "n_seeds": len(rows),
            }
        )
    per_len = pd.DataFrame(per_len_rows)

    cx_mean, cx_ci = mean_ci(per_seed_curls) if per_seed_curls else (None, float("nan"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    per_len.to_csv(RESULTS / "crossover.csv", index=False)
    plot(
        per_len,
        cx_mean,
        cx_ci,
        prov,
        decode_rate,
        prefill_rate,
        args.seeds,
        RESULTS / "crossover.png",
    )
    write_md(
        per_len,
        cx_mean,
        cx_ci,
        len(per_seed_curls),
        args.seeds,
        prov,
        decode_rate,
        prefill_rate,
        args.alpha_ms,
        args.output_tokens,
        args.n_requests,
        RESULTS / "CROSSOVER.md",
    )

    print(per_len.to_string(index=False))
    if cx_mean is not None:
        print(
            f"[crossover] crossover ≈ {cx_mean:.0f} ± {cx_ci:.0f} tokens "
            f"(95% CI, {len(per_seed_curls)}/{args.seeds} seeds crossed)"
        )
    else:
        print("[crossover] no crossing within the swept range")
    print(f"[crossover] wrote {RESULTS / 'crossover.csv'}, crossover.png, CROSSOVER.md")


if __name__ == "__main__":
    main()
