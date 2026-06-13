"""Calibrate the engine's latency constants from real hardware (DESIGN.md §5.3,
ADR-11) — or from a clearly-labelled synthetic ground truth.

The latency model the simulator uses is

    batch_latency(b) = alpha + beta * b + prefill_ms * (#distinct missed prefixes)

so three constants need values:
  * ``alpha_ms``    fixed per-batch overhead (launch/dispatch/detokenize),
  * ``beta_ms``     marginal cost of one item's decode, and
  * ``prefill_ms``  cost to build one prefix's KV — the cost a cache *hit* avoids.

Two modes:

  ``--ollama``   Measure on a live Ollama daemon (Metal-accelerated on a Mac).
                 IMPORTANT: a laptop Ollama runs a SINGLE slot (``-np 1``), so it
                 does not batch multiple sequences in one forward pass. We do NOT
                 fake a batch by firing concurrent requests (that just measures a
                 queue draining serially — the bug this file used to have, which
                 produced a ~1700 ms/item "beta" and a meaningless frontier).
                 Instead we measure the quantities that single-slot inference can
                 report *honestly*, straight from llama.cpp's own timing fields in
                 the /api/generate response:
                   - prefill cost per prompt token  (prompt_eval_duration/count)
                   - decode cost per output token   (eval_duration/eval_count)
                 and derive the engine constants from those. ``beta`` here is the
                 per-item decode cost measured SERIALLY; a true batching engine
                 (vLLM on a multi-slot GPU) would show beta shrink as batch size
                 grows — that batching-speedup measurement is the cloud step, and
                 is deliberately out of scope on an 8 GB single-slot box.

  ``--synthetic``  Plausible samples from a known ground truth, zero deps, fully
                 reproducible. Records ``"source": "synthetic"`` so it is never
                 mistaken for a hardware measurement.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CALIB_PATH = RESULTS_DIR / "calibration.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Output lengths swept to separate fixed overhead from per-token decode.
DECODE_TOKEN_GRID = (16, 32, 64, 128)
# The shared-prefix block length the workload uses (chars); we size prefill to it
# so the measured prefill_ms is the cost a real cache hit would actually save.
PREFIX_CHARS = 800
# One simulator "item" represents this many decoded tokens (a short completion on
# top of a long shared prompt — the regime where prefix caching matters).
NOMINAL_OUTPUT_TOKENS = 64


# --------------------------------------------------------------------------- #
# Linear fit helper (used by both paths)
# --------------------------------------------------------------------------- #
def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, R^2) for ys ~ slope*xs + intercept."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return float(slope), float(intercept), 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# Real path: Ollama (single-slot — measure prefill & decode honestly)
# --------------------------------------------------------------------------- #
def _unique_prefix(rng: np.random.Generator, n_chars: int) -> str:
    # A fresh prefix every call so each request pays FULL prefill (llama.cpp
    # caches identical prompt prefixes between calls; a repeated prefix would
    # report a near-zero prompt_eval and under-measure the prefill cost).
    words = [
        "alpha",
        "bravo",
        "delta",
        "echo",
        "gamma",
        "kilo",
        "lima",
        "nova",
        "sierra",
        "tango",
        "umbra",
        "victor",
        "yankee",
        "zephyr",
    ]
    out: list[str] = []
    total = 0
    while total < n_chars:
        w = words[int(rng.integers(0, len(words)))] + str(int(rng.integers(0, 9999)))
        out.append(w)
        total += len(w) + 1
    return " ".join(out)[:n_chars]


def _ollama_generate(model: str, prompt: str, num_predict: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.0},
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def measure_ollama(model: str, reps: int) -> dict:
    """Measure real prefill and decode costs on the M2 from llama.cpp's own
    timing fields. Returns the derived engine constants + the raw measurements.
    """
    rng = np.random.default_rng(0)
    prompt_tok: list[int] = []
    prefill_ms: list[float] = []  # prompt_eval_duration (ms)
    decode_counts: list[float] = []  # eval_count (tokens generated)
    decode_ms: list[float] = []  # eval_duration (ms)
    overhead_ms: list[float] = []  # total - prompt_eval - eval (ms)
    samples: list[dict] = []

    for k in DECODE_TOKEN_GRID:
        for _ in range(reps):
            prompt = _unique_prefix(rng, PREFIX_CHARS)
            r = _ollama_generate(model, prompt, k)
            # llama.cpp reports durations in nanoseconds.
            pe_n = float(r.get("prompt_eval_count", 0))
            pe_ms = float(r.get("prompt_eval_duration", 0)) / 1e6
            ev_n = float(r.get("eval_count", 0))
            ev_ms = float(r.get("eval_duration", 0)) / 1e6
            tot_ms = float(r.get("total_duration", 0)) / 1e6
            if pe_n <= 0 or ev_n <= 0:
                continue  # skip a degenerate (fully cached / empty) sample
            prompt_tok.append(pe_n)
            prefill_ms.append(pe_ms)
            decode_counts.append(ev_n)
            decode_ms.append(ev_ms)
            overhead_ms.append(max(0.0, tot_ms - pe_ms - ev_ms))
            samples.append(
                {
                    "prompt_tokens": pe_n,
                    "prompt_eval_ms": round(pe_ms, 2),
                    "output_tokens": ev_n,
                    "eval_ms": round(ev_ms, 2),
                    "total_ms": round(tot_ms, 2),
                }
            )

    if len(decode_counts) < 4:
        raise SystemExit(
            "Ollama returned too few usable samples — is the daemon healthy and "
            "the model pulled? (Each /api/generate must report prompt_eval/eval.)"
        )

    # Decode cost per output token: slope of eval_duration vs tokens generated.
    decode_ms_per_token, decode_intercept, r2 = _linfit(decode_counts, decode_ms)
    decode_ms_per_token = max(0.0, decode_ms_per_token)
    # Prefill cost per prompt token: direct ratio, averaged.
    prefill_ms_per_token = float(np.mean(np.asarray(prefill_ms) / np.asarray(prompt_tok)))
    mean_prompt_tokens = float(np.mean(prompt_tok))

    # --- map measured rates onto the engine's three constants (consistent ms) ---
    engine_prefill_ms = prefill_ms_per_token * mean_prompt_tokens  # cost a hit saves
    engine_beta_ms = decode_ms_per_token * NOMINAL_OUTPUT_TOKENS  # one item's decode (serial)
    engine_alpha_ms = float(np.mean(overhead_ms)) + max(0.0, decode_intercept)

    return {
        "alpha_ms": round(engine_alpha_ms, 3),
        "beta_ms": round(engine_beta_ms, 3),
        "prefill_ms": round(engine_prefill_ms, 3),
        "r2": round(r2, 4),
        "source": f"ollama:{model}",
        "measured": True,
        "decode_ms_per_token": round(decode_ms_per_token, 4),
        "prefill_ms_per_token": round(prefill_ms_per_token, 4),
        "mean_prompt_tokens": round(mean_prompt_tokens, 1),
        "nominal_output_tokens": NOMINAL_OUTPUT_TOKENS,
        "n_samples": len(samples),
        "samples": samples,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Measured on a single-slot Ollama daemon (no real multi-sequence "
            "batching). prefill_ms and decode rate are real M2 measurements from "
            "llama.cpp timing fields; beta_ms is the SERIAL per-item decode cost "
            f"at {NOMINAL_OUTPUT_TOKENS} output tokens. The batching speedup (beta "
            "shrinking as batch size grows) is NOT observable here and is the "
            "vLLM/multi-GPU validation step."
        ),
    }


# --------------------------------------------------------------------------- #
# Synthetic path: known ground truth + noise (clearly labelled)
# --------------------------------------------------------------------------- #
def measure_synthetic(
    alpha_true: float = 18.0,
    beta_true: float = 7.5,
    prefill_true: float = 160.0,
    reps: int = 6,
    sigma: float = 0.10,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    bsizes = (1, 4, 8, 16)
    xs: list[float] = []
    ys: list[float] = []
    samples: list[list[float]] = []
    for b in bsizes:
        for _ in range(reps):
            base = alpha_true + beta_true * b
            noisy = float(base * np.exp(rng.normal(-0.5 * sigma**2, sigma)))
            xs.append(b)
            ys.append(noisy)
            samples.append([b, noisy])
    beta, alpha, r2 = _linfit(xs, ys)
    return {
        "alpha_ms": round(max(alpha, 0.0), 3),
        "beta_ms": round(max(beta, 0.0), 3),
        "prefill_ms": round(prefill_true, 3),
        "r2": round(r2, 4),
        "source": "synthetic",
        "measured": False,
        "n_samples": len(samples),
        "samples": samples,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "Synthetic ground truth (no hardware). Sets a realistic prefill:decode "
            "RATIO so the frontier SHAPE is right; absolute ms are illustrative."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate engine latency constants.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--ollama", action="store_true", help="measure on a live Ollama daemon")
    mode.add_argument(
        "--synthetic", action="store_true", help="synthetic ground truth (default; no deps)"
    )
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    if args.ollama and not args.synthetic:
        print(
            f"Calibrating on Ollama model={args.model} (single-slot; "
            f"measuring real prefill + decode) ..."
        )
        out = measure_ollama(args.model, args.reps)
    else:
        print("Calibrating against SYNTHETIC ground truth (no hardware).")
        out = measure_synthetic(reps=args.reps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"  alpha_ms={out['alpha_ms']:.2f}  beta_ms={out['beta_ms']:.2f}  "
        f"prefill_ms={out['prefill_ms']:.2f}  R²={out['r2']:.4f}  "
        f"(source={out['source']})"
    )
    if out.get("measured"):
        print(
            f"  [measured] decode={out['decode_ms_per_token']:.2f} ms/tok, "
            f"prefill={out['prefill_ms_per_token']:.2f} ms/tok over "
            f"~{out['mean_prompt_tokens']:.0f} prompt tokens"
        )
    print(f"  wrote {CALIB_PATH}")


if __name__ == "__main__":
    main()
