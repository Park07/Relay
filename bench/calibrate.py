"""Calibrate the MockEngine's ``alpha`` / ``beta`` from real hardware
(DESIGN.md §5.3, ADR-11).

Procedure: run a small real model (e.g. ``qwen2.5:0.5b`` on Ollama) at batch
sizes b ∈ {1,4,8,16}, a handful of reps each, record per-batch wall-clock
latency, and least-squares fit

    latency(b) = alpha + beta * b.

The fitted constants are written to ``bench/results/calibration.json`` and loaded
by the engines, so every downstream sweep is anchored to measured hardware and the
mock is a *disclosed* simulation rather than a guess.

Two modes:
  * ``--ollama``    : real fit against a running Ollama daemon (Metal-accelerated
                      on a Mac). Requires ``ollama serve`` + a pulled model.
  * ``--synthetic`` : generate plausible samples from a known ground truth so the
                      whole pipeline is reproducible with zero external deps. The
                      JSON records ``"source": "synthetic"`` so this is never
                      mistaken for a hardware measurement.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CALIB_PATH = RESULTS_DIR / "calibration.json"
BATCH_SIZES = (1, 4, 8, 16)
OLLAMA_URL = "http://localhost:11434/api/generate"


def fit_alpha_beta(samples: list[tuple[int, float]]) -> dict:
    """Least-squares fit of latency = alpha + beta*b. Returns fit + R²."""
    b = np.array([s[0] for s in samples], dtype=np.float64)
    lat = np.array([s[1] for s in samples], dtype=np.float64)
    beta, alpha = np.polyfit(b, lat, 1)  # slope, intercept
    pred = alpha + beta * b
    ss_res = float(np.sum((lat - pred) ** 2))
    ss_tot = float(np.sum((lat - lat.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {
        "alpha_ms": float(max(alpha, 0.0)),
        "beta_ms": float(max(beta, 0.0)),
        "r2": float(r2),
    }


# --------------------------------------------------------------------------- #
# Real path: Ollama
# --------------------------------------------------------------------------- #
def _ollama_once(model: str, prompt: str, max_tokens: int) -> float:
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"num_predict": max_tokens}}
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def bench_ollama(model: str, reps: int, max_tokens: int) -> list[tuple[int, float]]:
    """For each batch size, fire b concurrent generations and record the
    makespan (proxy for per-batch latency). Ollama doesn't continuous-batch, so
    this characterizes the launch + marginal cost we want alpha/beta to capture.
    """
    prompt = "Summarize the following in one sentence: " + ("context " * 64)
    samples: list[tuple[int, float]] = []
    for b in BATCH_SIZES:
        for _ in range(reps):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=b) as ex:
                futs = [ex.submit(_ollama_once, model, prompt, max_tokens)
                        for _ in range(b)]
                concurrent.futures.wait(futs)
            samples.append((b, (time.perf_counter() - t0) * 1000.0))
    return samples


# --------------------------------------------------------------------------- #
# Synthetic path: known ground truth + noise (clearly labeled)
# --------------------------------------------------------------------------- #
def synthetic_samples(
    alpha_true: float = 18.0, beta_true: float = 7.5,
    reps: int = 6, sigma: float = 0.10, seed: int = 0,
) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[int, float]] = []
    for b in BATCH_SIZES:
        for _ in range(reps):
            base = alpha_true + beta_true * b
            samples.append((b, float(base * np.exp(rng.normal(-0.5 * sigma**2, sigma)))))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate MockEngine alpha/beta.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--ollama", action="store_true", help="fit against a live Ollama daemon")
    mode.add_argument("--synthetic", action="store_true",
                      help="fit against synthetic ground truth (default; no deps)")
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    use_ollama = args.ollama and not args.synthetic
    if use_ollama:
        print(f"Calibrating against Ollama model={args.model} ...")
        samples = bench_ollama(args.model, args.reps, args.max_tokens)
        source = f"ollama:{args.model}"
    else:
        print("Calibrating against SYNTHETIC ground truth (no hardware).")
        samples = synthetic_samples(reps=args.reps)
        source = "synthetic"

    fit = fit_alpha_beta(samples)
    out = {
        **fit,
        "source": source,
        "batch_sizes": list(BATCH_SIZES),
        "n_samples": len(samples),
        "samples": samples,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "alpha_ms = fixed per-batch cost, beta_ms = marginal per-item cost; "
            "latency(b)=alpha+beta*b. 'synthetic' source means no hardware was used."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(out, indent=2))
    print(f"  alpha_ms={fit['alpha_ms']:.2f}  beta_ms={fit['beta_ms']:.2f}  "
          f"R²={fit['r2']:.4f}  (source={source})")
    print(f"  wrote {CALIB_PATH}")


if __name__ == "__main__":
    main()
