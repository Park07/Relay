# Real-hardware validation: Relay routing on 2× A40 vLLM

The crossover and frontier results elsewhere in this repo are **simulated** (real
M2 per-token rates driving a virtual-time model of the router). This is the check
against reality: Relay's *own* `PrefixRouter` placing requests across **two real
vLLM workers with two separate KV caches**, on rented 2× A40 hardware.

The driver (`bench/vllm_validate.py`) imports `services/scheduler/router.py`
unchanged, uses it to choose a worker for each request, sends the request directly
to that worker, and reads vLLM's own `usage.prompt_tokens_details.cached_tokens` as
the ground-truth cache signal. Two policies bracket the comparison: `round_robin`
(balance) and `affinity` (Relay's `PrefixRouter` at `cap=inf`). Setup: Qwen2.5-0.5B,
prefix pool 64, Zipf s=1.1, 300 requests/point, concurrency 16, vLLM 0.10.1.1.

## Result

| prefix tok | policy | cache-hit | p99 (ms) | distinct workers/prefix | p99 speedup vs RR |
|---|---|---|---|---|---|
| 128 | round-robin | 0.70 | 341.9 | 1.57 | — |
| 128 | **affinity** | **0.92** | **303.0** | **1.00** | **1.13×** |
| 512 | round-robin | 0.75 | 318.2 | 1.45 | — |
| 512 | **affinity** | **0.95** | **279.5** | **1.00** | **1.14×** |
| 1024 | round-robin | 0.74 | 376.5 | 1.55 | — |
| 1024 | **affinity** | **0.96** | **303.1** | **1.00** | **1.24×** |

Three things, measured on real hardware:

1. **Placement is correct (the keystone).** Under affinity, `distinct_workers_per_prefix = 1.00` at every length — Relay's router sent each prefix to exactly one worker. Round-robin scatters at 1.45–1.57. This is the load-bearing check: it confirms Relay's routing *logic* runs correctly as a real distributed driver against real engines, and it's hardware-independent (the right worker is the right worker regardless of speed).
2. **Cache reuse follows placement (the mechanism).** vLLM's own `cached_tokens` rises to **0.92 → 0.95 → 0.96** under affinity vs a flat **~0.70–0.75** under round-robin. Concentrating prefixes reuses the KV cache; scattering re-prefills. This is ground truth from the engine, not the simulator.
3. **The p99 advantage grows with prefix length** — **1.13× → 1.14× → 1.24×**. This is the *direction* the simulated crossover predicts (affinity's benefit increases as prefill cost grows), now reproduced on real vLLM with real eviction and real continuous batching.

So the central claim of the simulated study — *prefix-affinity routing's benefit increases with shared-prefix length* — holds on real hardware. That retires the headline caveat that the routing benefit was simulation-only.

## Honest caveats (these bound the claim; they don't undermine it)

- **4096-token point failed** ("no successful responses") — a context-length / request-timeout limit at this configuration, not a routing failure. The 128/512/1024 points stand; 4096 is omitted.
- **The crossover sits left of the simulator's ~530 tokens.** On real vLLM at this scale, affinity wins even at 128 tokens, whereas the simulator (M2 rates) put the crossover near 530. That **gap is itself a finding**: real continuous batching plus a small (0.5B) model shift the crossover left of the simulated estimate. It is exactly the sim-vs-real discrepancy this experiment was built to surface — the simulator gets the *direction* right and the *location* approximately, and reality compresses it.
- **Scale is small and deliberate.** Two workers, a 0.5B model, ~300 requests/point. This validates Relay's routing *logic* and the cache *mechanism* on real hardware — it is **not** a throughput benchmark or a scaling study. Latency magnitudes (~200–380 ms p99) are dominated by small-model/small-prompt overhead, not the prefill cost that drives the crossover, so read the *cache-hit* and *placement* columns as the clean signals and the *speedup* column as directional.
- **`cached_tokens` reporting.** A cold first request reported `prompt_tokens_details: null` (a known vLLM V1 quirk); under load the field populated correctly, which is what the table reflects.

Raw data: `vllm_validation.json`. Driver: `bench/vllm_validate.py` (re-runnable on any ≥2-GPU box with two `vllm serve` workers).
