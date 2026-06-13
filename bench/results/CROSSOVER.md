# Crossover: when prefix-aware routing beats round-robin

Per-token costs held fixed at their measured (ollama:qwen2.5:0.5b) values (**decode 30.4 ms/tok**, **prefill 1.32 ms/tok**); fixed overhead alpha = 30 ms; completion = 64 tokens; 10,000 requests/run; **10 seeds** per point. Only the **shared-prefix length** varies.

**Crossover prefix length: **≈ 530 ± 58 tokens** (95% CI, 10/10 seeds crossed).** Below it, decode dominates and round-robin's better balance wins; above it, prefill dominates and reusing the KV cache (prefix-affinity) wins.

- short prefix (128 tok): speedup **0.80 ± 0.02×** (round-robin wins).
- long prefix (8192 tok): speedup **1.17 ± 0.02×**.

Every figure is mean ± 95% CI across independent seeds (Student-t). The takeaway is the *condition*, not a single number: KV-cache-aware routing is worth its load-imbalance cost only once the shared prefix is long enough that prefill is the bottleneck — the RAG / shared-system-prompt regime. On short chat prompts it is a net loss, and a router should load-balance there.

Full per-length data with CIs: `crossover.csv`. Figure: `crossover.png`.
