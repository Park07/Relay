# Longest-prefix (radix) routing: when block-granular affinity beats first-block hashing

`PrefixRouter` (the original §8.2 router) keys on a hash of the prompt's **first
block**. That is adequate when distinct prefixes are independent islands, but it
has a sharp failure mode in the regime KV-cache routing exists for: a **broadly
shared stem**. If one system prompt (or RAG preamble) is block 0 of almost all
traffic, first-block hashing maps all of it to a single ring position → one
worker. Bounded load then spills the overflow onto *cold* workers that re-prefill
from scratch, so the shared cache barely helps and load piles unevenly.

Longest-prefix routing (SGLang's RadixAttention, the vLLM cache-aware policy)
instead places each request on the worker holding the longest *block* prefix it
shares. Traffic is co-located by its deepest common stem, so each worker owns a
subtree of the prefix space, keeps that subtree warm, and load spreads across the
branches rather than piling on one owner.

## What was built

- `relay_core/radix.py` — `RadixPrefixTree`: a radix tree over block sequences with
  longest-prefix worker lookup and **reference-counted per-worker LRU eviction**
  (a worker reached via a longer retained path survives eviction of a shorter
  one). 13 unit tests pin longest-match, partial-prefix credit, tie-break-to-MRU,
  refcounted eviction, and node pruning.
- `services/scheduler/radix_router.py` — `RadixPrefixRouter`: same seam as
  `PrefixRouter` (`pick` / `load_fn` / `admit_fn` / `add_worker`), longest-match
  **among workers under the load cap**, consistent-hash ring fallback for cold
  prefixes. 8 unit tests pin warm-prefix placement, pure-affinity concentration,
  bounded-load spill, deterministic cold fallback, and eviction. Bounded-load
  consistent hashing is preserved — the cap logic lives in the router, the tree
  holds no load state (mirroring `BoundedLoadConsistentHashRing`).

96 tests pass (75 prior + 21 here).

## The finding (`bench/radix_compare.py`)

A self-contained harness (touches none of the simulator core) drives **both real
routers** over a branching block-workload — one shared root block across all
traffic, Zipfian branches, divergent tails — with an inline per-worker block-level
LRU cache (prefill charged per *missing* block; latency = queue-wait + service).
4 workers × 2 concurrent, `cap_factor = 1.5`, 6000 requests.

**Routing quality (load-independent):**

| policy | block-reuse | imbalance (max/mean load) |
|---|---|---|
| single-block-hash | 0.833 | 2.23 |
| **radix** | **0.897** | **1.37** |

Radix reuses ~6 points more cache and balances ~1.6× better — it pins no single
worker with the shared-stem traffic.

**p99 latency vs offered load** — where the imbalance actually costs something:

| offered rps | single-hash p99 (ms) | radix p99 (ms) | ratio |
|---|---|---|---|
| 40 | 186 | 186 | 1.0× |
| 70 | 226 | 186 | 1.2× |
| 100 | 706 | 321 | 2.2× |
| 130 | 13889 | 723 | **19.2×** |
| 160 | 24338 | 4692 | 5.2× |
| 200 | 32397 | 11970 | 2.7× |

The story is **a crossover in load, not in prefix length**: at low load p99 is
*identical* (no worker is saturated, so latency is just service time and routing
balance is invisible). As load rises, single-hash's hottest worker — pinned by the
common block 0 — saturates first (around 100 rps) and its queue explodes, while
radix's flatter distribution stays stable. The gap peaks where single-hash has
tipped over but radix has not (~130 rps, 19×); past that, radix saturates too, so
the *ratio* shrinks back even though single-hash remains worse in absolute terms.

## Honest scope

- **The 19× is the peak of a curve, not a headline constant.** It is the
  best-case separation, at the one load where single-hash is saturated and radix
  is not. The defensible claim is directional: radix sustains meaningfully higher
  load at equal p99 (here, roughly 100 → 130+ rps before p99 degrades) and reuses
  more cache — not "19× faster".
- **This is a cost model, not measured hardware.** Block-prefill is a fixed
  per-block cost with a simple FIFO/concurrency queue; there is no continuous-
  batching plateau, no real KV eviction timing, no network. It demonstrates the
  *mechanism* (first-block hashing pins broadly-shared stems; longest-prefix
  routing spreads them) on a workload built to exercise exactly that. It does not
  claim these millisecond values transfer to vLLM.
- **The regime is chosen to expose the difference.** Where prefixes are *not*
  broadly shared (independent islands), first-block hashing and radix behave
  similarly — radix's advantage is specific to deep/branching shared structure,
  which is also where it matters in practice (shared system prompts, few-shot,
  RAG preambles).
- **Effect on the crossover study.** Block-granular partial reuse changes the
  prefill-savings curve, so any crossover-vs-prefix-length number should be
  recomputed on the radix path before being treated as current; the ~530-token
  M2 figure predates this and is first-block-hash-specific.

Reproduce: `PYTHONPATH=. python bench/radix_compare.py` (CPU-only, ~seconds).