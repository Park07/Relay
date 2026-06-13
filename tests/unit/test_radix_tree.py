"""Tests for the radix prefix tree (longest-prefix routing substrate).

These pin the properties the routing depends on: longest-match wins, partial
prefixes get credit single-hash misses, ties break to warmest, and LRU eviction
removes a worker from exactly the right nodes (reference-counted, so shared stems
survive). Correctness here is load-bearing — a wrong match silently degrades
cache reuse without crashing, so the tree is tested in isolation before it is
trusted in the router.
"""

from relay_core.radix import RadixPrefixTree


def blk(*xs: str) -> tuple[str, ...]:
    return tuple(xs)


def test_empty_tree_no_match():
    t = RadixPrefixTree()
    wid, depth = t.match_longest(blk("a", "b"))
    assert wid is None and depth == 0


def test_exact_path_full_match():
    t = RadixPrefixTree()
    t.insert(blk("a", "b", "c"), "w0")
    wid, depth = t.match_longest(blk("a", "b", "c"))
    assert wid == "w0" and depth == 3


def test_longest_prefix_beats_shorter():
    # w0 holds [a,b]; w1 holds [a,b,c,d]. A request for [a,b,c,e] shares 3 blocks
    # with w1 (a,b,c) but only 2 with w0 — longest match must pick w1.
    t = RadixPrefixTree()
    t.insert(blk("a", "b"), "w0")
    t.insert(blk("a", "b", "c", "d"), "w1")
    wid, depth = t.match_longest(blk("a", "b", "c", "e"))
    assert wid == "w1" and depth == 3


def test_partial_credit_single_hash_would_miss():
    # The whole point of radix: two prompts share a long stem but differ in the
    # last block. Single-first-block-hash co-locates them only if block 0 matches
    # AND treats them as identical; radix gives graded credit by shared depth.
    t = RadixPrefixTree()
    t.insert(blk("sys", "fewshot", "doc", "qA"), "w0")
    # New request shares the first 3 blocks, diverges at block 4.
    wid, depth = t.match_longest(blk("sys", "fewshot", "doc", "qB"))
    assert wid == "w0" and depth == 3  # 3 of 4 blocks reusable, not 0 and not 4


def test_divergent_first_block_no_match():
    t = RadixPrefixTree()
    t.insert(blk("a", "b", "c"), "w0")
    wid, depth = t.match_longest(blk("x", "b", "c"))
    assert wid is None and depth == 0  # nothing shared from block 0 → ring fallback


def test_eligibility_filter_excludes_capped_worker():
    t = RadixPrefixTree()
    t.insert(blk("a", "b", "c"), "w0")
    # If w0 is ineligible (e.g. over its load cap), no match is returned.
    wid, depth = t.match_longest(blk("a", "b", "c"), is_eligible=lambda w: w != "w0")
    assert wid is None and depth == 0


def test_tie_breaks_to_most_recently_used():
    t = RadixPrefixTree()
    t.insert(blk("a", "b"), "w0")
    t.insert(blk("a", "b"), "w1")  # w1 inserted later → warmer at equal depth
    wid, depth = t.match_longest(blk("a", "b"))
    assert depth == 2 and wid == "w1"
    # Re-serving w0 refreshes its recency; now w0 should win the tie.
    t.insert(blk("a", "b"), "w0")
    wid2, _ = t.match_longest(blk("a", "b"))
    assert wid2 == "w0"


def test_owners_recorded_along_whole_path():
    t = RadixPrefixTree()
    t.insert(blk("a", "b", "c"), "w0")
    assert t.owners_of(blk("a")) == {"w0"}
    assert t.owners_of(blk("a", "b")) == {"w0"}
    assert t.owners_of(blk("a", "b", "c")) == {"w0"}


def test_lru_eviction_drops_oldest_path():
    t = RadixPrefixTree(capacity=2)
    t.insert(blk("a"), "w0")
    t.insert(blk("b"), "w0")
    t.insert(blk("c"), "w0")  # exceeds capacity 2 → "a" (oldest) evicted
    assert t.path_count("w0") == 2
    assert t.owners_of(blk("a")) == set()      # evicted
    assert t.owners_of(blk("b")) == {"w0"}
    assert t.owners_of(blk("c")) == {"w0"}


def test_eviction_refcount_keeps_shared_stem():
    # w0 owns [a,b] and [a,c]; both pass through node "a". Evicting [a,b] must NOT
    # remove w0 from "a" (still reached via [a,c]) — refcounting guards this.
    t = RadixPrefixTree(capacity=2)
    t.insert(blk("a", "b"), "w0")
    t.insert(blk("a", "c"), "w0")
    t.insert(blk("z"), "w0")  # capacity 2 → oldest path [a,b] evicted
    assert t.owners_of(blk("a", "b")) == set()       # the evicted leaf is gone
    assert t.owners_of(blk("a")) == {"w0"}           # stem survives via [a,c]
    assert t.owners_of(blk("a", "c")) == {"w0"}


def test_reserving_does_not_inflate_path_count():
    t = RadixPrefixTree(capacity=2)
    t.insert(blk("a"), "w0")
    t.insert(blk("a"), "w0")  # same path again
    t.insert(blk("a"), "w0")
    assert t.path_count("w0") == 1  # re-serve refreshes, doesn't add an entry


def test_two_workers_independent_capacity():
    t = RadixPrefixTree(capacity=1)
    t.insert(blk("a"), "w0")
    t.insert(blk("b"), "w1")
    t.insert(blk("c"), "w0")  # evicts w0's "a", leaves w1's "b" untouched
    assert t.owners_of(blk("a")) == set()
    assert t.owners_of(blk("b")) == {"w1"}
    assert t.owners_of(blk("c")) == {"w0"}


def test_pruning_removes_dead_nodes():
    t = RadixPrefixTree(capacity=1)
    t.insert(blk("a", "b", "c"), "w0")
    n_before = t.node_count()
    assert n_before == 3
    t.insert(blk("x"), "w0")  # evicts [a,b,c]; its now-ownerless nodes prune away
    # Only "x" should remain (a,b,c pruned since no owners, no children).
    assert t.owners_of(blk("a", "b", "c")) == set()
    assert t.node_count() == 1