"""A radix tree over prefix *block* sequences, for longest-prefix routing.

DESIGN.md §8.2 follow-up. The original `PrefixRouter` hashes only the first block
of a prompt, so affinity is all-or-nothing at the block boundary: two requests
that share a 500-token stem but differ in the last block get *zero* shared-cache
credit if that difference lands in the hashed block, and conversely two requests
that merely share a first block are treated as fully co-located even if they
diverge immediately after. Production routers (SGLang's RadixAttention, the vLLM
cache-aware policy) instead route by the *longest matching prefix*: a request goes
to the worker that already holds the most of its KV cache.

This module is the substrate for that. A prompt is split into an ordered tuple of
opaque block hashes (tokenization lives in the caller). The tree maps block paths
to the workers that have served — and therefore cache — a prompt passing through
that path. `match_longest` returns the eligible worker sharing the deepest block
prefix with a new request; that is the worker with the most reusable cache.

Finite KV cache is modelled with a per-worker LRU bound (`capacity` distinct
full-paths per worker). When a worker exceeds it, its least-recently-used path is
evicted and the worker is removed from the nodes along that path (reference
counted, so a node a worker still reaches via a *longer* retained path is kept).
This is what lets the eviction-pressure regime be studied honestly rather than
assuming an infinite cache.

The tree carries no live-load state — bounded-load spillover lives in the router
(`services/scheduler/radix_router.py`), exactly as the consistent-hash ring keeps
load out of `BoundedLoadConsistentHashRing`. That keeps this structure
deterministic and unit-testable in isolation.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


@dataclass
class _Node:
    # worker_id -> [refcount, last_used_seq]; refcount = how many retained paths
    # owned by this worker pass through this node (so eviction of one path doesn't
    # drop a worker still reachable here via a longer path).
    owners: dict[str, list[int]] = field(default_factory=dict)
    children: dict[str, "_Node"] = field(default_factory=dict)


class RadixPrefixTree:
    """Longest-prefix worker lookup over block sequences, with per-worker LRU.

    ``capacity`` is the max number of distinct full block-paths a single worker
    may retain (a proxy for its KV-cache budget in blocks-worth-of-prefixes).
    ``capacity <= 0`` means unbounded (no eviction).
    """

    def __init__(self, capacity: int = 0) -> None:
        self.root = _Node()
        self.capacity = capacity
        self._seq = itertools.count()
        # worker_id -> OrderedDict[path_tuple -> None], LRU order (oldest first)
        self._paths: dict[str, "OrderedDict[tuple[str, ...], None]"] = {}

    # -- insert ------------------------------------------------------------ #
    def insert(self, blocks: tuple[str, ...], worker_id: str) -> None:
        """Record that ``worker_id`` now caches a prompt whose block path is
        ``blocks`` (and, by prefix property, every prefix of it)."""
        if not blocks:
            return
        seq = next(self._seq)
        owned = self._paths.setdefault(worker_id, OrderedDict())

        if blocks in owned:
            # Re-served: refresh recency, do not double-count refs.
            owned.move_to_end(blocks)
            self._touch_path(blocks, worker_id, seq)
        else:
            self._add_path(blocks, worker_id, seq)
            owned[blocks] = None
            if self.capacity > 0 and len(owned) > self.capacity:
                oldest, _ = owned.popitem(last=False)  # LRU evict
                self._remove_path(oldest, worker_id)

    def _add_path(self, blocks: tuple[str, ...], worker_id: str, seq: int) -> None:
        node = self.root
        for b in blocks:
            node = node.children.setdefault(b, _Node())
            ref = node.owners.get(worker_id)
            if ref is None:
                node.owners[worker_id] = [1, seq]
            else:
                ref[0] += 1
                ref[1] = seq

    def _touch_path(self, blocks: tuple[str, ...], worker_id: str, seq: int) -> None:
        node = self.root
        for b in blocks:
            node = node.children[b]
            ref = node.owners.get(worker_id)
            if ref is not None:
                ref[1] = seq

    def _remove_path(self, blocks: tuple[str, ...], worker_id: str) -> None:
        # Walk down, decrement refcounts; prune nodes that become empty.
        path: list[tuple[_Node, str, _Node]] = []
        node = self.root
        for b in blocks:
            child = node.children.get(b)
            if child is None:
                return  # path already gone (shouldn't happen)
            path.append((node, b, child))
            node = child
        for parent, b, child in reversed(path):
            ref = child.owners.get(worker_id)
            if ref is not None:
                ref[0] -= 1
                if ref[0] <= 0:
                    del child.owners[worker_id]
            if not child.owners and not child.children:
                parent.children.pop(b, None)

    # -- query ------------------------------------------------------------- #
    def match_longest(
        self, blocks: tuple[str, ...],
        is_eligible: Optional[Callable[[str], bool]] = None,
    ) -> tuple[Optional[str], int]:
        """Return ``(worker_id, matched_blocks)`` for the eligible worker sharing
        the deepest block prefix with ``blocks``. ``matched_blocks`` is how many
        leading blocks that worker already caches (0 ⇒ no eligible match → caller
        should fall back to the ring). Ties at equal depth break to the
        most-recently-used owner (warmest cache)."""
        node = self.root
        best_wid: Optional[str] = None
        best_depth = 0
        for depth, b in enumerate(blocks, start=1):
            child = node.children.get(b)
            if child is None:
                break
            # Among this node's owners, the deepest eligible one seen so far wins;
            # since we descend, any eligible owner here is at least `depth` deep.
            cand_wid, cand_seq = None, -1
            for wid, (_ref, last) in child.owners.items():
                if is_eligible is not None and not is_eligible(wid):
                    continue
                if last > cand_seq:
                    cand_wid, cand_seq = wid, last
            if cand_wid is not None:
                best_wid, best_depth = cand_wid, depth
            node = child
        return best_wid, best_depth

    # -- introspection (tests / metrics) ----------------------------------- #
    def owners_of(self, blocks: tuple[str, ...]) -> set[str]:
        node = self.root
        for b in blocks:
            node = node.children.get(b)
            if node is None:
                return set()
        return set(node.owners)

    def path_count(self, worker_id: str) -> int:
        return len(self._paths.get(worker_id, ()))

    def node_count(self) -> int:
        total = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            total += len(n.children)
            stack.extend(n.children.values())
        return total