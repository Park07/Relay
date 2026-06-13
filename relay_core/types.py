"""Shared, transport-agnostic contracts for Relay.

These dataclasses are the in-process mirror of the protobuf wire messages in
``proto/relay/v1/worker.proto``. Keeping a plain-Python copy lets the scheduler,
the worker harness, and the benchmark harness share one set of types and lets the
*entire core algorithmic result* (router + engines + workload + frontier) run
in-process with no gRPC/Redis dependency — see DESIGN.md §13/§17.

When the gRPC stack is generated (``make proto``), the worker harness converts
between these and the generated ``relay.v1`` messages at the edge.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Request / result payloads  (mirror: InferItem, InferParams, ResultItem)
# --------------------------------------------------------------------------- #
class Priority(StrEnum):
    HIGH = "high"
    DEFAULT = "default"


@dataclass(slots=True)
class InferParams:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


@dataclass(slots=True)
class InferItem:
    """One unit of work. ``prefix_hash`` is what the router keys on."""

    request_id: str
    input: str
    params: InferParams = field(default_factory=InferParams)
    prefix_hash: str = ""
    # Bench-only bookkeeping (never crosses the wire): when the item was admitted.
    enqueue_ts: float = 0.0
    priority: Priority = Priority.DEFAULT

    def __post_init__(self) -> None:
        if not self.prefix_hash:
            self.prefix_hash = prefix_hash_of(self.input)


@dataclass(slots=True)
class ResultItem:
    request_id: str
    batch_id: str
    output: str | None = None
    token: str | None = None
    error: str | None = None
    final: bool = True
    # WorkerMetrics, inlined:
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0
    batch_size: int = 0
    gpu_util: float = 0.0
    cache_hit: bool = False


@dataclass(slots=True)
class BatchAssignment:
    batch_id: str
    model: str
    items: list[InferItem]


# --------------------------------------------------------------------------- #
# Worker view  (mirror: WorkerInfo, LeaseRequest; plus scheduler-side load state)
# --------------------------------------------------------------------------- #
@dataclass
class WorkerState:
    """The scheduler's mutable view of one worker.

    ``inflight`` and ``free_slots`` are the two numbers the router and batch
    former reason about; the worker advertises ``free_slots`` on every lease
    tick (ADR-2, pull-based leasing → backpressure).
    """

    worker_id: str
    engine: str
    models: tuple[str, ...]
    max_batch: int = 16
    max_concurrent_batches: int = 1
    inflight: int = 0  # batches currently executing

    @property
    def capacity(self) -> int:
        return self.max_concurrent_batches

    @property
    def free_slots(self) -> int:
        return max(0, self.max_concurrent_batches - self.inflight)

    def has_model(self, model: str) -> bool:
        return model in self.models


@dataclass(slots=True)
class EngineStats:
    gpu_util: float = 0.0
    mem_used_mb: float = 0.0
    loaded_models: tuple[str, ...] = ()
    # cache-aware engines expose their realized hit-rate so the harness can
    # measure locality directly rather than inferring it (DESIGN.md §14).
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


# --------------------------------------------------------------------------- #
# The one interface, four backends  (DESIGN.md §5.3 / ADR-7)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Engine(Protocol):
    name: str

    async def load(self, model: str) -> None: ...

    def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]:
        """Always yields ``ResultItem``s, streaming or not, so the harness need
        not special-case streaming vs non-streaming engines (ADR-7)."""
        ...

    def stats(self) -> EngineStats: ...


# --------------------------------------------------------------------------- #
# Prefix hashing — the key the whole depth feature turns on.
# --------------------------------------------------------------------------- #
# Real systems hash the *tokenized* prefix block (system prompt + few-shot +
# RAG doc) because the KV cache is keyed on token-block prefixes. Here we hash a
# stable string prefix; the workload generator (bench/workload.py) is what makes
# that prefix realistically *shared* across requests (Zipfian), which is the
# thing that makes the result non-trivial.
PREFIX_BLOCK_CHARS = 512


def prefix_hash_of(text: str, block_chars: int = PREFIX_BLOCK_CHARS) -> str:
    block = text[:block_chars]
    return hashlib.blake2b(block.encode("utf-8"), digest_size=16).hexdigest()
