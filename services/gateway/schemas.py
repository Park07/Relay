"""Pydantic v2 schemas for the gateway REST surface (DESIGN.md §7.1).

These validate the only public contract. They are deliberately separate from the
transport-agnostic dataclasses in relay_core.types: the wire schema can evolve
(new optional fields, defaults) without touching the scheduler's internal types.
"""

from __future__ import annotations

from enum import StrEnum

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - allows import without pydantic installed
    # Minimal shim so the module imports in the bench-only environment. The live
    # gateway requires real pydantic (declared in pyproject).
    class BaseModel:  # type: ignore
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    def Field(default=None, **kw):  # type: ignore
        return default


class Mode(StrEnum):
    sync = "sync"
    async_ = "async"


class PriorityIn(StrEnum):
    high = "high"
    default = "default"


class InferParamsIn(BaseModel):
    max_tokens: int = Field(128, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False


class InferRequest(BaseModel):
    model: str
    input: str
    mode: Mode = Mode.sync
    priority: PriorityIn = PriorityIn.default
    params: InferParamsIn = Field(default_factory=InferParamsIn)


class InferSyncResponse(BaseModel):
    job_id: str
    model: str
    output: str | None = None
    cache_hit: bool = False
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


class InferAcceptedResponse(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued|running|done|error
    model: str | None = None
    output: str | None = None
    error: str | None = None
    cache_hit: bool | None = None
    queue_wait_ms: float | None = None
    inference_ms: float | None = None
    total_ms: float | None = None


class ModelInfo(BaseModel):
    model: str
    loaded_on: list[str] = Field(default_factory=list)  # worker ids


class ModelsResponse(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
