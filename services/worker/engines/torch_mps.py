"""TorchMPSEngine — PyTorch with ``device="mps"`` for vision/embedding/small
transformer models on the Apple M-series GPU (DESIGN.md §5.3).

This is a real batching surface (no competing intra-worker batcher), so it is a
good place to demonstrate Relay's own batch-former value with actual GPU work.
The engine batches a list of items into one forward pass and reports coarse
GPU memory as the USE signal. ``gpu_util`` on Mac is only approximable (no
nvidia-smi); the design doc treats utilization as a CUDA-validated bonus.

NEEDS PyTorch with MPS available. Excluded from the default import path;
imported lazily by the harness when RELAY_ENGINE=torch-mps.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

from relay_core.types import EngineStats, InferItem, ResultItem


class TorchMPSEngine:
    name = "torch-mps"

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self._model = None
        self._tok = None
        self._device = None
        self._loaded: tuple[str, ...] = ()

    async def load(self, model: str) -> None:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_id = model or self.model_id
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
        self._loaded = (self.model_id,)

    async def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]:
        import torch  # type: ignore

        assert self._model is not None and self._tok is not None, "call load() first"
        bid = batch[0].request_id if batch else ""
        started = time.monotonic()
        inputs = self._tok(
            [it.input for it in batch], return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        # Mean-pool to a sentence embedding; serialize a short preview per item.
        emb = out.last_hidden_state.mean(dim=1)
        torch.mps.synchronize() if self._device == "mps" else None
        latency_ms = (time.monotonic() - started) * 1000.0
        for i, item in enumerate(batch):
            vec = emb[i].tolist()
            yield ResultItem(
                request_id=item.request_id,
                batch_id=bid,
                output=f"[emb dim={len(vec)}] {vec[:4]}...",
                final=True,
                inference_ms=latency_ms,
                batch_size=len(batch),
                cache_hit=False,
            )

    def stats(self) -> EngineStats:
        mem = 0.0
        try:
            import torch  # type: ignore

            if self._device == "mps":
                mem = torch.mps.current_allocated_memory() / 1e6
        except Exception:
            pass
        return EngineStats(loaded_models=self._loaded, mem_used_mb=mem)
