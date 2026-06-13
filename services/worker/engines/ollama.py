"""OllamaEngine — Metal-accelerated LLM inference on a Mac via a local Ollama
daemon (DESIGN.md §5.3).

This is the easiest path to *real streaming tokens* and is what bench/calibrate.py
fits ``alpha``/``beta`` against. Important caveat from the design doc: **Ollama
hides its KV cache**, so this engine is for general serving and latency
calibration — it CANNOT observe or exploit prefix-cache state. The cache-locality
result is therefore measured on CacheAwareMockEngine and confirmed later on vLLM,
never here.

NEEDS a running Ollama daemon (``ollama serve`` + ``ollama pull qwen2.5:0.5b``)
and ``httpx``. Excluded from the default import path; imported lazily by the
harness when RELAY_ENGINE=ollama.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from relay_core.types import EngineStats, InferItem, ResultItem


class OllamaEngine:
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str = "qwen2.5:0.5b") -> None:
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model
        self._loaded: tuple[str, ...] = ()

    async def load(self, model: str) -> None:
        # Ollama loads on first use; a tiny warm-up request pre-pulls into VRAM.
        import httpx  # type: ignore

        self.model = model
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as c:
            await c.post("/api/generate", json={"model": model, "prompt": "", "stream": False})
        self._loaded = (model,)

    async def infer(self, batch: list[InferItem]) -> AsyncIterator[ResultItem]:
        # Ollama has no batch endpoint; Relay's batching value is shown on the
        # mock/MPS engines (no competing batcher). Here we issue the items
        # concurrently and stream each one's tokens back.
        import httpx  # type: ignore

        bid = batch[0].request_id if batch else ""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=300.0) as c:
            for item in batch:
                started = time.monotonic()
                async with c.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": self.model,
                        "prompt": item.input,
                        "stream": item.params.stream,
                        "options": {"num_predict": item.params.max_tokens},
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        # Each line is a JSON chunk: {"response": "...", "done": bool}
                        import json

                        chunk = json.loads(line)
                        done = bool(chunk.get("done"))
                        yield ResultItem(
                            request_id=item.request_id,
                            batch_id=bid,
                            token=chunk.get("response", "") if not done else None,
                            output=None,
                            final=done,
                            inference_ms=(time.monotonic() - started) * 1000.0,
                            batch_size=len(batch),
                            cache_hit=False,  # cache state is hidden by Ollama
                        )

    def stats(self) -> EngineStats:
        return EngineStats(loaded_models=self._loaded)
