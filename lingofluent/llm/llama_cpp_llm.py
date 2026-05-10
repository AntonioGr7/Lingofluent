from __future__ import annotations

import os

import aiohttp

from lingofluent.llm.openai_compat_llm import OpenAICompatLLM


class LlamaCppLLM(OpenAICompatLLM):
    """LLM backend targeting a local llama.cpp server."""

    name = "llamacpp"

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        **kw,
    ) -> None:
        super().__init__(
            base_url=base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8080"),
            api_key=None,
            default_model=default_model,
            **kw,
        )

    async def health(self) -> bool:
        assert self._session is not None
        try:
            async with self._session.get(f"{self.base_url}/health") as r:
                return r.status == 200
        except aiohttp.ClientError:
            return False
