from __future__ import annotations

import os

from lingofluent.llm.openai_compat_llm import OpenAICompatLLM

_OPENAI_BASE_URL = "https://api.openai.com/"


class OpenAILLM(OpenAICompatLLM):
    """LLM backend for OpenAI or any private OpenAI-compatible server.

    URL resolution (in order):
      1. explicit base_url argument
      2. LLM_BASE_URL env var  (use this for private servers)
      3. https://api.openai.com/v1  (OpenAI default)

    API key resolution (in order):
      1. explicit api_key argument
      2. OPENAI_API_KEY env var
    """

    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str = "gpt-4o",
        **kw,
    ) -> None:
        resolved_url = base_url or os.environ.get("LLM_BASE_URL") or _OPENAI_BASE_URL
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAILLM requires an API key — set OPENAI_API_KEY or pass api_key="
            )
        super().__init__(
            base_url=resolved_url,
            api_key=resolved_key,
            default_model=default_model,
            **kw,
        )
