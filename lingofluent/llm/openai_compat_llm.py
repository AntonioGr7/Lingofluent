from __future__ import annotations

import base64
import json
from typing import AsyncIterator, Sequence

import aiohttp

from lingofluent.llm.llm_base import (
    ChatChunk,
    ChatOptions,
    ChatResult,
    ImagePart,
    LLMAuthError,
    LLMBackend,
    LLMInputError,
    LLMRateLimitError,
    LLMUnavailableError,
    Message,
    TextPart,
    TokenUsage,
    ToolCall,
)


def _safe_json_load(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"_raw": s}


def _serialize_content(content) -> str | list:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.url is not None:
                url = part.url
            else:
                b64 = base64.b64encode(part.data).decode()
                url = f"data:{part.mime};base64,{b64}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _serialize_message(msg: Message) -> dict:
    d: dict = {"role": msg.role, "content": _serialize_content(msg.content)}
    if msg.name:
        d["name"] = msg.name
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in msg.tool_calls
        ]
    return d


def _parse_tool_calls(raw_calls: list) -> list[ToolCall]:
    return [
        ToolCall(
            id=tc.get("id", ""),
            name=tc.get("function", {}).get("name", ""),
            arguments=_safe_json_load(tc.get("function", {}).get("arguments", "{}")),
        )
        for tc in raw_calls
    ]


def _build_body(messages: Sequence[Message], options: ChatOptions, default_model: str | None) -> dict:
    body: dict = {"messages": [_serialize_message(m) for m in messages]}
    model = options.model or default_model
    if model:
        body["model"] = model
    if options.temperature != 0.7:
        body["temperature"] = options.temperature
    if options.top_p is not None:
        body["top_p"] = options.top_p
    if options.max_tokens is not None:
        body["max_tokens"] = options.max_tokens
    if options.stop:
        body["stop"] = list(options.stop)
    if options.seed is not None:
        body["seed"] = options.seed
    if options.response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    if options.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in options.tools
        ]
        body["tool_choice"] = options.tool_choice
    body.update(options.extra)
    return body


def _raise_for_status(status: int, payload: dict) -> None:
    if status in (401, 403):
        raise LLMAuthError(f"auth failed: {status}")
    if status == 429:
        raise LLMRateLimitError("rate limited")
    if status == 400:
        msg = payload.get("error", {}).get("message", f"bad request: {status}")
        raise LLMInputError(msg)
    if status >= 500:
        raise LLMUnavailableError(f"server error: {status}")


class OpenAICompatLLM(LLMBackend):
    """Async LLM backend for any server that speaks OpenAI /v1/chat/completions."""

    name = "openai-compat"
    supports_streaming = True
    supports_vision = True
    supports_tools = True
    supports_json_mode = True

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        default_model: str | None = None,
        extra_headers: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._session: aiohttp.ClientSession | None = None

    async def startup(self) -> None:
        self._session = aiohttp.ClientSession()

    async def aclose(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def health(self) -> bool:
        assert self._session is not None
        try:
            async with self._session.get(f"{self.base_url}/v1/models") as r:
                return r.status == 200
        except aiohttp.ClientError:
            return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _chat(self, messages: Sequence[Message], options: ChatOptions) -> ChatResult:
        assert self._session is not None, "call startup() or use async with"
        body = _build_body(messages, options, self.default_model)
        try:
            async with self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=self._headers(),
            ) as resp:
                payload = await resp.json(content_type=None)
                _raise_for_status(resp.status, payload)
                resp.raise_for_status()
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(str(e)) from e

        choice = payload["choices"][0]
        msg = choice["message"]
        usage_raw = payload.get("usage", {})
        return ChatResult(
            text=msg.get("content") or "",
            tool_calls=_parse_tool_calls(msg.get("tool_calls") or []),
            finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            model=payload.get("model"),
            backend=self.name,
            raw=payload,
        )

    async def _chat_stream(
        self,
        messages: Sequence[Message],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]:
        assert self._session is not None, "call startup() or use async with"
        body = _build_body(messages, options, self.default_model)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        try:
            async with self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    payload = await resp.json(content_type=None)
                    _raise_for_status(resp.status, payload)
                    resp.raise_for_status()

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")
                    delta_text = delta.get("content") or ""
                    if delta_text or finish_reason:
                        yield ChatChunk(
                            delta_text=delta_text,
                            finish_reason=finish_reason,
                            raw=chunk,
                        )
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(str(e)) from e
