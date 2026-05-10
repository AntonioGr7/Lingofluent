from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Sequence


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ImagePart:
    url: str | None = None
    data: bytes | None = None
    mime: str = "image/png"
    type: Literal["image"] = "image"

    def __post_init__(self) -> None:
        if self.url is None and self.data is None:
            raise ValueError("ImagePart requires either url or data")


ContentPart = TextPart | ImagePart
Content = str | Sequence[ContentPart]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: Content = ""
    name: str | None = None
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    tool_call_id: str | None = None


# ---------------------------------------------------------------------------
# Options & results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatOptions:
    model: str | None = None
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int | None = None
    stop: Sequence[str] = field(default_factory=tuple)
    seed: int | None = None
    response_format: Literal["text", "json_object"] = "text"
    tools: Sequence[Tool] = field(default_factory=tuple)
    tool_choice: str = "auto"
    timeout: float | None = 120.0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatResult:
    text: str
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None
    backend: str = ""
    raw: dict | None = None


@dataclass(frozen=True)
class ChatChunk:
    delta_text: str = ""
    delta_tool_call: ToolCall | None = None
    finish_reason: str | None = None
    raw: dict | None = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for all LLM backend errors."""

class LLMAuthError(LLMError):
    """401 / 403 or equivalent."""

class LLMTimeoutError(LLMError):
    """Request exceeded ChatOptions.timeout."""

class LLMUnavailableError(LLMError):
    """Network failure or 5xx."""

class LLMRateLimitError(LLMError):
    """429 — too many requests."""

class LLMInputError(LLMError):
    """400, missing capability, or bad request shape."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMBackend(abc.ABC):
    name: str = "abstract"

    supports_streaming: bool = False
    supports_vision: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False

    # ---- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "LLMBackend":
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def startup(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    # ---- public API --------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[Message],
        options: ChatOptions | None = None,
    ) -> ChatResult:
        opts = options or ChatOptions()
        self._validate_request(messages, opts)
        coro = self._chat(messages, opts)
        if opts.timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=opts.timeout)
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"{self.name} chat exceeded {opts.timeout}s") from e

    async def chat_stream(
        self,
        messages: Sequence[Message],
        options: ChatOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        if not self.supports_streaming:
            raise NotImplementedError(f"{self.name} does not support streaming")
        opts = options or ChatOptions()
        self._validate_request(messages, opts)
        async for chunk in self._chat_stream(messages, opts):
            yield chunk

    async def chat_many(
        self,
        batch: Sequence[Sequence[Message]],
        options: ChatOptions | None = None,
        max_concurrency: int = 4,
    ) -> list[ChatResult]:
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(msgs: Sequence[Message]) -> ChatResult:
            async with sem:
                return await self.chat(msgs, options)

        return list(await asyncio.gather(*(_one(m) for m in batch)))

    # ---- subclass contract -------------------------------------------------

    @abc.abstractmethod
    async def _chat(self, messages: Sequence[Message], options: ChatOptions) -> ChatResult: ...

    @abc.abstractmethod
    async def health(self) -> bool: ...

    async def _chat_stream(
        self,
        messages: Sequence[Message],
        options: ChatOptions,
    ) -> AsyncIterator[ChatChunk]:
        raise NotImplementedError
        if False:
            yield ChatChunk()

    # ---- helpers -----------------------------------------------------------

    def _validate_request(self, messages: Sequence[Message], options: ChatOptions) -> None:
        if not messages:
            raise LLMInputError("messages must not be empty")

        for msg in messages:
            if msg.role == "tool" and not msg.tool_call_id:
                raise LLMInputError("Message with role='tool' requires tool_call_id")
            if not isinstance(msg.content, str):
                for part in msg.content:
                    if isinstance(part, ImagePart) and not self.supports_vision:
                        raise LLMInputError(f"{self.name} does not support vision inputs")

        if options.tools and not self.supports_tools:
            raise LLMInputError(f"{self.name} does not support tool calling")

        if options.response_format == "json_object" and not self.supports_json_mode:
            raise LLMInputError(f"{self.name} does not support json_object response format")
