"""
Abstract base class for async ASR backends.

Defines a stable contract that concrete backends (CrispASR, OpenAI Whisper,
Deepgram, AssemblyAI, a local faster-whisper instance, etc.) implement.
Callers code against `ASRBackend` and can swap implementations freely.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Literal, Sequence


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

ResponseFormat = Literal["json", "verbose_json", "text", "srt", "vtt"]
AudioSource = str | Path | bytes | BinaryIO


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single timestamped chunk of transcribed audio."""
    start: float                  # seconds
    end: float                    # seconds
    text: str
    speaker: str | None = None    # populated by diarizing backends
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Normalized result returned by every backend.

    `raw` keeps the backend's untouched response so callers that need
    backend-specific fields aren't blocked by the abstraction.
    """
    text: str
    language: str | None = None
    duration: float | None = None
    segments: Sequence[TranscriptionSegment] = field(default_factory=tuple)
    backend: str = ""
    raw: dict | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    """Per-request options. Backends ignore fields they don't support."""
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    response_format: ResponseFormat = "verbose_json"
    timeout: float | None = 300.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ASRError(Exception):
    """Base class for all ASR backend errors."""


class ASRAuthError(ASRError):
    """Raised on 401/403 or equivalent."""


class ASRTimeoutError(ASRError):
    """Raised when a request exceeds its deadline."""


class ASRUnavailableError(ASRError):
    """Raised when the backend is unreachable or returns 5xx."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ASRBackend(abc.ABC):
    """Abstract async ASR backend.

    Subclasses MUST implement `_transcribe`. Everything else has a sensible
    default — override only what you need.

    Lifecycle: use as an async context manager, or call `aclose()` manually.
    Backends that hold no resources can leave `aclose` as a no-op.
    """

    name: str = "abstract"

    # ---- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> "ASRBackend":
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def startup(self) -> None:
        """Optional: warm connections, load models, etc."""

    async def aclose(self) -> None:
        """Optional: release sessions, sockets, GPU memory, etc."""

    # ---- public API (do not override) ------------------------------------

    async def transcribe(
        self,
        audio: AudioSource,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        """Transcribe a single audio source.

        This is the stable entry point callers should use. It handles
        option defaulting and timeout enforcement, then delegates to
        `_transcribe`.
        """
        opts = options or TranscriptionOptions()
        coro = self._transcribe(audio, opts)
        if opts.timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=opts.timeout)
        except asyncio.TimeoutError as e:
            raise ASRTimeoutError(
                f"{self.name} transcription exceeded {opts.timeout}s"
            ) from e

    async def transcribe_many(
        self,
        sources: Sequence[AudioSource],
        options: TranscriptionOptions | None = None,
        max_concurrency: int = 4,
    ) -> list[TranscriptionResult]:
        """Transcribe a batch with bounded concurrency.

        Default uses a semaphore. Override if your backend has a native
        batch endpoint that's more efficient.
        """
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(src: AudioSource) -> TranscriptionResult:
            async with sem:
                return await self.transcribe(src, options)

        return await asyncio.gather(*(_one(s) for s in sources))

    async def transcribe_stream(
        self,
        chunks: AsyncIterator[bytes],
        options: TranscriptionOptions | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Streaming transcription. Default: not supported.

        Backends with WebSocket / chunked HTTP support should override.
        Made an async generator so callers can `async for` over partials.
        """
        raise NotImplementedError(
            f"{self.name} does not support streaming transcription"
        )
        # Unreachable, but keeps the type checker happy about the
        # async-generator return signature:
        if False:
            yield  # type: ignore[unreachable]

    # ---- subclass contract ----------------------------------------------

    @abc.abstractmethod
    async def _transcribe(
        self,
        audio: AudioSource,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        """Backend-specific transcription. Must return a TranscriptionResult.

        Implementations should raise the exceptions defined above
        (ASRAuthError, ASRUnavailableError, etc.) rather than leaking
        backend-specific errors.
        """

    @abc.abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is reachable and ready."""

    # ---- helpers subclasses can use --------------------------------------

    @staticmethod
    def _read_audio(audio: AudioSource) -> tuple[bytes, str]:
        """Normalize an AudioSource to (bytes, filename)."""
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            return path.read_bytes(), path.name
        if isinstance(audio, bytes):
            return audio, "audio.wav"
        # File-like
        data = audio.read()
        name = getattr(audio, "name", "audio.wav")
        return data, Path(str(name)).name