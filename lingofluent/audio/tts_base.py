"""
Abstract base class for async TTS backends.

Mirrors the ASRBackend design: one abstract method (`_synthesize`),
sensible defaults for everything else, and a normalized result type
that preserves the backend's raw payload as an escape hatch.
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

AudioFormat = Literal["wav", "mp3", "opus", "flac", "pcm"]
ReferenceAudio = str | Path | bytes | BinaryIO


@dataclass(frozen=True)
class Voice:
    """A voice that can be passed to `synthesize`.

    Three shapes a backend may accept (not all backends support all):
      - preset name:        Voice(name="tara")
      - reference audio:    Voice(reference=Path("clone.wav"),
                                  reference_text="exact transcription")
      - description prompt: Voice(description="young female, British accent")
    """
    name: str | None = None
    reference: ReferenceAudio | None = None
    reference_text: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        # At least one identifier must be set.
        if not any((self.name, self.reference, self.description)):
            raise ValueError(
                "Voice requires at least one of: name, reference, description"
            )


@dataclass(frozen=True)
class SynthesisOptions:
    """Per-request synthesis options. Backends ignore unsupported fields."""
    voice: Voice | str | None = None      # str → Voice(name=str)
    language: str | None = None
    speed: float = 1.0                    # 0.25–4.0, OpenAI convention
    temperature: float | None = None      # backend default if None
    response_format: AudioFormat = "wav"
    sample_rate: int | None = None        # None = backend default (24 kHz)
    instructions: str | None = None       # OpenAI-style style prompt
    seed: int | None = None
    timeout: float | None = 300.0

    def resolved_voice(self) -> Voice | None:
        if self.voice is None:
            return None
        if isinstance(self.voice, str):
            return Voice(name=self.voice)
        return self.voice


@dataclass(frozen=True)
class SynthesisResult:
    """Normalized synthesis result.

    `audio` holds the encoded bytes in `format` at `sample_rate`.
    `raw` keeps the backend's untouched response for callers that need
    backend-specific fields (e.g. token timings, alignment).
    """
    audio: bytes
    format: AudioFormat
    sample_rate: int
    duration: float | None = None
    backend: str = ""
    voice: str | None = None
    raw: dict | None = None

    def save(self, path: str | Path) -> Path:
        """Write the audio bytes to disk, returning the resolved path."""
        p = Path(path)
        p.write_bytes(self.audio)
        return p


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TTSError(Exception):
    """Base class for TTS backend errors."""


class TTSAuthError(TTSError):
    """Raised on 401/403 or equivalent."""


class TTSTimeoutError(TTSError):
    """Raised when a request exceeds its deadline."""


class TTSUnavailableError(TTSError):
    """Raised when the backend is unreachable or returns 5xx."""


class TTSVoiceError(TTSError):
    """Raised for unknown voice / missing reference text / bad ref audio."""


class TTSInputError(TTSError):
    """Raised for input the backend cannot handle (too long, empty, etc.)."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TTSBackend(abc.ABC):
    """Abstract async TTS backend.

    Subclasses MUST implement `_synthesize` and `health`. Streaming and
    voice listing are optional — defaults raise NotImplementedError so
    callers can detect support with capability flags.

    Lifecycle: use as an async context manager, or call aclose() manually.
    """

    name: str = "abstract"

    # Capability flags — subclasses override what they actually support.
    supports_streaming: bool = False
    supports_voice_cloning: bool = False
    supports_voice_listing: bool = False
    supports_voice_description: bool = False
    max_input_chars: int | None = None    # None = unbounded

    # ---- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> "TTSBackend":
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def startup(self) -> None:
        """Optional: warm connections, load models, etc."""

    async def aclose(self) -> None:
        """Optional: release sessions, sockets, GPU memory, etc."""

    # ---- public API (do not override) ------------------------------------

    async def synthesize(
        self,
        text: str,
        options: SynthesisOptions | None = None,
    ) -> SynthesisResult:
        """Synthesize speech for `text`.

        Stable entry point: handles validation, option defaulting, and
        timeout enforcement, then delegates to `_synthesize`.
        """
        opts = options or SynthesisOptions()
        self._validate_request(text, opts)

        coro = self._synthesize(text, opts)
        if opts.timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=opts.timeout)
        except asyncio.TimeoutError as e:
            raise TTSTimeoutError(
                f"{self.name} synthesis exceeded {opts.timeout}s"
            ) from e

    async def synthesize_to_file(
        self,
        text: str,
        path: str | Path,
        options: SynthesisOptions | None = None,
    ) -> Path:
        """Convenience: synthesize and write the audio to `path`."""
        result = await self.synthesize(text, options)
        return result.save(path)

    async def synthesize_many(
        self,
        items: Sequence[str | tuple[str, SynthesisOptions]],
        max_concurrency: int = 4,
        default_options: SynthesisOptions | None = None,
    ) -> list[SynthesisResult]:
        """Synthesize a batch with bounded concurrency.

        Each item is either a string (uses `default_options`) or a
        (text, options) tuple. Override if a backend has a native batch
        endpoint that's more efficient.
        """
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(item: str | tuple[str, SynthesisOptions]) -> SynthesisResult:
            if isinstance(item, tuple):
                text, opts = item
            else:
                text, opts = item, default_options
            async with sem:
                return await self.synthesize(text, opts)

        return await asyncio.gather(*(_one(i) for i in items))

    async def synthesize_stream(
        self,
        text: str,
        options: SynthesisOptions | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated.

        Default: not supported. Backends with chunked HTTP / WebSocket
        TTS should override and set `supports_streaming = True`.
        """
        if not self.supports_streaming:
            raise NotImplementedError(
                f"{self.name} does not support streaming synthesis"
            )
        async for chunk in self._synthesize_stream(text, options or SynthesisOptions()):
            yield chunk

    async def list_voices(self) -> list[str]:
        """List available voice names."""
        if not self.supports_voice_listing:
            raise NotImplementedError(
                f"{self.name} does not expose a voice list"
            )
        return await self._list_voices()

    # ---- subclass contract ----------------------------------------------

    @abc.abstractmethod
    async def _synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> SynthesisResult:
        """Backend-specific synthesis. Translate backend errors into the
        TTSError hierarchy rather than leaking transport exceptions."""

    @abc.abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is reachable and ready."""

    async def _synthesize_stream(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> AsyncIterator[bytes]:
        """Override to provide streaming. Default unreachable."""
        raise NotImplementedError
        if False:                         # keeps async-generator typing happy
            yield b""

    async def _list_voices(self) -> list[str]:
        """Override to provide voice listing."""
        return []

    # ---- helpers ---------------------------------------------------------

    def _validate_request(self, text: str, opts: SynthesisOptions) -> None:
        """Cheap precondition checks before hitting the network."""
        if not text or not text.strip():
            raise TTSInputError("text must be non-empty")
        if self.max_input_chars and len(text) > self.max_input_chars:
            raise TTSInputError(
                f"text length {len(text)} exceeds {self.name} cap "
                f"of {self.max_input_chars} chars"
            )
        voice = opts.resolved_voice()
        if voice and voice.description and not self.supports_voice_description:
            raise TTSVoiceError(
                f"{self.name} does not support description-based voices"
            )
        if voice and voice.reference and not self.supports_voice_cloning:
            raise TTSVoiceError(
                f"{self.name} does not support reference-WAV voice cloning"
            )

    @staticmethod
    def _read_reference(ref: ReferenceAudio) -> tuple[bytes, str]:
        """Normalize a reference audio source to (bytes, filename)."""
        if isinstance(ref, (str, Path)):
            path = Path(ref)
            return path.read_bytes(), path.name
        if isinstance(ref, bytes):
            return ref, "reference.wav"
        data = ref.read()
        name = getattr(ref, "name", "reference.wav")
        return data, Path(str(name)).name