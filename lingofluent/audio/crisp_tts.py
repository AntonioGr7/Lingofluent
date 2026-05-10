"""CrispASR implementation of TTSBackend, hitting /v1/audio/speech."""

import json
import os
import aiohttp

from lingofluent.audio.tts_base import (
    AudioFormat,
    SynthesisOptions,
    SynthesisResult,
    TTSAuthError,
    TTSBackend,
    TTSInputError,
    TTSUnavailableError,
    TTSVoiceError,
)


# Loose mapping: response_format → MIME type CrispASR returns.
_FORMAT_MIME = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "pcm":  "audio/L16",
}


class CrispASRTTSBackend(TTSBackend):
    """TTS via a CrispASR --server instance.

    Uses the OpenAI-compatible POST /v1/audio/speech endpoint, which
    accepts JSON with {input, voice, response_format, speed, ...}.
    Voice strings refer to entries in the server's --voice-dir.
    """

    name = "crispasr-tts"
    supports_streaming = False             # endpoint returns full payload
    supports_voice_cloning = True          # via voice files in --voice-dir
    supports_voice_listing = True          # GET /v1/voices
    supports_voice_description = False     # qwen3-tts VoiceDesign would set True
    max_input_chars = 4096                 # CrispASR's --tts-max-input-chars default

    def __init__(
        self,
        base_url: str = os.environ.get('TTS_BASE_URL'),
        api_key: str | None = None,
        default_sample_rate: int = 24000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_sample_rate = default_sample_rate
        self._session: aiohttp.ClientSession | None = None

    # ---- lifecycle -------------------------------------------------------

    async def startup(self) -> None:
        self._session = aiohttp.ClientSession()

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def health(self) -> bool:
        assert self._session is not None
        try:
            async with self._session.get(f"{self.base_url}/health") as r:
                if r.status != 200:
                    return False
            # If a voice was specified at startup we'd want to know about it,
            # but at minimum: confirm the server can answer voice queries.
            async with self._session.get(
                f"{self.base_url}/v1/voices",
                headers=self._auth_headers(),
            ) as r:
                return r.status == 200
        except aiohttp.ClientError:
            return False

    # ---- voice listing ---------------------------------------------------

    async def _list_voices(self) -> list[str]:
        assert self._session is not None
        headers = self._auth_headers()
        try:
            async with self._session.get(
                f"{self.base_url}/v1/voices", headers=headers
            ) as r:
                r.raise_for_status()
                payload = await r.json()
        except aiohttp.ClientError as e:
            raise TTSUnavailableError(str(e)) from e

        # OpenAI-style: {"data": [{"id": "alice", ...}, ...]}.
        # Fall back to a plain {"voices": [...]} shape just in case.
        if "data" in payload:
            return [v.get("id", v.get("name", "")) for v in payload["data"]]
        return list(payload.get("voices", []))

    # ---- synthesis -------------------------------------------------------

    async def _synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> SynthesisResult:
        assert self._session is not None, "call startup() or use async with"

        body: dict = {
            "input": text,
            "response_format": options.response_format,
            "speed": options.speed,
        }

        voice = options.resolved_voice()
        if voice is not None:
            if voice.reference is not None:
                # Reference WAV → multipart, not JSON. Different code path.
                return await self._synthesize_multipart(text, options, voice)
            if voice.name is not None:
                body["voice"] = voice.name

        if options.instructions:
            body["instructions"] = options.instructions
        if options.temperature is not None:
            body["temperature"] = options.temperature
        if options.seed is not None:
            body["seed"] = options.seed

        headers = self._auth_headers() | {"Content-Type": "application/json"}

        try:
            async with self._session.post(
                f"{self.base_url}/v1/audio/speech",
                data=json.dumps(body),
                headers=headers,
            ) as resp:
                return await self._handle_audio_response(resp, options, voice)
        except aiohttp.ClientError as e:
            raise TTSUnavailableError(str(e)) from e

    async def _synthesize_multipart(
        self,
        text: str,
        options: SynthesisOptions,
        voice,
    ) -> SynthesisResult:
        """Voice-cloning path: send the reference WAV as multipart."""
        assert self._session is not None

        if voice.reference_text is None:
            raise TTSVoiceError(
                "reference_text is required when voice.reference is a WAV"
            )

        ref_bytes, ref_name = self._read_reference(voice.reference)

        form = aiohttp.FormData()
        form.add_field("input", text)
        form.add_field("response_format", options.response_format)
        form.add_field("speed", str(options.speed))
        form.add_field("voice_file", ref_bytes, filename=ref_name,
                       content_type="audio/wav")
        form.add_field("ref_text", voice.reference_text)
        if options.instructions:
            form.add_field("instructions", options.instructions)

        headers = self._auth_headers()
        try:
            async with self._session.post(
                f"{self.base_url}/v1/audio/speech",
                data=form,
                headers=headers,
            ) as resp:
                return await self._handle_audio_response(resp, options, voice)
        except aiohttp.ClientError as e:
            raise TTSUnavailableError(str(e)) from e

    # ---- response handling ----------------------------------------------

    async def _handle_audio_response(
        self,
        resp: aiohttp.ClientResponse,
        options: SynthesisOptions,
        voice,
    ) -> SynthesisResult:
        if resp.status in (401, 403):
            raise TTSAuthError(f"auth failed: {resp.status}")
        if resp.status == 400:
            detail = await resp.text()
            raise TTSInputError(f"bad request: {detail}")
        if resp.status == 404:
            # Server returns 404 when the requested voice is unknown.
            detail = await resp.text()
            raise TTSVoiceError(f"voice not found: {detail}")
        if resp.status >= 500:
            raise TTSUnavailableError(f"server error: {resp.status}")
        resp.raise_for_status()

        audio = await resp.read()
        return SynthesisResult(
            audio=audio,
            format=options.response_format,
            sample_rate=options.sample_rate or self.default_sample_rate,
            backend=self.name,
            voice=(voice.name if voice else None),
            raw=None,                   # binary endpoint, no JSON envelope
        )

    # ---- helpers ---------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}