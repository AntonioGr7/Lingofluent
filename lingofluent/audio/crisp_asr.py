"""CrispASR implementation of ASRBackend."""

import aiohttp
import os

from lingofluent.audio.asr_base import (
    ASRAuthError,
    ASRBackend,
    ASRUnavailableError,
    AudioSource,
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSegment,
)


class CrispASRBackend(ASRBackend):
    name = "crispasr"

    def __init__(
        self,
        base_url: str = os.environ.get('ASR_BASE_URL'),
        api_key: str | None = None,
        use_openai_endpoint: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.use_openai_endpoint = use_openai_endpoint
        self._session: aiohttp.ClientSession | None = None

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
                return r.status == 200
        except aiohttp.ClientError:
            return False

    async def _transcribe(
        self,
        audio: AudioSource,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        assert self._session is not None, "call startup() or use async with"

        data, filename = self._read_audio(audio)
        headers = (
            {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )

        form = aiohttp.FormData()
        form.add_field("file", data, filename=filename,
                       content_type="audio/wav")

        if self.use_openai_endpoint:
            url = f"{self.base_url}/v1/audio/transcriptions"
            form.add_field("response_format", options.response_format)
            form.add_field("temperature", str(options.temperature))
            if options.language:
                form.add_field("language", options.language)
            if options.prompt:
                form.add_field("prompt", options.prompt)
        else:
            url = f"{self.base_url}/inference"

        try:
            async with self._session.post(url, data=form,
                                          headers=headers) as resp:
                if resp.status in (401, 403):
                    raise ASRAuthError(f"auth failed: {resp.status}")
                if resp.status >= 500:
                    raise ASRUnavailableError(f"server error: {resp.status}")
                resp.raise_for_status()
                payload = await resp.json()
        except aiohttp.ClientError as e:
            raise ASRUnavailableError(str(e)) from e

        segments = tuple(
            TranscriptionSegment(
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", ""),
            )
            for s in payload.get("segments", [])
        )
        return TranscriptionResult(
            text=payload.get("text", ""),
            language=payload.get("language"),
            duration=payload.get("duration"),
            segments=segments,
            backend=payload.get("backend", self.name),
            raw=payload,
        )