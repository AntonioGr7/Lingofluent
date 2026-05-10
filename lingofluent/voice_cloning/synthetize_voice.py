from lingofluent.audio.crisp_asr import CrispASRBackend
from lingofluent.audio.asr_base import TranscriptionOptions
import asyncio
from lingofluent.audio.crisp_tts import CrispASRTTSBackend
from lingofluent.audio.tts_base import SynthesisOptions
from lingofluent.audio.utils import convert_oga_to_wav, batch_convert



async def main():
    async with CrispASRTTSBackend() as tts:
        if not await tts.health():
            raise RuntimeError("server not ready")

        # Preset voice (lives in --voice-dir on the server)
        await tts.synthesize_to_file(
            "This update makes everyday interactions more useful and more enjoyable: stronger and tighter answers across subject areas, a more natural conversational tone, and better use of the context you’ve already shared when personalization can help.",
            "output_voice_cloning.wav",
            SynthesisOptions(voice="antonio", language="en"),
        )


if __name__ == "__main__":
    #asyncio.run(main())
    #convert_oga_to_wav("file_3.oga", "file.wav")
    asyncio.run(main())