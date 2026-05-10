import logging
import os
import re
import tempfile

from lingofluent.audio.crisp_tts import CrispASRTTSBackend
from lingofluent.audio.tts_base import SynthesisOptions
from lingofluent.audio.utils import wav_to_ogg_opus
from lingofluent.bot.telegram.utils import md_to_html

logger = logging.getLogger(__name__)



async def _handle_reset_command(**kwargs):
    kwargs["session"].reset()
    await kwargs["message"].reply_text("Conversation has been reset. Let's start fresh!")
    await _handle_start_command(**kwargs)  

_handle_reset_command.description = "Reset the current conversation."

async def _handle_start_command(**kwargs):
    kwargs["session"].reset()
    with open("lingofluent/configs/welcome_message.md", "r", encoding="utf-8") as f:
        welcome_message = f.read()
    await kwargs["message"].reply_text(md_to_html(welcome_message), parse_mode="HTML")
_handle_start_command.description = "Start a new conversation. Reset the context and show the welcome message."

async def _handle_read_command(**kwargs):
    message = kwargs["message"]
    text = message.text.removeprefix("/read").strip()
    if not text:
        if kwargs["session"].history():
            text = kwargs["session"].history()[-1].content
        else:
            await message.reply_text("Please provide some text after the /read command.")
            return
    logger.info("Received text for TTS: %s", text)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "out.wav")
        ogg_path = os.path.join(tmp, "out.ogg")
        async with CrispASRTTSBackend() as tts:
            if not await tts.health():
                await message.reply_text("TTS server not ready.")
                return
            tts_voice = os.environ.get("TTS_VOICE") # Cannot be None using Qwen3-TTS backend
            await tts.synthesize_to_file(
                text, wav_path, SynthesisOptions(voice=tts_voice, language="en"),
            )
        wav_to_ogg_opus(wav_path, ogg_path)
        await message.reply_voice(voice=ogg_path)

_handle_read_command.description = "Generate audio from the text you send me."


async def _handle_help_command(**kwargs):
    message = kwargs["message"]
    help_text = "**Available Commands:**\n\n"
    for command, handler in _command_handlers.items():
        desc = getattr(handler, "description", "No description available.")
        help_text += f"`{command}` — {desc}\n"
    await message.reply_text(md_to_html(help_text), parse_mode="HTML")

_handle_help_command.description = "Show available commands and their descriptions."


_command_handlers = {
    "/help": _handle_help_command,
    "/read": _handle_read_command,
    "/reset": _handle_reset_command,
    "/start": _handle_start_command, 
}
