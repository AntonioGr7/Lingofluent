import functools
import logging
import re

from lingofluent.audio.tts_base import TTSError, TTSInputError, TTSUnavailableError, TTSVoiceError
from lingofluent.llm.llm_base import LLMInputError, LLMUnavailableError

logger = logging.getLogger(__name__)


def handle_backend_errors(fn):
    """Decorator for Telegram handler methods (self, message, session, ...).

    Catches LLM and TTS exceptions and sends a user-facing reply instead of
    crashing the handler silently.
    """
    @functools.wraps(fn)
    async def wrapper(self, message, session, *args, **kwargs):
        try:
            return await fn(self, message, session, *args, **kwargs)
        except LLMInputError as exc:
            logger.warning("LLM input error in %s: %s", fn.__name__, exc)
            await message.reply_text(
                "Your conversation history is too long for the model's context window. "
                "Use /reset to start a fresh session and try again."
            )
        except LLMUnavailableError as exc:
            logger.error("LLM unavailable in %s: %s", fn.__name__, exc)
            await message.reply_text(
                "The language model is currently unavailable. "
                "Please check that the LLM server is running and try again."
            )
        except TTSVoiceError as exc:
            logger.warning("TTS voice error in %s: %s", fn.__name__, exc)
            await message.reply_text(
                "Voice not found. Set TTS_VOICE in your .env to a valid voice name, "
                "or leave it empty to use the server default."
            )
        except TTSInputError as exc:
            logger.warning("TTS input error in %s: %s", fn.__name__, exc)
            await message.reply_text(f"TTS input error: {exc}")
        except TTSUnavailableError as exc:
            logger.error("TTS unavailable in %s: %s", fn.__name__, exc)
            await message.reply_text(
                "The TTS server is currently unavailable. "
                "Please check that the CrispASR TTS server is running and try again."
            )
        except TTSError as exc:
            logger.error("TTS error in %s: %s", fn.__name__, exc)
            await message.reply_text("An unexpected TTS error occurred. Please try again.")
    return wrapper


# Keep old name as alias so existing usages don't break
handle_llm_errors = handle_backend_errors


def md_to_html(text: str) -> str:
    """Convert a subset of Markdown to Telegram-compatible HTML.

    Supported: **bold**, *italic*, `inline code`, ```code block```.
    Also escapes raw &, <, > so they render literally.
    """
    # Escape HTML special chars first (before we insert tags)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Fenced code blocks (``` ... ```) — must come before inline-code pass
    text = re.sub(r"```(?:\w+\n)?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold (**text** or __text__)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic (*text* or _text_) — after bold so ** isn't caught here
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)
    return text