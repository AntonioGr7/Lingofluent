import logging
import os
import re
import tempfile
from pathlib import Path

from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from lingofluent.audio.asr_base import TranscriptionOptions
from lingofluent.audio.crisp_asr import CrispASRBackend
from lingofluent.bot.telegram.bot_cmds import _command_handlers
from lingofluent.bot.telegram.utils import handle_llm_errors, md_to_html
from lingofluent.llm.llm_base import ChatOptions, Message, TextPart, ImagePart
from lingofluent.memory.session import Session, SessionStore
from lingofluent.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class Telegram():
    commands = ["/read", "/help", "/reset", "/start"]  

    def __init__(self, token, chat_id, allowed_chat_ids=None, allowed_user_ids=None):
        if not token:
            raise ValueError("token is required")
        if not allowed_user_ids:
            raise ValueError(
                "allowed_user_ids must be a non-empty list — refusing to start with an open allowlist"
            )

        self.token = token
        self.chat_id = chat_id
        self.allowed_chat_ids = set(allowed_chat_ids or [])
        self.allowed_user_ids = set(allowed_user_ids)
        self.bot = Bot(token=self.token)
        self.application = ApplicationBuilder().token(self.token).build()
        self.store = SessionStore()
        self.llm = LLMFactory(os.environ.get("LLM_TYPE")).llm

        # Build a filter that matches only the whitelisted commands in `self.commands`.
        command_names = [c.lstrip("/") for c in self.commands]
        allowed_command_pattern = (
            r"^/(?:" + "|".join(re.escape(n) for n in command_names) + r")(?:@\w+)?(?:\s|$)"
        )
        allowed_commands = filters.Regex(allowed_command_pattern)

        content_filter = (
            (filters.TEXT & ~filters.COMMAND)
            | allowed_commands
            | filters.VOICE
            | filters.AUDIO
            | filters.PHOTO
        )
        # Layer 1: framework-level filter. Unauthorized updates never reach the handler.
        full_filter = (
            filters.ChatType.PRIVATE
            & filters.User(user_id=list(self.allowed_user_ids))
            & content_filter
        )
        if self.allowed_chat_ids:
            full_filter = full_filter & filters.Chat(chat_id=list(self.allowed_chat_ids))

        self.application.add_handler(MessageHandler(full_filter, self.wait_for_message))

        prompt_path = Path(os.environ.get("SYSTEM_PROMPT_PATH", "lingofluent\configs\system_prompt.md"))
        self._system_message = Message(role="system", content=prompt_path.read_text(encoding="utf-8").strip()) if prompt_path.exists() else None
        if not self._system_message:
            raise ValueError(f"System prompt file not found at {prompt_path}. Please provide a valid SYSTEM_PROMPT_PATH environment variable.")
    async def send_message(self, text):
        await self.bot.send_message(chat_id=self.chat_id, text=text)

    def _is_authorized(self, message) -> bool:
        # Layer 2: defense-in-depth re-check inside the handler.
        if message.chat.type != "private":
            return False
        if message.sender_chat is not None:
            return False
        if message.from_user is None:
            return False
        if message.from_user.id not in self.allowed_user_ids:
            return False
        if self.allowed_chat_ids and message.chat.id not in self.allowed_chat_ids:
            return False
        if message.text and message.text.startswith("/"):
            command = message.text.split(maxsplit=1)[0].split("@", 1)[0]
            if command not in self.commands:
                return False
        return True

    async def _chat_with_history(self, session: Session, user_content) -> str:
        """Add user turn, call the LLM with full history, persist assistant reply."""
        session.add("user", user_content)
        history = session.history()
        if self._system_message:
            history = [self._system_message] + history
        async with self.llm as llm:
            result = await llm.chat(
                history,
                ChatOptions(temperature=0.7, max_tokens=None),
            )
        session.add("assistant", result.text)
        return result.text

    async def wait_for_message(self, update: Update, _context: ContextTypes.DEFAULT_TYPE):
        logger.debug("Update received: %s", update)
        message = update.effective_message
        if not message:
            return
        if not self._is_authorized(message):
            logger.warning(
                "Ignored message from unauthorized sender: chat=%s user=%s",
                message.chat.id, message.from_user,
            )
            return

        session = Session(message.chat.id, self.store)

        if message.voice:
            await self.handle_voice_message(message, session)

        elif message.audio:
            await self.handle_audio_message(message, session)

        elif message.photo:
            await self.handle_photo_message(message, session)

        elif message.text and message.text.startswith("/"):
            await self.handle_command(message, session)

        elif message.text:
            await self.handle_text_message(message, session)
    
    def start(self):
        self.application.run_polling()


## --- Handlers for different message types ---

    @handle_llm_errors
    async def handle_voice_message(self, message, session):
        logger.info("[SESSION %d] Voice message received, duration: %ds", session.session_id, message.voice.duration)
        with tempfile.TemporaryDirectory() as tmp_dir:
            oga_path = os.path.join(tmp_dir, f"{message.voice.file_unique_id}.oga")
            voice_file = await message.voice.get_file()
            await voice_file.download_to_drive(oga_path)
            logger.info("Voice message downloaded to %s", oga_path)
            async with CrispASRBackend(api_key=None) as asr:
                result = await asr.transcribe(
                    oga_path,
                    TranscriptionOptions(language="en", response_format="verbose_json"),
                )
            
            logger.info("Transcription complete (%d chars)", len(result.text))
            transcription_info = f"**Transcription**:\n{result.text}"
            await message.reply_text(md_to_html(transcription_info), parse_mode="HTML")
            if not result.text.strip():
                await message.reply_text("I couldn't transcribe the audio. Please try again with clearer speech or less background noise.")
            else:
                reply = await self._chat_with_history(session, result.text)
                await message.reply_text(md_to_html(reply), parse_mode="HTML")

    @handle_llm_errors
    async def handle_audio_message(self, message, session):
        # TODO
        logger.info("[SESSION %d] Audio file received: %s", session.session_id, message.audio.file_name)
        audio_file = await message.audio.get_file()
        file_path = await audio_file.download_to_drive()
        await message.reply_text("I've received your audio file!")
    
    @handle_llm_errors
    async def handle_photo_message(self, message, session):
        caption = message.caption or "Tell the user that you receive the photo and asking what they want to do with it."
        logger.info("[SESSION %d] Photo received, caption: %s", session.session_id, caption)
        file = await message.photo[-1].get_file()
        data = await file.download_as_bytearray()
        
        content = [TextPart(text=caption), ImagePart(data=bytes(data), mime="image/jpeg")]
        
        reply = await self._chat_with_history(session, content)
        await message.reply_text(md_to_html(reply), parse_mode="HTML")
    
    @handle_llm_errors
    async def handle_text_message(self, message, session):
        logger.info("[SESSION %d] Text content: %s", session.session_id, message.text)
        reply = await self._chat_with_history(session, message.text)
        await message.reply_text(md_to_html(reply), parse_mode="HTML")
    
    @handle_llm_errors
    async def handle_command(self, message, session):
        logger.info("[SESSION %d] Command received: %s", session.session_id, message.text)
        command = message.text.split(maxsplit=1)[0].split("@", 1)[0]
        handler = _command_handlers.get(command)
        if handler:
            await handler(bot=self, message=message, session=session)
        else:
            await message.reply_text("I cannot use this command. Try /help for available commands.")
    
    



