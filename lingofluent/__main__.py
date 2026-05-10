from dotenv import load_dotenv
load_dotenv() 

import logging
import os
from lingofluent.bot.telegram.telegram_bot import Telegram
from lingofluent.logging_config import setup_logging
from lingofluent.configs.config import Configuration

setup_logging()
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# os.environ[...] raises KeyError immediately if the var is missing — fail loud, fail closed.

def main():
    llm_type = os.environ['LLM_TYPE']
    if llm_type not in Configuration.SUPPORTED_LLM_TYPES:
        raise ValueError(f"Unsupported LLM_TYPE '{llm_type}'. Supported values: {Configuration.SUPPORTED_LLM_TYPES}")

    if llm_type == "openai":
        # Check OPENAI_API_KEY presence early, since it's required to even start the bot.
        _ = os.environ['OPENAI_API_KEY']
    elif llm_type == "llama_cpp":
        # Check LLM_BASE_URL presence early, since it's required to even start the bot.
        _ = os.environ['LLM_BASE_URL']


    TOKEN = os.environ['BOT_TOKEN']
    USER_ID = int(os.environ['USER_ID'])
    # In a 1:1 DM with a bot, chat.id == user.id. Use the same value for both.
    CHAT_ID = int(os.environ.get('CHAT_ID', USER_ID))
    ALLOWED_CHAT_IDS = [CHAT_ID]
    ALLOWED_USER_IDS = [USER_ID]
    logger.info("[Lingofluent] Starting bot")
    telegram_bot = Telegram(
        TOKEN,
        CHAT_ID,
        allowed_chat_ids=ALLOWED_CHAT_IDS,
        allowed_user_ids=ALLOWED_USER_IDS,
    )
    telegram_bot.start()


if __name__ == "__main__":
    main()