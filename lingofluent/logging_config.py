import logging
import os
import sys
from typing import Optional, Union

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_NOISY_LOGGERS = ("httpx", "httpcore", "telegram", "telegram.ext", "apscheduler")


def setup_logging(
    level: Optional[Union[str, int]] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure application logging. Call once from the entry point.

    `level` and `log_file` fall back to the LOG_LEVEL and LOG_FILE env vars.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_file = log_file or os.getenv("LOG_FILE")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATEFMT,
        handlers=handlers,
        force=True,
    )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
