import logging
import os

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def convert_oga_to_wav(input_path: str, output_path: str):
    """Convert an .oga/.ogg audio file to .wav."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} was not found.")

    logger.info("Converting '%s' to '%s'", input_path, output_path)
    try:
        audio = AudioSegment.from_file(input_path, format="ogg")
        audio.export(output_path, format="wav")
        logger.debug("Conversion completed: %s", output_path)
    except Exception:
        logger.exception("Failed to convert %s", input_path)
        raise

def wav_to_ogg_opus(wav_path: str, ogg_path: str, bitrate: str = "64k") -> None:
    """Encode a WAV file to OGG/OPUS so Telegram renders it as a voice note."""
    audio = AudioSegment.from_file(wav_path, format="wav")
    audio.export(ogg_path, format="ogg", codec="libopus", bitrate=bitrate)


def batch_convert(folder_path="."):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".oga"):
            continue
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.wav")
        try:
            audio = AudioSegment.from_file(input_path, format="ogg")
            audio.export(output_path, format="wav")
            logger.info("Converted: %s", filename)
        except Exception:
            logger.exception("Failed to convert %s", filename)