# Voice packs

Place voice reference files here. The TTS server (qwen3-tts) is started
with `--voice-dir` pointing to this folder — each file pair becomes a named
voice that the bot can select via `TTS_VOICE` in `.env`.

## Adding a custom voice

### 1. Record or prepare a reference clip

Requirements:

| Property | Value |
|---|---|
| Format | WAV (PCM) |
| Sample rate | 24 000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit (2 bytes/sample) |
| Duration | 5–15 seconds |

Keep the clip clean — no background music, no noise, no reverb. The
speaker should talk at a natural pace with clear diction.

### 2. Write the transcription

Create a `.txt` file with **exactly** the same base name as the WAV.
The content must be a verbatim transcription of what is said in the clip
— every word, punctuation, and comma matters.

```
voices/
  antonio.wav   ← reference audio
  antonio.txt   ← exact transcription of antonio.wav
```

Example `antonio.txt`:
```
Hello, my name is Antonio. I am recording this voice sample to clone my voice.
```

### 3. Set the voice in `.env`

```env
TTS_VOICE=antonio
```

The bot will send `"voice": "antonio"` in every TTS request. The server
looks up `voices/antonio.wav` + `voices/antonio.txt` and uses them as the
cloning reference.

### 4. Restart the TTS server

```bash
bash scripts/start_servers.sh stop
bash scripts/start_servers.sh start
```

## Tips

- The transcription must match the WAV **exactly** — it is fed as `ref_text` during synthesis. A mismatch degrades clone quality noticeably.
- Longer clips (10–15 s) generally produce better clones than very short ones.
- If you have multiple speakers, add one pair per speaker and switch `TTS_VOICE` in `.env` to change who speaks.
