"""
Voice module
- transcribe(audio_bytes) → str  using faster-whisper (CPU, base model)
- synthesize(text)        → bytes using edge-tts (cloud, no GPU needed)
"""

import tempfile
import os
import asyncio
import edge_tts
from faster_whisper import WhisperModel

# Lazy-load Whisper model (downloaded once, ~75MB)
_whisper_model = None

def _get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            download_root="/tmp/whisper"
        )
    return _whisper_model


def transcribe(audio_bytes: bytes) -> str:
    """Convert audio bytes → text using faster-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        model    = _get_whisper()
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


async def _synthesize_async(text: str, voice: str = "en-US-JennyNeural") -> bytes:
    """Convert text → MP3 bytes using edge-tts."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


async def synthesize(text: str, voice: str = "en-US-JennyNeural") -> bytes:
    """Public async interface for TTS synthesis."""
    # Trim to reasonable length for voice output
    if len(text) > 800:
        text = text[:800] + "..."
    return await _synthesize_async(text, voice)
