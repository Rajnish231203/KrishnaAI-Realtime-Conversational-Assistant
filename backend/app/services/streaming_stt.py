"""
[DEPRECATED — DO NOT IMPORT IN ACTIVE RUNTIME CODE]
====================================================
streaming_stt.py — Legacy FasterWhisper STT service.

STATUS: DEAD CODE as of STT migration Phase 2.
REPLACED BY: backend.app.services.stt (STTManager + provider layer)

This file is kept as historical reference ONLY.

WHY IT MUST NOT BE IMPORTED:
    The new system (services/stt/faster_whisper_provider.py) has its
    OWN FasterWhisper singleton (_whisper_model).  If this module is also
    imported at runtime, WhisperModel loads TWICE, causing:
      - ~2× RAM usage (~300 MB extra)
      - CPU contention between two inference workers
      - latency spikes on the realtime pipeline

MIGRATION PATH:
    Old: from backend.app.services.streaming_stt import StreamingSTT
    New: from backend.app.services.stt import STTManager

    Old: stt = StreamingSTT()
    New: stt = STTManager()  # warmup() + shutdown() lifecycle required
         await stt.warmup()

    Old: await stt.transcribe_partial(audio)  # → Optional[str]
    New: await stt.transcribe_partial(audio)  # → str ('' on silence)

    Old: await stt.transcribe_final(audio)    # → Optional[str]
    New: await stt.transcribe_final(audio)    # → str ('' on failure)

    Pre-warm at server startup:
        from backend.app.services.stt.faster_whisper_provider import prewarm_model
        await prewarm_model()

DO NOT REMOVE THIS FILE until all references are confirmed migrated.
"""

# ---------------------------------------------------------------------------
# HARD IMPORT GUARD — prevents this file from being used at runtime.
# Remove this block only after all tests and scripts have been migrated to
# the new STT provider layer.
# ---------------------------------------------------------------------------
import os as _os
if not _os.getenv("ALLOW_LEGACY_STT_IMPORT", "").lower() in ("1", "true", "yes"):
    raise ImportError(
        "\n\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  DEPRECATED MODULE — streaming_stt.py must not be imported  ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  Use the new STT provider layer instead:                    ║\n"
        "║                                                              ║\n"
        "║    from backend.app.services.stt import STTManager          ║\n"
        "║    stt = STTManager()                                        ║\n"
        "║    await stt.warmup()                                        ║\n"
        "║                                                              ║\n"
        "║  See streaming_stt.py header for full migration guide.      ║\n"
        "║                                                              ║\n"
        "║  To run legacy tests, set env var:                          ║\n"
        "║    ALLOW_LEGACY_STT_IMPORT=1 python <your_script>           ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

# ---------------------------------------------------------------------------
# LEGACY CODE BELOW — kept for reference only, NOT active runtime code.
# ---------------------------------------------------------------------------

"""
Streaming Speech-to-Text Service — Krishna Voice Assistant
===========================================================
Provides partial and final audio transcription via local FasterWhisper.

Target latency: first partial < 400 ms

Public interface (consumed by websocket/server.py):
    StreamingSTT.transcribe_partial(audio_bytes) → Optional[str]
    StreamingSTT.transcribe_final(audio_bytes)   → Optional[str]
    StreamingSTT.reset()                          → None

The module-level alias ``StreamingSTT`` always points to the local provider
so callers never need to know which backend is active.

Production notes
----------------
- Fix 1  : WhisperModel is now a module-level singleton protected by
           threading.Lock.  Every LocalStreamingSTT instance shares the
           same model object — no extra RAM per connection.
- Fix 2  : _transcribe_in_thread() exhausts the FasterWhisper segment
           generator entirely inside the worker thread so the asyncio event
           loop is never blocked by generator iteration.
- Issue 3 : WAV wrapping deferred — PCM wrapped once per API call.
- Issue 9 : Minimum audio length validated before STT call.
"""


import asyncio
import io
import threading
import time
import wave
import logging
from typing import Optional

from faster_whisper import WhisperModel

from backend.app.config.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

STT_MIN_AUDIO_BYTES: int = int(Config.SAMPLE_RATE * 2 * 0.2)


# ===========================================================================
# Fix 1 — Module-level FasterWhisper singleton
# ===========================================================================
# Loading WhisperModel costs ~150 MB RAM and 2–8 s on CPU.  Doing it once
# per process (rather than once per WebSocket connection) keeps memory flat
# at any connection count.
#
# Thread safety: double-checked locking with threading.Lock guarantees the
# model is initialised exactly once even when concurrent worker threads race
# to call _get_whisper_model() at startup.
# ===========================================================================

_model_lock: threading.Lock = threading.Lock()
_whisper_model: Optional[WhisperModel] = None


def _get_whisper_model() -> WhisperModel:
    """
    Return the process-wide FasterWhisper singleton, loading it on first call.

    Fix 1 — double-checked locking ensures at-most-one initialisation.

    Returns:
        The shared ``WhisperModel`` instance.
    """
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:          # re-check inside the lock
                logger.info(
                    "Loading FasterWhisper model '%s' on %s (singleton — first call).",
                    Config.LOCAL_WHISPER_MODEL,
                    Config.WHISPER_DEVICE,
                )
                _whisper_model = WhisperModel(
                    Config.LOCAL_WHISPER_MODEL,
                    device=Config.WHISPER_DEVICE,
                    compute_type="int8_float32",
                    cpu_threads=4,
                )
                logger.info("FasterWhisper singleton model loaded and ready.")
    return _whisper_model


# ===========================================================================
# Fix 2 — Transcription helper that exhausts generator inside worker thread
# ===========================================================================
# FasterWhisper's model.transcribe() returns a *lazy generator*.  If that
# generator is iterated on the asyncio event loop thread (e.g., in a join
# expression after await asyncio.to_thread(...)), it blocks the loop.
#
# This helper is passed directly to asyncio.to_thread so that both the
# model.transcribe() call and the segment iteration happen in the same
# worker thread and never touch the event loop.
# ===========================================================================

def _transcribe_in_thread(
    model: WhisperModel,
    audio_buffer: io.BytesIO,
    beam_size: int,
    language: Optional[str],
) -> str:
    """
    Run FasterWhisper inference and fully consume the segment generator
    inside the worker thread.

    Fix 2 — this function is designed to be called exclusively via
    ``asyncio.to_thread``.  Neither the ``model.transcribe()`` invocation
    nor the ``seg.text`` iteration ever execute on the event loop thread.

    Args:
        model:        Shared ``WhisperModel`` singleton.
        audio_buffer: In-memory WAV bytes, seeked to position 0.
        beam_size:    Beam width (1 = greedy/fast, 5 = accurate).
        language:     ISO 639-1 language code or None for auto-detect.
                      Use 'hi' for Hindi/Hinglish to prevent autodetect
                      instability on code-switched conversational speech.

    Returns:
        Whitespace-stripped transcript string (empty string on silence).
    """
    segments, _info = model.transcribe(
        audio_buffer,
        beam_size=beam_size,
        language=language,
        # condition_on_previous_text=False keeps successive segments independent
        # — prevents hallucinated continuations from bleeding across pauses.
        condition_on_previous_text=False,
    )
    # Exhaust the lazy generator here, inside the worker thread.
    return " ".join(seg.text for seg in segments).strip()


# ===========================================================================
# Shared WAV helper
# ===========================================================================

def _wrap_wav(pcm_bytes: bytes) -> bytes:
    """
    Wrap raw PCM audio bytes in a RIFF/WAV container.

    Issue 3 — wrapped once per API call on the already-sliced window.

    Args:
        pcm_bytes: Raw 16-bit signed PCM, little-endian, mono at SAMPLE_RATE.

    Returns:
        Complete WAV file bytes including RIFF header, ready for Whisper.
    """
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(Config.CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(Config.SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)
        return wav_io.getvalue()


def _validate_audio(audio_bytes: bytes, label: str) -> bool:
    """
    Issue 9 — reject buffers too short to contain meaningful speech.

    Args:
        audio_bytes: Raw PCM buffer to validate.
        label:       Context label for logging (``"partial"`` / ``"final"``).

    Returns:
        True if usable, False if the buffer should be skipped.
    """
    if len(audio_bytes) < STT_MIN_AUDIO_BYTES:
        logger.debug(
            "STT %s skipped — audio too short (%d bytes, minimum %d bytes).",
            label,
            len(audio_bytes),
            STT_MIN_AUDIO_BYTES,
        )
        return False
    return True


# ===========================================================================
# FasterWhisper STT (local)
# ===========================================================================

class LocalStreamingSTT:
    """
    Transcription via local FasterWhisper.

    Fix 1 — the heavy ``WhisperModel`` is no longer constructed here.
    ``__init__`` calls ``_get_whisper_model()`` which returns the process-wide
    singleton, so all connections share one model with zero extra RAM cost.

    Fix 2 — both partial and final transcription use ``_transcribe_in_thread``
    so the asyncio event loop is never blocked by FasterWhisper inference.
    """

    def __init__(self) -> None:
        # Fix 1 — obtain singleton reference; never construct a new WhisperModel.
        self.model: WhisperModel = _get_whisper_model()
        self.last_partial: str = ""
        self._inference_lock: asyncio.Lock = asyncio.Lock()

        self._bytes_per_second: int = Config.SAMPLE_RATE * 2
        # Use the configured window directly — no artificial cap.
        self._partial_window_bytes: int = (
            self._bytes_per_second * Config.STT_PARTIAL_WINDOW_SECONDS
        )

        logger.info(
            "STT provider: FasterWhisper local (model=%s, lang=%s, window=%ds)",
            Config.LOCAL_WHISPER_MODEL,
            Config.STT_LANGUAGE,
            Config.STT_PARTIAL_WINDOW_SECONDS,
        )

    # -----------------------------------------------------------------------
    # Partial transcription — UI display only, does NOT trigger the LLM
    # -----------------------------------------------------------------------

    async def transcribe_partial(self, audio_bytes: bytes) -> Optional[str]:
        """
        Request a quick partial transcript for real-time UI display.

        Fix 2 — ``_transcribe_in_thread`` is used so the segment generator
                 is exhausted entirely inside the worker thread.
        Issue 9 — validates minimum audio length before calling the model.

        Args:
            audio_bytes: Raw PCM audio sliced to rolling window by caller.

        Returns:
            New transcript string if text changed since last call, else None.
        """
        if self._inference_lock.locked():
            logger.debug("STT partial skipped — inference in progress.")
            return None

        if not _validate_audio(audio_bytes, "partial"):
            return None

        try:
            async with self._inference_lock:
                audio_window = (
                    audio_bytes[-self._partial_window_bytes:]
                    if len(audio_bytes) > self._partial_window_bytes
                    else audio_bytes
                )
                wav_data = _wrap_wav(audio_window)
                audio_buffer = io.BytesIO(wav_data)
                audio_buffer.seek(0)

                start = time.time()

                # Fix 2 — generator exhausted inside the worker thread.
                text = await asyncio.to_thread(
                    _transcribe_in_thread,
                    self.model,
                    audio_buffer,
                    1,                   # beam_size — greedy for speed
                    Config.STT_LANGUAGE, # forced language for stable Hinglish
                )

                elapsed_ms = (time.time() - start) * 1000
                audio_len = len(audio_window)
                logger.info(
                    "STT partial latency: %sms | audio: %d bytes | window: %d bytes (%.2fs)",
                    f"{elapsed_ms:.0f}",
                    audio_len,
                    self._partial_window_bytes,
                    audio_len / self._bytes_per_second,
                )

                if text and text != self.last_partial:
                    logger.debug("STT partial transcript: %s", text)
                    self.last_partial = text
                    return text

                return None

        except Exception:
            logger.exception("STT partial transcription failed.")
            return None

    # -----------------------------------------------------------------------
    # Final transcription — triggers the LLM pipeline
    # -----------------------------------------------------------------------

    async def transcribe_final(self, audio_bytes: bytes) -> Optional[str]:
        """
        Transcribe the complete user utterance with highest accuracy.

        Fix 2 — ``_transcribe_in_thread`` exhausts generator in worker thread.
        Issue 9 — audio validated before the model call.

        Args:
            audio_bytes: Full PCM audio for the completed utterance.

        Returns:
            Transcript string, or None on failure / empty audio.
        """
        if not _validate_audio(audio_bytes, "final"):
            return None

        try:
            async with self._inference_lock:
                wav_data = _wrap_wav(audio_bytes)
                audio_buffer = io.BytesIO(wav_data)
                audio_buffer.seek(0)

                start = time.time()

                # Fix 2 — generator exhausted inside the worker thread.
                text = await asyncio.to_thread(
                    _transcribe_in_thread,
                    self.model,
                    audio_buffer,
                    1,                   # beam_size — greedy for speed
                    Config.STT_LANGUAGE, # forced language for stable Hinglish
                )

                elapsed_ms = (time.time() - start) * 1000
                logger.info(
                    "STT final completed in %sms | length: %d chars | audio: %d bytes (%.2fs)",
                    f"{elapsed_ms:.0f}",
                    len(text),
                    len(audio_bytes),
                    len(audio_bytes) / self._bytes_per_second,
                )
                logger.debug("STT final transcript: %s", text)

                self.last_partial = ""
                return text

        except Exception:
            logger.exception("STT final transcription failed.")
            return None

    def reset(self) -> None:
        """Clear cached partial transcript state between turns."""
        self.last_partial = ""
        logger.debug("STT state reset.")


# ---------------------------------------------------------------------------
# Provider alias
# ---------------------------------------------------------------------------

StreamingSTT = LocalStreamingSTT