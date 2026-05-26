"""
FasterWhisper STT Provider — Krishna Voice Assistant
=====================================================
Local CPU-based transcription via FasterWhisper.

Role in the provider hierarchy:
    Fallback provider — used when ElevenLabs is unavailable or fails.
    Also usable as the sole provider in offline / no-API-key mode.

Design:
    - Adapts the proven singleton + thread-offload pattern from streaming_stt.py.
    - Fully standalone: works without any cloud dependency.
    - Language hint is configurable; use None for auto-detect.
    - condition_on_previous_text=False prevents hallucinated continuations.

Logs are prefixed with [FasterWhisper STT] for easy provider identification
in multi-provider deployments.
"""

import asyncio
import io
import logging
import threading
import time
import wave
from typing import Optional

from faster_whisper import WhisperModel

from backend.app.config.config import Config
from backend.app.services.stt.base_stt import BaseSTTProvider, STTProviderError

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[FasterWhisper STT]"

# ---------------------------------------------------------------------------
# Audio validation constant
# ---------------------------------------------------------------------------
# Reject buffers shorter than 200 ms — too short to contain meaningful speech.
_MIN_AUDIO_BYTES: int = int(Config.SAMPLE_RATE * 2 * 0.2)
_BYTES_PER_SECOND: int = Config.SAMPLE_RATE * 2


# ===========================================================================
# Process-level singleton — shared across ALL provider instances / connections
# ===========================================================================
# Loading WhisperModel costs ~150 MB RAM and several seconds on CPU.
# One model per process keeps memory flat regardless of connection count.
# Double-checked locking guarantees at-most-one initialisation under concurrency.

_model_lock:    threading.Lock         = threading.Lock()
_whisper_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Return the process-wide WhisperModel singleton (thread-safe)."""
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                logger.info(
                    "%s Loading model '%s' on %s (singleton — first call).",
                    _LOG_PREFIX,
                    Config.LOCAL_WHISPER_MODEL,
                    Config.WHISPER_DEVICE,
                )
                _whisper_model = WhisperModel(
                    Config.LOCAL_WHISPER_MODEL,
                    device=Config.WHISPER_DEVICE,
                    compute_type="int8_float32",
                    cpu_threads=4,
                )
                logger.info("%s Singleton model loaded and ready.", _LOG_PREFIX)
    return _whisper_model


async def prewarm_model() -> None:
    """
    Pre-load the FasterWhisper singleton at server startup.

    Call this once from the server entrypoint (before accepting connections)
    so the first user utterance is not penalised by the 2–8 s model load time.

    This is the ONLY sanctioned external entry-point for triggering the
    singleton load.  It replaces the old ``_get_whisper_model`` import from
    ``streaming_stt.py`` — that module must never be imported in active
    runtime code.

    Usage (server.py)::

        from backend.app.services.stt.faster_whisper_provider import prewarm_model
        await prewarm_model()
    """
    logger.info(
        "%s Pre-warming singleton model at server startup "
        "(model=%s, device=%s)…",
        _LOG_PREFIX,
        Config.LOCAL_WHISPER_MODEL,
        Config.WHISPER_DEVICE,
    )
    await asyncio.to_thread(_get_model)
    logger.info("%s Pre-warm complete — singleton model is hot and ready.", _LOG_PREFIX)


# ===========================================================================
# Thread worker — exhausts generator INSIDE the worker thread
# ===========================================================================
# FasterWhisper returns a lazy generator.  Iterating it on the asyncio event
# loop blocks the loop.  This helper runs entirely inside asyncio.to_thread.

def _run_transcription(
    model:       WhisperModel,
    wav_bytes:   bytes,
    beam_size:   int,
    language:    Optional[str],
) -> str:
    """
    Run FasterWhisper inference synchronously (meant for asyncio.to_thread).

    Args:
        model:     Shared WhisperModel singleton.
        wav_bytes: Complete WAV-wrapped audio bytes (RIFF header + PCM data).
        beam_size: Beam width — 1 (greedy/fast) for partial, higher for final.
        language:  ISO 639-1 code or None for auto-detect.

    Returns:
        Whitespace-stripped transcript (empty string if silence/noise only).
    """
    buf = io.BytesIO(wav_bytes)
    buf.seek(0)
    segments, _ = model.transcribe(
        buf,
        beam_size=beam_size,
        language=language,
        # Prevents hallucinated continuations bleeding across pause boundaries.
        condition_on_previous_text=False,
    )
    return " ".join(seg.text for seg in segments).strip()


# ===========================================================================
# WAV helper
# ===========================================================================

def _wrap_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM16 mono audio in a RIFF/WAV container for Whisper."""
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(Config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()


# ===========================================================================
# Provider
# ===========================================================================

class FasterWhisperProvider(BaseSTTProvider):
    """
    Local FasterWhisper STT provider.

    Singleton model + asyncio.to_thread offloading keeps the event loop
    free during CPU-bound inference (typically 200 ms – 4 s on CPU).

    The lazy segment generator is exhausted entirely inside the worker
    thread to prevent accidental event-loop blocking.

    Attributes:
        _inference_lock: Per-instance asyncio.Lock.  Prevents concurrent
                 transcription calls on the same provider instance
                 (the underlying model is not re-entrant).
        _last_partial:   Deduplication cache — avoids sending unchanged
                         partial transcripts to the pipeline.
    """

    provider_name = "FasterWhisper"

    def __init__(self) -> None:
        self._model:          Optional[WhisperModel] = None
        self._inference_lock: asyncio.Lock           = asyncio.Lock()
        self._last_partial:   str                    = ""

        # Rolling partial window: send at most N seconds to avoid O(n²) work.
        self._partial_window_bytes: int = (
            _BYTES_PER_SECOND * Config.STT_PARTIAL_WINDOW_SECONDS
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Load (or reuse) the singleton model at connection startup."""
        try:
            self._model = _get_model()
            logger.info(
                "%s Provider ready (model=%s, lang=%s, window=%ds).",
                _LOG_PREFIX,
                Config.LOCAL_WHISPER_MODEL,
                Config.STT_LANGUAGE,
                Config.STT_PARTIAL_WINDOW_SECONDS,
            )
        except Exception as exc:
            raise STTProviderError(
                f"{_LOG_PREFIX} Model load failed: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Nothing to release — singleton model lives for the process lifetime."""
        self._last_partial = ""
        logger.debug("%s Provider shut down (per-instance state cleared).", _LOG_PREFIX)

    # ------------------------------------------------------------------
    # Transcription — partial
    # ------------------------------------------------------------------

    async def transcribe_partial(self, audio_bytes: bytes) -> str:
        """
        Fast partial transcript using greedy (beam_size=1) decoding.

        Skips inference if a previous call is still running (backpressure).
        Returns empty string if text is unchanged since last call.
        """
        if len(audio_bytes) < _MIN_AUDIO_BYTES:
            return ""

        if self._inference_lock.locked():
            logger.debug("%s Partial skipped — inference in progress.", _LOG_PREFIX)
            return ""

        try:
            async with self._inference_lock:
                window = (
                    audio_bytes[-self._partial_window_bytes:]
                    if len(audio_bytes) > self._partial_window_bytes
                    else audio_bytes
                )
                wav = _wrap_wav(window)
                t0  = time.time()

                text = await asyncio.to_thread(
                    _run_transcription,
                    self._model,
                    wav,
                    1,                   # beam_size — greedy for speed
                    Config.STT_LANGUAGE,
                )

                elapsed_ms = (time.time() - t0) * 1000
                logger.info(
                    "%s partial latency: %dms | window: %.1fs | bytes: %d",
                    _LOG_PREFIX,
                    elapsed_ms,
                    len(window) / _BYTES_PER_SECOND,
                    len(window),
                )

                if text and text != self._last_partial:
                    self._last_partial = text
                    return text
                return ""

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise STTProviderError(
                f"{_LOG_PREFIX} Partial transcription error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Transcription — final
    # ------------------------------------------------------------------

    async def transcribe_final(self, audio_bytes: bytes) -> str:
        """
        Full-utterance transcription using greedy decoding (beam_size=1).

        Uses beam_size=1 for realtime conversational latency.
        Accuracy vs. latency tradeoff: acceptable for conversational UX.
        """
        if len(audio_bytes) < _MIN_AUDIO_BYTES:
            logger.debug("%s Final skipped — audio too short.", _LOG_PREFIX)
            return ""

        try:
            async with self._inference_lock:
                wav = _wrap_wav(audio_bytes)
                t0  = time.time()

                text = await asyncio.to_thread(
                    _run_transcription,
                    self._model,
                    wav,
                    1,                   # beam_size — greedy; raise to 3 for accuracy
                    Config.STT_LANGUAGE,
                )

                elapsed_ms = (time.time() - t0) * 1000
                logger.info(
                    "%s final latency: %dms | audio: %.1fs | bytes: %d | chars: %d",
                    _LOG_PREFIX,
                    elapsed_ms,
                    len(audio_bytes) / _BYTES_PER_SECOND,
                    len(audio_bytes),
                    len(text),
                )

                self._last_partial = ""
                return text

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise STTProviderError(
                f"{_LOG_PREFIX} Final transcription error: {exc}"
            ) from exc
