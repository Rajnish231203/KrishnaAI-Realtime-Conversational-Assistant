"""
ElevenLabs STT Provider — Krishna Voice Assistant
==================================================
Cloud-based transcription via ElevenLabs Scribe v2.

Role in the provider hierarchy:
    PRIMARY provider — preferred over FasterWhisper for:
      * lower latency on fast hardware / good connectivity
      * superior multilingual accuracy
      * Hindi/Hinglish/English code-switch handling

Design:
    - Uses the official ElevenLabs Python SDK (elevenlabs>=2.0).
        - Both partial and final calls use the synchronous convert() API run
            inside asyncio.to_thread so the event loop is never blocked.
        - This is pseudo-streaming: partials are periodic batch uploads of a
            rolling window, not true websocket streaming.
    - Per-provider async lock prevents concurrent cloud calls on the same
      connection instance (rate-limit safety).
    - All failures are wrapped in STTProviderError so STTManager can catch
      them cleanly and fall back to FasterWhisper without crashing.

Logs are prefixed with [ElevenLabs STT] for easy provider identification.

Future:
    When ElevenLabs releases a stable streaming WebSocket STT API, this
    provider can be upgraded to stream partials incrementally without
    changing the BaseSTTProvider interface.
"""

import asyncio
import io
import logging
import time
import wave
from typing import List, Optional

from backend.app.config.config import Config
from backend.app.services.stt.base_stt import BaseSTTProvider, STTProviderError

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[ElevenLabs STT]"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_MIN_AUDIO_BYTES: int     = int(Config.SAMPLE_RATE * 2 * 0.2)  # 200 ms minimum
_BYTES_PER_SECOND: int    = Config.SAMPLE_RATE * 2
_API_TIMEOUT_SECONDS: int = Config.ELEVENLABS_TIMEOUT_SECONDS

# ElevenLabs Scribe v2 — best multilingual accuracy as of 2025.
# Swap to "scribe_v1" if v2 is unavailable on the account tier.
_SCRIBE_MODEL: str = "scribe_v2"

# BCP-47 / ISO language hint for Hindi.  ElevenLabs uses "hin" for Hindi.
# Set to None to let Scribe auto-detect.
_LANGUAGE_CODE: Optional[str] = "hin"


# ===========================================================================
# WAV helper (shared pattern with FasterWhisperProvider)
# ===========================================================================

def _wrap_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM16 mono audio in a RIFF/WAV container."""
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(Config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()


# ===========================================================================
# Thread worker — runs ElevenLabs SDK call inside asyncio.to_thread
# ===========================================================================

def _call_elevenlabs(client, wav_bytes: bytes) -> str:
    """
    Synchronous ElevenLabs API call (meant for asyncio.to_thread).

    Args:
        client:    Initialised ElevenLabs client.
        wav_bytes: WAV-wrapped PCM16 audio bytes.

    Returns:
        Stripped transcript string (empty on silence).

    Raises:
        Exception: Any SDK / network error — caller wraps in STTProviderError.
    """
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"   # SDK uses the name to infer format

    result = client.speech_to_text.convert(
        file=audio_file,
        model_id=_SCRIBE_MODEL,
        language_code=_LANGUAGE_CODE,
        tag_audio_events=False,     # keep output clean — no [laughter] tags
        diarize=False,              # single-speaker conversational mode
    )
    return (result.text or "").strip()


# ===========================================================================
# Provider
# ===========================================================================

class ElevenLabsProvider(BaseSTTProvider):
    """
    ElevenLabs Scribe v2 STT provider.

    Uses the batch convert() API for both partial and final transcription.
    Calls are offloaded to a thread pool so the asyncio event loop is never
    blocked by network I/O. This is pseudo-streaming (rolling uploads).

    Per-call timeout (_API_TIMEOUT_SECONDS) prevents pipeline stalls when
    ElevenLabs API is slow — STTManager will automatically fall back to
    FasterWhisper in that case.

    Attributes:
        _clients:         Initialised ElevenLabs SDK clients (empty until warmup).
        _api_keys:        Parsed ElevenLabs API keys.
        _current_key_index: Active client index.
        _lock:            Per-instance asyncio.Lock preventing concurrent calls.
        _last_partial:    Deduplication cache for partial transcripts.
        _partial_window:  Max bytes sent for partial inference.
    """

    provider_name = "ElevenLabs"

    def __init__(self) -> None:
        self._api_keys: List[str] = Config.ELEVENLABS_API_KEYS
        self._current_key_index: int = 0
        self._clients: List[object] = []
        self._lock: asyncio.Lock = asyncio.Lock()
        self._last_partial: str = ""
        self._partial_window: int = (
            _BYTES_PER_SECOND * Config.STT_PARTIAL_WINDOW_SECONDS
        )

    # ------------------------------------------------------------------
    # Key/client management
    # ------------------------------------------------------------------

    def _get_current_client(self) -> Optional[object]:
        if not self._clients:
            return None
        if self._current_key_index < 0:
            self._current_key_index = 0
        if self._current_key_index >= len(self._clients):
            self._current_key_index = 0
        return self._clients[self._current_key_index]

    def _rotate_client(self) -> None:
        if not self._clients:
            self._current_key_index = 0
            return
        self._current_key_index = (self._current_key_index + 1) % len(self._clients)
        logger.info(
            "%s Failover successful — now using key #%d",
            _LOG_PREFIX,
            self._current_key_index,
        )

    @staticmethod
    def _should_rotate_provider(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)

        if status_code in (401, 429):
            return True

        message = str(exc).lower()
        keywords = (
            "quota",
            "rate limit",
            "credit",
            "exhausted",
            "unusual activity",
            "unauthorized",
            "insufficient",
        )
        keyword_hit = any(keyword in message for keyword in keywords)
        if keyword_hit:
            return True

        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """
        Validate API keys and initialise the ElevenLabs clients.

        Raises:
            STTProviderError: if ELEVENLABS_API_KEYS are missing or the client
                              cannot be constructed.
        """
        if not self._api_keys:
            raise STTProviderError(
                f"{_LOG_PREFIX} ELEVENLABS_API_KEYS are not set — "
                "provider cannot initialise."
            )
        try:
            from elevenlabs.client import ElevenLabs
            self._clients = [
                ElevenLabs(api_key=key)
                for key in self._api_keys
            ]
            logger.info(
                "%s Provider ready (keys=%d, model=%s, lang=%s, timeout=%ss).",
                _LOG_PREFIX,
                len(self._clients),
                _SCRIBE_MODEL,
                _LANGUAGE_CODE,
                _API_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise STTProviderError(
                f"{_LOG_PREFIX} Client initialisation failed: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Release per-instance state. No persistent connections to close."""
        self._last_partial = ""
        self._clients      = []
        logger.debug("%s Provider shut down.", _LOG_PREFIX)

    # ------------------------------------------------------------------
    # Transcription — partial
    # ------------------------------------------------------------------

    async def transcribe_partial(self, audio_bytes: bytes) -> str:
        """
        Low-latency partial transcript for live UI display.

        Uses the same Scribe API as final transcription — partial vs. final
        is currently a scheduling distinction, not an API distinction.
        Returns empty string if text is unchanged since last call.
        """
        if len(audio_bytes) < _MIN_AUDIO_BYTES:
            return ""

        if self._lock.locked():
            logger.debug("%s Partial skipped — previous call in progress.", _LOG_PREFIX)
            return ""

        if not self._clients:
            raise STTProviderError(f"{_LOG_PREFIX} Provider not warmed up.")

        try:
            async with self._lock:
                window = (
                    audio_bytes[-self._partial_window:]
                    if len(audio_bytes) > self._partial_window
                    else audio_bytes
                )
                wav = _wrap_wav(window)

                total_clients = len(self._clients)
                for attempt in range(total_clients):
                    client = self._get_current_client()
                    if client is None:
                        raise STTProviderError(f"{_LOG_PREFIX} Provider not warmed up.")

                    t0 = time.time()
                    try:
                        text = await asyncio.wait_for(
                            asyncio.to_thread(_call_elevenlabs, client, wav),
                            timeout=_API_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        raise
                    except Exception as exc:
                        if (
                            self._should_rotate_provider(exc)
                            and total_clients > 1
                            and attempt < total_clients - 1
                        ):
                            logger.warning(
                                "%s API key #%d failed — rotating provider.",
                                _LOG_PREFIX,
                                self._current_key_index,
                            )
                            self._rotate_client()
                            continue
                        raise

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

                logger.warning(
                    "%s All ElevenLabs keys failed for partial STT — skipping partial transcript.",
                    _LOG_PREFIX,
                )
                return ""

        except asyncio.TimeoutError:
            logger.warning(
                "%s partial timed out after %ss | window: %.1fs",
                _LOG_PREFIX,
                _API_TIMEOUT_SECONDS,
                len(window) / _BYTES_PER_SECOND,
            )
            raise STTProviderError(
                f"{_LOG_PREFIX} Partial timed out after {_API_TIMEOUT_SECONDS}s."
            )
        except asyncio.CancelledError:
            raise
        except STTProviderError:
            raise
        except Exception as exc:
            raise STTProviderError(
                f"{_LOG_PREFIX} Partial error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Transcription — final
    # ------------------------------------------------------------------

    async def transcribe_final(self, audio_bytes: bytes) -> str:
        """
        Full-utterance transcription — drives the LLM pipeline.

        Sends the complete utterance to Scribe v2 for highest accuracy.
        """
        if len(audio_bytes) < _MIN_AUDIO_BYTES:
            logger.debug("%s Final skipped — audio too short.", _LOG_PREFIX)
            return ""

        if not self._clients:
            raise STTProviderError(f"{_LOG_PREFIX} Provider not warmed up.")

        try:
            async with self._lock:
                wav = _wrap_wav(audio_bytes)

                total_clients = len(self._clients)
                for attempt in range(total_clients):
                    client = self._get_current_client()
                    if client is None:
                        raise STTProviderError(f"{_LOG_PREFIX} Provider not warmed up.")

                    t0 = time.time()
                    try:
                        text = await asyncio.wait_for(
                            asyncio.to_thread(_call_elevenlabs, client, wav),
                            timeout=_API_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        raise
                    except Exception as exc:
                        if (
                            self._should_rotate_provider(exc)
                            and total_clients > 1
                            and attempt < total_clients - 1
                        ):
                            logger.warning(
                                "%s API key #%d failed — rotating provider.",
                                _LOG_PREFIX,
                                self._current_key_index,
                            )
                            self._rotate_client()
                            continue
                        raise

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

                raise STTProviderError(f"{_LOG_PREFIX} All ElevenLabs keys failed.")

        except asyncio.TimeoutError:
            logger.warning(
                "%s final timed out after %ss | audio: %.1fs",
                _LOG_PREFIX,
                _API_TIMEOUT_SECONDS,
                len(audio_bytes) / _BYTES_PER_SECOND,
            )
            raise STTProviderError(
                f"{_LOG_PREFIX} Final timed out after {_API_TIMEOUT_SECONDS}s."
            )
        except asyncio.CancelledError:
            raise
        except STTProviderError:
            raise
        except Exception as exc:
            raise STTProviderError(
                f"{_LOG_PREFIX} Final error: {exc}"
            ) from exc
