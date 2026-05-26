"""
Streaming Text-to-Speech Service — Krishna Voice Assistant
===========================================================
Provides sentence-by-sentence audio streaming via ElevenLabs (primary)
or OpenAI TTS (fallback).

Target latency: first audio chunk < 500 ms

Public interface (consumed by websocket/server.py):
    StreamingTTS.stream_audio(text)          → AsyncGenerator[bytes, None]
    StreamingTTS._stream_elevenlabs(text)    → AsyncGenerator[bytes, None]
    StreamingTTS._stream_openai(text)        → AsyncGenerator[bytes, None]
    StreamingTTS.generate_full_audio(text)   → Optional[bytes]
    StreamingTTS.close()                     → None  (call on shutdown)

Provider selection:
    ELEVENLABS_API_KEYS present → ElevenLabs REST (PCM 16 kHz)
    Fallback                    → OpenAI TTS (PCM)

The module-level alias ``StreamingTTS`` always points to
``OptimizedStreamingTTS`` which adds LRU caching and long-text splitting.

Production notes
----------------
- Problem 1 : ``close()`` method added; call it on server shutdown to cleanly
              drain the persistent ``httpx.AsyncClient``.
- Problem 2 : ElevenLabs ``Accept`` header corrected from ``audio/mpeg`` to
              ``audio/pcm`` to match the ``pcm_16000`` output format.
- Problem 3 : OpenAI streaming call wrapped in ``asyncio.wait_for`` with a
              30-second hard timeout.
- Problem 4 : Per-sentence completion log moved from INFO → DEBUG; first-chunk
              latency stays at INFO (one line per sentence, genuinely useful).
- Problem 5 : Serialisation behaviour documented; already enforced by the
              single-consumer ``llm_token_buffer`` queue in orchestrator.
- Improvement 1 : ElevenLabs chunks standardised to 4 096 bytes via
                  ``aiter_bytes(chunk_size=4096)``.
- Improvement 2 : Phrase cache capped at ``MAX_CACHE_ITEMS`` with LRU eviction
                  (``collections.OrderedDict``).
- Improvement 3 : Sentence-split regex updated to lookbehind pattern.
- Improvement 4 : Split threshold and cache size read from Config constants.
"""

import asyncio
import collections
import re
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from backend.app.config.config import Config

# ---------------------------------------------------------------------------
# Module-level logger — activated by setup_logging() in the server entrypoint.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTS tuning constants
# ---------------------------------------------------------------------------
TTS_REQUEST_TIMEOUT_S: float = 30.0    # Problem 3 — hard timeout for API calls
TTS_CHUNK_SIZE:        int   = 4096    # Improvement 1 — standardised chunk size

# Improvement 2 — LRU cache cap; oldest entries evicted when limit is reached.
MAX_CACHE_ITEMS: int = 100

# Improvement 4 — split threshold; override via Config if available.
_SPLIT_THRESHOLD: int = getattr(Config, "TTS_SPLIT_THRESHOLD", 2000)
_CACHE_MAX_CHARS: int = 50   # phrases shorter than this are cache-eligible


# ===========================================================================
# Base Streaming TTS
# ===========================================================================

class StreamingTTS:
    """
    Async TTS wrapper that streams PCM audio chunks sentence by sentence.

    ElevenLabs is used when an API key is configured; OpenAI TTS acts as the
    automatic fallback.  A single persistent ``httpx.AsyncClient`` is shared
    across all ElevenLabs requests to maximise connection reuse and reduce
    per-request latency.

    Problem 5 — serialisation note:
        Concurrent TTS requests for the same connection are prevented by the
        single-consumer ``llm_token_buffer`` queue in ``orchestrator``.
        ``process_tts`` always awaits the previous sentence before pulling the
        next one, so this class never needs to manage its own concurrency guard.
    """

    def __init__(self) -> None:
        self.elevenlabs_api_keys: List[str]     = Config.ELEVENLABS_API_KEYS
        self.current_key_index:   int           = 0
        self.elevenlabs_voice_id: str           = Config.ELEVENLABS_VOICE_ID
        self.use_elevenlabs:      bool          = bool(self.elevenlabs_api_keys)
        self.provider:            str           = ""

        # Persistent HTTP client — created once, reused for every ElevenLabs
        # request.  Call ``close()`` on server shutdown to drain connections.
        self.http_client: httpx.AsyncClient = httpx.AsyncClient()

        if Config.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

        if self.use_elevenlabs:
            self.provider = "ElevenLabs"
            logger.info(
                "TTS provider: ElevenLabs | voice: %s | model: %s",
                self.elevenlabs_voice_id,
                Config.ELEVENLABS_MODEL,
            )
        elif Config.OPENAI_API_KEY:
            self.provider      = "OpenAI"
            logger.info("TTS provider: OpenAI TTS (fallback)")
        else:
            logger.error(
                "No TTS provider available. "
                "Set ELEVENLABS_API_KEYS/ELEVENLABS_API_KEY or OPENAI_API_KEY in your .env file."
            )

    # -----------------------------------------------------------------------
    # ElevenLabs key management
    # -----------------------------------------------------------------------

    def _get_current_api_key(self) -> Optional[str]:
        if not self.elevenlabs_api_keys:
            return None
        if self.current_key_index < 0:
            self.current_key_index = 0
        if self.current_key_index >= len(self.elevenlabs_api_keys):
            self.current_key_index = 0
        return self.elevenlabs_api_keys[self.current_key_index]

    def _rotate_api_key(self) -> None:
        if not self.elevenlabs_api_keys:
            self.current_key_index = 0
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.elevenlabs_api_keys)
        logger.info(
            "ElevenLabs TTS failover activated — now using key #%d",
            self.current_key_index,
        )

    @staticmethod
    def _should_rotate_elevenlabs(status_code: int, body_text: str) -> bool:
        if status_code in (401, 429):
            return True
        if status_code not in (402, 403):
            return False
        lowered = body_text.lower()
        keywords = (
            "quota",
            "rate limit",
            "credit",
            "exhausted",
            "unusual activity",
        )
        return any(keyword in lowered for keyword in keywords)

    # -----------------------------------------------------------------------
    # Graceful shutdown
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """
        Problem 1 (FIXED) — cleanly close the persistent HTTP client.

        Call this once during server shutdown to drain open connections and
        avoid resource leaks in long-running services.

        Example (in websocket/server.py main())::

            tts = StreamingTTS()
            try:
                await server.serve_forever()
            finally:
                await tts.close()
        """
        await self.http_client.aclose()
        logger.info("TTS HTTP client closed.")

    # -----------------------------------------------------------------------
    # Public streaming entry-point
    # -----------------------------------------------------------------------

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream PCM audio bytes for the supplied text.

        Selects ElevenLabs when configured, otherwise falls back to OpenAI.
        Empty or whitespace-only text is silently ignored.

        Args:
            text: Sentence or short paragraph to synthesise.

        Yields:
            Raw PCM audio bytes as they arrive from the provider.
        """
        if not text.strip():
            logger.debug("TTS skipping empty text — nothing to synthesise.")
            return

        if self.use_elevenlabs:
            async for chunk in self._stream_elevenlabs(text):
                yield chunk
        elif hasattr(self, "openai_client"):
            async for chunk in self._stream_openai(text):
                yield chunk
        else:
            logger.error("TTS stream_audio called but no provider is initialised.")

    # -----------------------------------------------------------------------
    # ElevenLabs streaming
    # -----------------------------------------------------------------------

    async def _stream_elevenlabs(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream PCM audio from the ElevenLabs REST streaming endpoint.

        Uses the persistent ``self.http_client`` to avoid the overhead of
        creating a new TCP connection for every sentence.

        Problem 1 (FIXED) — client is reused and closed via ``close()``.
        Problem 2 (FIXED) — ``Accept`` header set to ``audio/pcm`` to match
                             the ``pcm_16000`` output format.
        Improvement 1 (FIXED) — chunks standardised to ``TTS_CHUNK_SIZE`` bytes.

        Output format: ``pcm_16000`` (16-bit PCM, 16 kHz, mono).

        Args:
            text: Text to synthesise.

        Yields:
            Raw PCM bytes in standardised ``TTS_CHUNK_SIZE``-byte chunks.
        """
        start_time = time.time()

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech"
            f"/{self.elevenlabs_voice_id}/stream"
        )

        payload = {
            "text":     text,
            "model_id": Config.ELEVENLABS_MODEL,
            "voice_settings": {
                "stability":         Config.ELEVENLABS_STABILITY,
                "similarity_boost":  Config.ELEVENLABS_SIMILARITY,
                "style":             Config.ELEVENLABS_STYLE,
                "use_speaker_boost": Config.ELEVENLABS_SPEAKER_BOOST,
            },
        }

        params = {"output_format": "pcm_16000"}

        try:
            if not self.elevenlabs_api_keys:
                raise RuntimeError("ElevenLabs API keys are not configured.")

            total_keys = len(self.elevenlabs_api_keys)

            for attempt in range(total_keys):
                api_key = self._get_current_api_key()
                if not api_key:
                    raise RuntimeError("ElevenLabs API key is missing.")

                # Problem 2 (FIXED) — Accept matches the requested pcm_16000 format.
                headers: Dict[str, str] = {
                    "Accept":       "audio/pcm",           # was: "audio/mpeg" — mismatch fixed
                    "Content-Type": "application/json",
                    "xi-api-key":   api_key,
                }

                first_chunk = True
                chunk_count = 0

                async with self.http_client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    params=params,
                    timeout=TTS_REQUEST_TIMEOUT_S,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_text = error_body.decode(errors="replace")
                        if (
                            self._should_rotate_elevenlabs(response.status_code, error_text)
                            and total_keys > 1
                            and attempt < total_keys - 1
                        ):
                            logger.warning(
                                "ElevenLabs TTS key #%d failed with %s — rotating to next key.",
                                self.current_key_index,
                                response.status_code,
                            )
                            self._rotate_api_key()
                            continue

                        raise RuntimeError(
                            f"ElevenLabs API error {response.status_code}: "
                            f"{error_text}"
                        )

                    # Improvement 1 — standardised chunk size for consistent
                    # downstream buffering and WebSocket frame sizes.
                    async for chunk in response.aiter_bytes(chunk_size=TTS_CHUNK_SIZE):
                        if first_chunk:
                            latency_ms = (time.time() - start_time) * 1000
                            # Problem 4 — first-chunk latency stays at INFO (useful).
                            logger.info(
                                "TTS first audio chunk latency: %sms",
                                f"{latency_ms:.0f}",
                            )
                            first_chunk = False

                        chunk_count += 1
                        logger.debug("TTS ElevenLabs chunk #%d received.", chunk_count)
                        yield chunk

                total_ms = (time.time() - start_time) * 1000
                # Problem 4 (FIXED) — completion moved from INFO → DEBUG.
                logger.debug(
                    "TTS ElevenLabs complete | chunks: %d | total: %sms",
                    chunk_count,
                    f"{total_ms:.0f}",
                )
                return

            raise RuntimeError("ElevenLabs API keys exhausted.")

        except Exception:
            logger.exception("ElevenLabs streaming error.")
            # Attempt OpenAI fallback so the pipeline keeps running.
            if hasattr(self, "openai_client"):
                logger.info("TTS falling back to OpenAI TTS after ElevenLabs error.")
                async for chunk in self._stream_openai(text):
                    yield chunk

    # -----------------------------------------------------------------------
    # OpenAI TTS streaming
    # -----------------------------------------------------------------------

    async def _stream_openai(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream PCM audio from the OpenAI TTS API.

        Problem 3 (FIXED) — the streaming call is now wrapped in
        ``asyncio.wait_for`` with a ``TTS_REQUEST_TIMEOUT_S`` hard timeout so
        a stalled OpenAI response never freezes the pipeline.

        Uses the ``tts-1`` model with the ``onyx`` voice (deep, authoritative)
        and raw PCM output for lowest client-side decoding overhead.

        Args:
            text: Text to synthesise.

        Yields:
            Raw PCM bytes in ``TTS_CHUNK_SIZE``-byte chunks.
        """
        start_time = time.time()

        try:
            first_chunk = True
            chunk_count = 0

            # Problem 3 (FIXED) — hard timeout prevents pipeline freeze.
            streaming_response = await asyncio.wait_for(
                self.openai_client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="onyx",
                    input=text,
                    response_format="pcm",
                    speed=1.0,
                ),
                timeout=TTS_REQUEST_TIMEOUT_S,
            )

            async with streaming_response as response:
                async for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                    if first_chunk:
                        latency_ms = (time.time() - start_time) * 1000
                        # Problem 4 — first-chunk stays at INFO.
                        logger.info(
                            "TTS first audio chunk latency: %sms",
                            f"{latency_ms:.0f}",
                        )
                        first_chunk = False

                    chunk_count += 1
                    logger.debug("TTS OpenAI chunk #%d received.", chunk_count)
                    yield chunk

            total_ms = (time.time() - start_time) * 1000
            # Problem 4 (FIXED) — completion moved from INFO → DEBUG.
            logger.debug(
                "TTS OpenAI complete | chunks: %d | total: %sms",
                chunk_count,
                f"{total_ms:.0f}",
            )

        except asyncio.TimeoutError:
            logger.warning(
                "OpenAI TTS timed out after %.1fs — sentence skipped.",
                TTS_REQUEST_TIMEOUT_S,
            )
        except Exception:
            logger.exception("OpenAI TTS streaming error.")

    # -----------------------------------------------------------------------
    # Non-streaming full audio generation (tests / health-checks)
    # -----------------------------------------------------------------------

    async def generate_full_audio(self, text: str) -> Optional[bytes]:
        """
        Generate a complete audio buffer without streaming (non-real-time).

        Intended for unit tests and health-check endpoints only.

        Args:
            text: Text to synthesise.

        Returns:
            Complete PCM audio bytes, or None on failure.
        """
        try:
            if self.use_elevenlabs:
                chunks: List[bytes] = []
                async for chunk in self._stream_elevenlabs(text):
                    chunks.append(chunk)
                return b"".join(chunks)

            elif hasattr(self, "openai_client"):
                response = await asyncio.wait_for(
                    self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice="onyx",
                        input=text,
                    ),
                    timeout=TTS_REQUEST_TIMEOUT_S,
                )
                return response.content

            return None

        except asyncio.TimeoutError:
            logger.warning("generate_full_audio timed out.")
            return None
        except Exception:
            logger.exception("TTS generate_full_audio error.")
            return None


# ===========================================================================
# Optimized TTS — LRU caching + long-text splitting
# ===========================================================================

class OptimizedStreamingTTS(StreamingTTS):
    """
    Production-optimised TTS layer built on top of ``StreamingTTS``.

    Enhancements:
        1. **LRU phrase cache** — common short phrases are stored as raw PCM
           bytes after first synthesis. Cache is capped at ``MAX_CACHE_ITEMS``
           with oldest-entry eviction to prevent memory growth.
           (Improvement 2 FIXED.)
        2. **Long-text splitting** — text longer than ``_SPLIT_THRESHOLD``
           characters is split on sentence boundaries before synthesis,
           keeping individual API requests small and first-chunk latency low.
           (Improvement 4 FIXED — threshold read from Config.)
        3. **Improved split regex** — lookbehind pattern avoids splitting on
           mid-word periods. (Improvement 3 FIXED.)
    """

    def __init__(self) -> None:
        super().__init__()
        # Improvement 2 (FIXED) — OrderedDict as a bounded LRU cache.
        # When the cap is reached the oldest (first-inserted) entry is evicted.
        self._cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        logger.debug(
            "OptimizedStreamingTTS initialised "
            "(LRU cache cap: %d, split threshold: %d chars).",
            MAX_CACHE_ITEMS,
            _SPLIT_THRESHOLD,
        )

    # -----------------------------------------------------------------------
    # Public streaming entry-point (overrides base)
    # -----------------------------------------------------------------------

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream PCM audio with LRU caching and automatic long-text splitting.

        Workflow:
            1. Return cached bytes immediately for known short phrases (O(1)).
            2. Split text > ``_SPLIT_THRESHOLD`` chars into sentences and
               stream each chunk individually.
            3. For short uncached text, synthesise normally and cache the
               result if it qualifies (≤ ``_CACHE_MAX_CHARS`` chars).

        Cache eviction:
            When the cache reaches ``MAX_CACHE_ITEMS``, the oldest entry is
            removed before inserting the new one (LRU via ``OrderedDict``).

        Args:
            text: Sentence or paragraph to synthesise.

        Yields:
            Raw PCM audio bytes.
        """
        text = text.strip()

        if not text:
            logger.debug("OptimizedStreamingTTS skipping empty text.")
            return

        # --- 1. LRU cache hit -------------------------------------------
        if text in self._cache:
            logger.debug("TTS cache hit: %s…", text[:30])
            # Move to end to mark as recently used.
            self._cache.move_to_end(text)
            yield self._cache[text]
            return

        # --- 2. Long-text: split and stream sentence by sentence ----------
        if len(text) > _SPLIT_THRESHOLD:
            logger.debug(
                "TTS text exceeds %d chars — splitting into sentences.",
                _SPLIT_THRESHOLD,
            )
            for sentence in self._split_into_sentences(text):
                async for chunk in super().stream_audio(sentence):
                    yield chunk
            return

        # --- 3. Short text: synthesise and optionally cache ---------------
        audio_chunks: List[bytes] = []

        async for chunk in super().stream_audio(text):
            audio_chunks.append(chunk)
            yield chunk

        if len(text) <= _CACHE_MAX_CHARS and audio_chunks:
            # Improvement 2 (FIXED) — evict oldest entry if cap is reached.
            if len(self._cache) >= MAX_CACHE_ITEMS:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("TTS cache evicted oldest entry: %s…", evicted_key[:30])

            self._cache[text] = b"".join(audio_chunks)
            logger.debug(
                "TTS cached phrase (%d items in cache): %s…",
                len(self._cache),
                text[:30],
            )

    # -----------------------------------------------------------------------
    # Sentence splitter helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split a long string into speakable sentence-sized chunks.

        Improvement 3 (FIXED) — uses a lookbehind assertion so the split
        only occurs immediately after sentence-ending punctuation followed
        by whitespace, avoiding false splits on abbreviations (e.g. "U.S.A.").

        Sentences are greedily combined up to ``_SPLIT_THRESHOLD`` characters
        to balance chunk size against API request overhead.

        Args:
            text: Long text string.

        Returns:
            List of sentence strings, each ≤ ~``_SPLIT_THRESHOLD`` characters.
        """
        # Improvement 3 (FIXED) — lookbehind keeps the delimiter attached to
        # the preceding sentence and avoids splitting mid-abbreviation.
        raw_parts: List[str] = re.split(r'(?<=[.!?।])\s+', text)

        sentences: List[str] = []
        current: str = ""

        for part in raw_parts:
            if len(current) + len(part) < _SPLIT_THRESHOLD:
                current += (" " if current else "") + part
            else:
                if current:
                    sentences.append(current.strip())
                current = part

        if current:
            sentences.append(current.strip())

        return sentences


# ---------------------------------------------------------------------------
# Module-level alias
# ---------------------------------------------------------------------------
# Re-assign StreamingTTS to the optimised subclass so that any import of
# ``StreamingTTS`` from this module automatically gets LRU caching and
# sentence splitting.
#
#   from backend.app.services.streaming_tts import StreamingTTS
#   tts = StreamingTTS()   # ← actually an OptimizedStreamingTTS instance
# ---------------------------------------------------------------------------
StreamingTTS = OptimizedStreamingTTS  