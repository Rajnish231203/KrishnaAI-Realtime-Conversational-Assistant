"""
STT Manager — Krishna Voice Assistant
======================================
Provider orchestration layer for speech-to-text.

The orchestrator and stt_pipeline see ONLY this class.
They call:

    await stt.transcribe_partial(audio_bytes)
    await stt.transcribe_final(audio_bytes)

They never know which provider is active, whether it is cloud or local,
or whether a fallback happened.

Provider selection (config-driven):
    STT_PROVIDER=elevenlabs   → ElevenLabs primary, FasterWhisper fallback
    STT_PROVIDER=faster_whisper → FasterWhisper only (no cloud dependency)

Fallback policy:
    Per-request.  If the primary provider raises STTProviderError on a
    given call, the fallback is tried immediately on the SAME request.
    No circuit breaker.  No permanent disabling.  Simple and reliable.

    Rationale: circuit breakers are valuable for high-throughput services.
    For a conversational assistant seeing ≤ 1 turn/second, per-request
    fallback is simpler, more transparent, and equally effective.
"""

import logging
from typing import Optional

from backend.app.config.config import Config
from backend.app.services.stt.base_stt import BaseSTTProvider, STTProviderError
from backend.app.services.stt.faster_whisper_provider import FasterWhisperProvider

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[STTManager]"


def _build_provider(name: str) -> BaseSTTProvider:
    """
    Instantiate a provider by name string.

    Args:
        name: 'elevenlabs' or 'faster_whisper' (case-insensitive).

    Returns:
        An uninitialised BaseSTTProvider subclass instance.

    Raises:
        ValueError: if name is not recognised.
    """
    key = name.strip().lower()
    if key == "elevenlabs":
        from backend.app.services.stt.elevenlabs_provider import ElevenLabsProvider
        return ElevenLabsProvider()
    if key in ("faster_whisper", "whisper", "local"):
        return FasterWhisperProvider()
    raise ValueError(
        f"{_LOG_PREFIX} Unknown STT provider '{name}'. "
        "Valid options: 'elevenlabs', 'faster_whisper'."
    )


class STTManager:
    """
    Provider-routing façade for STT.

    The orchestrator creates one STTManager per connection and calls
    warmup() once.  All subsequent transcription calls go through
    transcribe_partial / transcribe_final.

    Attributes:
        _primary:  The configured primary provider instance.
        _fallback: The fallback provider instance (FasterWhisper, always local).
                   None when primary IS FasterWhisper (no point falling back
                   to the same provider).
    """

    def __init__(self) -> None:
        primary_name: str = getattr(Config, "STT_PROVIDER", "faster_whisper")
        fallback_enabled: bool = getattr(Config, "STT_FALLBACK_ENABLED", True)

        self._fallback_count: int = 0

        self._primary: BaseSTTProvider = _build_provider(primary_name)

        # The fallback is always FasterWhisper (local, zero-dependency).
        # Skip creating a second instance when primary is already FasterWhisper.
        self._fallback: Optional[BaseSTTProvider] = None
        if fallback_enabled and not isinstance(self._primary, FasterWhisperProvider):
            self._fallback = FasterWhisperProvider()

        logger.info(
            "%s primary=%s | fallback=%s",
            _LOG_PREFIX,
            self._primary.provider_name,
            self._fallback.provider_name if self._fallback else "none",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """
        Warm up all configured providers.

        If the primary fails to warm up, the fallback is promoted to primary
        automatically so the rest of the connection is unaffected.
        """
        try:
            await self._primary.warmup()
            logger.info("%s Primary warmup OK: %s", _LOG_PREFIX, self._primary.provider_name)
        except STTProviderError as exc:
            logger.warning(
                "%s Primary warmup failed (%s). %s",
                _LOG_PREFIX,
                exc,
                "Falling back to local provider."
                if self._fallback else "No fallback available.",
            )
            if self._fallback:
                self._primary, self._fallback = self._fallback, None
                await self._primary.warmup()
                logger.info("%s Primary promoted to fallback: %s", _LOG_PREFIX, self._primary.provider_name)
            else:
                raise

        if self._fallback:
            try:
                await self._fallback.warmup()
                logger.info("%s Fallback warmup OK: %s", _LOG_PREFIX, self._fallback.provider_name)
            except STTProviderError as exc:
                logger.warning(
                    "%s Fallback warmup failed (%s). "
                    "Fallback will not be available this session.",
                    _LOG_PREFIX,
                    exc,
                )
                self._fallback = None

    async def shutdown(self) -> None:
        """Shut down all active providers."""
        for provider in filter(None, [self._primary, self._fallback]):
            try:
                await provider.shutdown()
            except Exception as exc:
                logger.debug(
                    "%s Shutdown error for %s: %s",
                    _LOG_PREFIX,
                    provider.provider_name,
                    exc,
                )

    # ------------------------------------------------------------------
    # Transcription — public interface
    # ------------------------------------------------------------------

    async def transcribe_partial(self, audio_bytes: bytes) -> str:
        """
        Return a partial transcript via primary → fallback routing.

        Returns:
            Transcript string (empty string on silence or total failure).
        """
        return await self._call_with_fallback(
            "partial",
            audio_bytes,
        )

    async def transcribe_final(self, audio_bytes: bytes) -> str:
        """
        Return a final transcript via primary → fallback routing.

        Returns:
            Transcript string (empty string on silence or total failure).
        """
        return await self._call_with_fallback(
            "final",
            audio_bytes,
        )

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    async def _call_with_fallback(
        self,
        call_type:   str,
        audio_bytes: bytes,
    ) -> str:
        """
        Route a transcription request through primary → fallback.

        Args:
            call_type:   'partial' or 'final' — used for logging and dispatch.
            audio_bytes: Raw PCM16 audio bytes.

        Returns:
            Transcript string, or empty string on total failure.
        """
        method = (
            self._primary.transcribe_partial
            if call_type == "partial"
            else self._primary.transcribe_final
        )

        try:
            return await method(audio_bytes)

        except STTProviderError as primary_err:
            if self._fallback is None:
                logger.warning(
                    "%s %s %s failed (no fallback): %s",
                    _LOG_PREFIX,
                    self._primary.provider_name,
                    call_type,
                    primary_err,
                )
                return ""

            logger.warning(
                "%s %s %s failed — falling back to %s. Error: %s",
                _LOG_PREFIX,
                self._primary.provider_name,
                call_type,
                self._fallback.provider_name,
                primary_err,
            )

            self._fallback_count += 1
            logger.info(
                "%s Fallback activated (%d total).",
                _LOG_PREFIX,
                self._fallback_count,
            )

            fallback_method = (
                self._fallback.transcribe_partial
                if call_type == "partial"
                else self._fallback.transcribe_final
            )

            try:
                return await fallback_method(audio_bytes)
            except STTProviderError as fallback_err:
                logger.error(
                    "%s Fallback %s also failed: %s",
                    _LOG_PREFIX,
                    self._fallback.provider_name,
                    fallback_err,
                )
                return ""
            except Exception as exc:
                logger.exception(
                    "%s Unexpected fallback error: %s", _LOG_PREFIX, exc
                )
                return ""

        except Exception as exc:
            logger.exception(
                "%s Unexpected primary error: %s", _LOG_PREFIX, exc
            )
            return ""

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        """Name of the currently active primary provider."""
        return self._primary.provider_name

    @property
    def fallback_provider(self) -> Optional[str]:
        """Name of the fallback provider, or None if none is configured."""
        return self._fallback.provider_name if self._fallback else None
