"""
Base STT Provider — Krishna Voice Assistant
============================================
Abstract contract that ALL speech-to-text providers must satisfy.

Design principles:
    - Zero provider-specific logic here.
    - The orchestrator only ever speaks to this interface.
    - Providers are fully interchangeable from the caller's perspective.

Public interface consumed by STTManager (and ultimately the pipeline):
    warmup()             → prepare model/connection at startup
    transcribe_partial() → low-latency partial transcript for UI display
    transcribe_final()   → highest-accuracy transcript that drives the LLM
    shutdown()           → release resources on disconnect
"""

from abc import ABC, abstractmethod


class STTProviderError(Exception):
    """
    Raised by a provider when transcription fails in a known, recoverable way.

    STTManager catches this to decide whether to fall back to the next
    provider.  It must NEVER propagate to the orchestrator or pipeline.

    Provider implementations should wrap all expected SDK/network/model
    failures in STTProviderError so callers can recover cleanly.  Allow
    asyncio.CancelledError to propagate for cooperative cancellation.
    """


class BaseSTTProvider(ABC):
    """
    Abstract base class for all STT providers.

    Subclass and implement the four abstract methods.  No provider-specific
    imports, config, or state belong in this file.

    Lifecycle::

        await provider.warmup()                      # once at startup
        text = await provider.transcribe_partial(...)  # many times per turn
        text = await provider.transcribe_final(...)    # once per turn
        await provider.shutdown()                    # once at connection close

    Thread-safety contract:
        All methods are async coroutines.  Implementations are responsible for
        protecting any shared state (e.g., with asyncio.Lock or threading.Lock
        when calling CPU-bound work via asyncio.to_thread).

    Lifecycle contract:
        warmup() and shutdown() must be idempotent.  They may be called more
        than once due to disconnect/reconnect or retry logic.

    Concurrency contract:
        A single provider instance may receive overlapping partial and final
        requests.  Implementations must enforce per-instance serialization if
        the underlying SDK/model is not re-entrant.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def warmup(self) -> None:
        """
        Prepare the provider for inference.

        Called once per connection at startup.  Should load models,
        validate API keys, open connections, etc.

        Raises:
            STTProviderError: if the provider cannot initialise.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Release provider resources cleanly.

        Called when the WebSocket connection closes.  Must be idempotent
        (safe to call more than once).  Should not raise on repeated calls.
        """

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    @abstractmethod
    async def transcribe_partial(self, audio_bytes: bytes) -> str:
        """
        Return a fast, low-latency partial transcript for UI display.

        Called repeatedly during active speech (every 250–400 ms).
        Accuracy is secondary to latency here.

        Args:
            audio_bytes: Raw PCM16 mono @ 16 kHz audio bytes.
                         May be a rolling window rather than the full buffer.

        Returns:
            Transcript string (may be empty if nothing intelligible detected).

        Raises:
            STTProviderError: on unrecoverable provider failure.
        """

    @abstractmethod
    async def transcribe_final(self, audio_bytes: bytes) -> str:
        """
        Return the highest-accuracy transcript for the completed utterance.

        Called once per turn after endpointing.  Drives the LLM pipeline.

        Args:
            audio_bytes: Full PCM16 mono @ 16 kHz audio for the utterance.

        Returns:
            Transcript string (may be empty on silence or very short audio).

        Raises:
            STTProviderError: on unrecoverable provider failure.
        """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable provider name used in logs."""
        return self.__class__.__name__
