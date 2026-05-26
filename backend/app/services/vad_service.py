"""
VAD Service — Krishna Voice Assistant
======================================
Realtime speech-activity detection using Silero VAD.

Responsibility:
    Observe whether human speech is currently active.
    Nothing else.

This service does NOT:
    - transcribe audio
    - detect language
    - finalize turns
    - own orchestration state

It ONLY:
    - processes PCM16 audio frames (size driven by Config.VAD_FRAME_MS)
    - estimates speech probability via Silero VAD
    - applies debounce  (N consecutive speech  frames → speech started)
    - applies hysteresis (N consecutive silence frames → speech ended)
    - exposes the current speech-active state

Audio format expected:
    PCM16 mono @ 16 kHz  (same stream fed to FasterWhisper)

Silero VAD requirements:
    - mono, 16 kHz
    - float32 normalized to [-1.0, 1.0]

Future consumers:
    endpointing, barge-in, turn finalization, conversational timing.
"""

import logging
import threading
from typing import Optional

import numpy as np
import torch

from backend.app.config.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derived frame constants — computed once from Config so .env changes take
# effect without touching this file.
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000

# Number of samples in one VAD frame.
FRAME_SAMPLES: int = SAMPLE_RATE * Config.VAD_FRAME_MS // 1000
# Number of PCM16 bytes in one VAD frame (2 bytes per int16 sample).
FRAME_BYTES: int = FRAME_SAMPLES * 2

# Consecutive speech  frames needed to declare speech started (debounce).
# e.g. VAD_MIN_SPEECH_MS=250, VAD_FRAME_MS=30 → ceil(250/30) = 9 frames
SPEECH_DEBOUNCE_FRAMES: int = max(1, Config.VAD_MIN_SPEECH_MS // Config.VAD_FRAME_MS)

# Consecutive silence frames needed to declare speech ended (hysteresis).
# e.g. VAD_MIN_SILENCE_MS=450, VAD_FRAME_MS=30 → 15 frames
SILENCE_HYSTERESIS_FRAMES: int = max(1, Config.VAD_MIN_SILENCE_MS // Config.VAD_FRAME_MS)

SPEECH_PROB_THRESHOLD: float = Config.VAD_SPEECH_THRESHOLD
_VAD_DEBUG: bool = Config.VAD_DEBUG


# ===========================================================================
# Silero model loader  (module-level singleton — loaded once per process)
# ===========================================================================

_silero_model = None
_silero_lock  = threading.Lock()


def _load_silero():
    """
    Load the Silero VAD model exactly once (thread-safe double-checked locking).

    Uses the installed ``silero-vad`` package directly — no GitHub/torch.hub
    network dependency at runtime.

    Returns:
        Silero VAD model (CPU, eval mode).

    Raises:
        RuntimeError: if the model cannot be loaded.
    """
    global _silero_model

    if _silero_model is not None:
        return _silero_model

    with _silero_lock:
        if _silero_model is not None:   # re-check under lock
            return _silero_model

        logger.info("VAD: Loading Silero VAD model (first call)…")
        try:
            from silero_vad import load_silero_vad  # installed package — no network call
            model = load_silero_vad()
            model.to("cpu")             # explicit CPU — prevents accidental CUDA init
            model.eval()
            _silero_model = model
            logger.info("VAD: Silero VAD model loaded successfully (CPU).")
        except Exception as exc:
            raise RuntimeError(f"VAD: Failed to load Silero model: {exc}") from exc

    return _silero_model


# ===========================================================================
# VAD Service
# ===========================================================================

class VADService:
    """
    Realtime speech-activity detector backed by Silero VAD.

    Lightweight, isolated, orchestration-independent.

    Usage::

        vad = VADService()
        for pcm16_chunk in audio_stream:
            for result in vad.feed(pcm16_chunk):
                if result.speech_started:
                    ...  # first frame of a new speech segment
                if result.speech_ended:
                    ...  # speech just stopped

    Thread safety:
        ``feed()`` must be called from a single thread (the audio ingestion
        thread).  Internal Silero model inference is guarded by a per-instance
        Lock for safety if ever called concurrently.
    """

    def __init__(self) -> None:
        # Load (or reuse) the shared Silero model.
        try:
            self._model = _load_silero()
            self._available: bool = True
            logger.info("VADService initialised (Silero ready).")
        except RuntimeError as exc:
            # Silero unavailable — service degrades gracefully.
            self._model     = None
            self._available = False
            logger.error("VADService: Silero unavailable — VAD disabled. %s", exc)

        # Per-call inference lock.
        self._lock: threading.Lock = threading.Lock()

        # --- Speech-activity state ----------------------------------------
        self._speech_active:         bool = False
        self._consec_speech_frames:  int  = 0
        self._consec_silence_frames: int  = 0

        # --- Frame buffer (accumulate partial chunks until a full frame) ----
        self._frame_buffer: bytes = b""

        # --- Statistics (for log summaries) --------------------------------
        self._total_frames:  int = 0
        self._speech_frames: int = 0

        logger.info(
            "VADService config | frame=%dms | debounce=%d frames (%dms) | "
            "hysteresis=%d frames (%dms) | threshold=%.2f",
            Config.VAD_FRAME_MS,
            SPEECH_DEBOUNCE_FRAMES,  Config.VAD_MIN_SPEECH_MS,
            SILENCE_HYSTERESIS_FRAMES, Config.VAD_MIN_SILENCE_MS,
            SPEECH_PROB_THRESHOLD,
        )

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def feed(self, pcm16_bytes: bytes) -> list["VADResult"]:
        """
        Feed raw PCM16 bytes into the VAD.

        Buffers partial data internally and processes complete frames as they
        accumulate.  A single call may produce zero, one, or several results.

        Args:
            pcm16_bytes: Raw PCM16 mono @ 16 kHz audio bytes.

        Returns:
            List of ``VADResult`` — one per complete frame processed.
            Empty list if no complete frame was available yet.
        """
        if not self._available:
            return []

        self._frame_buffer += pcm16_bytes
        results = []

        while len(self._frame_buffer) >= FRAME_BYTES:
            frame              = self._frame_buffer[:FRAME_BYTES]
            self._frame_buffer = self._frame_buffer[FRAME_BYTES:]
            result             = self._process_frame(frame)
            if result is not None:
                results.append(result)

        return results

    @property
    def is_speech_active(self) -> bool:
        """Current (debounced + hysteresis-filtered) speech activity state."""
        return self._speech_active

    @property
    def is_available(self) -> bool:
        """False if Silero failed to load; VAD is a no-op in that case."""
        return self._available

    def reset(self) -> None:
        """
        Reset all state for a new conversation turn.

        Clears the frame buffer, resets counters, and resets Silero's
        internal LSTM state so the next turn starts clean.
        """
        self._frame_buffer           = b""
        self._speech_active          = False
        self._consec_speech_frames   = 0
        self._consec_silence_frames  = 0
        self._total_frames           = 0
        self._speech_frames          = 0
        self._reset_model_state()
        logger.debug("VADService: state reset for new turn.")

    # -----------------------------------------------------------------------
    # Internal frame processing
    # -----------------------------------------------------------------------

    def _process_frame(self, pcm16_frame: bytes) -> Optional["VADResult"]:
        """
        Run Silero on one PCM16 frame and update speech activity state.

        Returns:
            ``VADResult`` describing this frame, or ``None`` on inference error.
        """
        # 1. PCM16 → float32 normalized tensor --------------------------------
        if len(pcm16_frame) < FRAME_BYTES:
            return None
        audio_int16   = np.frombuffer(pcm16_frame, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor        = torch.from_numpy(audio_float32)

        # 2. Silero inference -------------------------------------------------
        try:
            with self._lock:
                speech_prob = self._model(tensor, SAMPLE_RATE).item()
        except Exception as exc:
            # Short-frame errors are expected occasionally in realtime audio streams.
            # Downgrade to debug to avoid terminal spam.
            if "Input audio chunk is too short" in str(exc):
                logger.debug("VAD short frame skipped.")
            else:
                logger.warning("VAD inference error (frame skipped): %s", exc)

            return None

        # 3. Frame statistics -------------------------------------------------
        self._total_frames += 1
        is_speech_frame     = speech_prob >= SPEECH_PROB_THRESHOLD

        if is_speech_frame:
            self._speech_frames        += 1
            self._consec_speech_frames  += 1
            self._consec_silence_frames  = 0
        else:
            self._consec_silence_frames += 1
            self._consec_speech_frames   = 0

        # 4. State transitions (debounce + hysteresis) -----------------------
        speech_started = False
        speech_ended   = False

        if not self._speech_active:
            # Transition IN: N consecutive speech frames required.
            if self._consec_speech_frames >= SPEECH_DEBOUNCE_FRAMES:
                self._speech_active = True
                speech_started      = True
                logger.info(
                    "VAD: ▶ Speech STARTED  (prob=%.3f | consec=%d frames | %dms)",
                    speech_prob,
                    self._consec_speech_frames,
                    self._consec_speech_frames * Config.VAD_FRAME_MS,
                )
        else:
            # Transition OUT: N consecutive silence frames required.
            if self._consec_silence_frames >= SILENCE_HYSTERESIS_FRAMES:
                self._speech_active = False
                speech_ended        = True
                speech_ratio = (
                    self._speech_frames / self._total_frames
                    if self._total_frames else 0.0
                )
                logger.info(
                    "VAD: ■ Speech ENDED    (prob=%.3f | silence=%dms | "
                    "speech_ratio=%.1f%%)",
                    speech_prob,
                    self._consec_silence_frames * Config.VAD_FRAME_MS,
                    speech_ratio * 100,
                )

        # 5. Per-frame debug log (guarded — disable in production) ------------
        if _VAD_DEBUG:
            logger.debug(
                "VAD frame | prob=%.3f | speech=%s | consec_speech=%d | "
                "consec_silence=%d | active=%s",
                speech_prob,
                is_speech_frame,
                self._consec_speech_frames,
                self._consec_silence_frames,
                self._speech_active,
            )

        return VADResult(
            speech_probability = speech_prob,
            is_speech_frame    = is_speech_frame,
            speech_active      = self._speech_active,
            speech_started     = speech_started,
            speech_ended       = speech_ended,
        )

    def _reset_model_state(self) -> None:
        """Reset Silero's internal LSTM state between turns."""
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass  # safe no-op if the installed version lacks reset_states()


# ===========================================================================
# Result dataclass
# ===========================================================================

class VADResult:
    """
    Per-frame result returned by ``VADService.feed()``.

    Attributes:
        speech_probability: Raw Silero probability for this frame (0.0–1.0).
        is_speech_frame:    True if ``speech_probability >= threshold``.
        speech_active:      Current debounced speech-active state.
        speech_started:     True only on the frame where speech officially began.
        speech_ended:       True only on the frame where speech officially ended.
    """

    __slots__ = (
        "speech_probability",
        "is_speech_frame",
        "speech_active",
        "speech_started",
        "speech_ended",
    )

    def __init__(
        self,
        speech_probability: float,
        is_speech_frame:    bool,
        speech_active:      bool,
        speech_started:     bool,
        speech_ended:       bool,
    ) -> None:
        self.speech_probability = speech_probability
        self.is_speech_frame    = is_speech_frame
        self.speech_active      = speech_active
        self.speech_started     = speech_started
        self.speech_ended       = speech_ended

    def __repr__(self) -> str:
        return (
            f"VADResult(prob={self.speech_probability:.3f}, "
            f"active={self.speech_active}, "
            f"started={self.speech_started}, "
            f"ended={self.speech_ended})"
        )
