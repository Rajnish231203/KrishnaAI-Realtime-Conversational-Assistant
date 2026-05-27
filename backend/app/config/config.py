"""
Configuration Management for Krishna Voice Assistant
=====================================================
Production-ready configuration module.

Usage:
    from app.config.config import Config

Environment:
    All settings are controlled via environment variables or a .env file.
    See .env.example for the full list of supported variables.

Sections:
    - API Keys
    - Model Configuration
    - Audio Settings
    - RAG Configuration
    - Security Settings
    - Conversation Settings
    - Feature Flags
"""

import os
import logging
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment Loading
# ---------------------------------------------------------------------------
# Must run before Config class is evaluated so all os.getenv() calls
# can read values from the .env file.
load_dotenv()


# ---------------------------------------------------------------------------
# Structured Logging Setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """
    Configure application-wide structured logging.

    This function should be called ONCE from the server entrypoint
    (e.g. main() in websocket/server.py), NOT on module import.

    Returns:
        logging.Logger: Root logger configured with a structured format.

    Example:
        from app.config.config import setup_logging
        logger = setup_logging()
        logger.info("Server starting...")
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("krishna")


# ---------------------------------------------------------------------------
# Configuration Class
# ---------------------------------------------------------------------------

class Config:
    """
    Central configuration for the Krishna Voice Assistant pipeline.

    All values are read from environment variables with sensible defaults.
    Call Config.validate() on startup to catch mis-configurations early.
    """

    # -----------------------------------------------------------------------
    # API KEYS
    # -----------------------------------------------------------------------

    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY:   str | None = os.getenv("GROQ_API_KEY")
    ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_API_KEYS: list[str] = [
        key.strip()
        for key in os.getenv("ELEVENLABS_API_KEYS", "").split(",")
        if key.strip()
    ]
    if not ELEVENLABS_API_KEYS and ELEVENLABS_API_KEY:
        ELEVENLABS_API_KEYS = [ELEVENLABS_API_KEY]

    # -----------------------------------------------------------------------
    # MODEL CONFIGURATION
    # -----------------------------------------------------------------------

    # LLM provider — set USE_GROQ=false in .env to force OpenAI
    USE_GROQ: bool = os.getenv("USE_GROQ", "True").lower() == "true"

    # STT provider — set USE_GROQ_STT=true in .env to use Groq Whisper
    USE_GROQ_STT: bool = os.getenv("USE_GROQ_STT", "False").lower() == "true"

    # OpenAI models
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Groq models (faster inference, higher rate limits)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Whisper STT model (used by both OpenAI and Groq paths)
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")

    # Local Whisper (FasterWhisper) settings
    LOCAL_WHISPER_MODEL: str = os.getenv("LOCAL_WHISPER_MODEL", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")

    # Language forced into FasterWhisper for both partial and final transcription.
    # 'hi' stabilises Hinglish / code-switch decoding.  Set to None to re-enable
    # autodetection (useful if the assistant handles pure English sessions).
    STT_LANGUAGE: str | None = os.getenv("STT_LANGUAGE", "hi") or None

    # ElevenLabs TTS
    # Recommended voice IDs for Krishna persona:
    #   pNInz6obpgDQGcFmaJgB — Adam  (deep, authoritative)  ← default
    #   onwK4e9ZLuTAKqWW03F9 — Daniel (calm, wise, male)
    #   21m00Tcm4TlvDq8ikWAM — Rachel (warm, gentle, female)
    ELEVENLABS_VOICE_ID: str = os.getenv(
        "ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB"
    )
    ELEVENLABS_MODEL: str = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

    # ElevenLabs voice tuning — balanced for a steady, wise spiritual persona
    ELEVENLABS_STABILITY:   float = float(os.getenv("ELEVENLABS_STABILITY",   "0.6"))
    ELEVENLABS_SIMILARITY:  float = float(os.getenv("ELEVENLABS_SIMILARITY",  "0.85"))
    ELEVENLABS_STYLE:       float = float(os.getenv("ELEVENLABS_STYLE",       "0.5"))
    ELEVENLABS_SPEAKER_BOOST: bool = (
        os.getenv("ELEVENLABS_SPEAKER_BOOST", "True").lower() == "true"
    )

    # ElevenLabs STT request timeout (seconds) to avoid pipeline stalls.
    ELEVENLABS_TIMEOUT_SECONDS: int = int(
        os.getenv("ELEVENLABS_TIMEOUT_SECONDS", "10")
    )

    # -----------------------------------------------------------------------
    # AUDIO SETTINGS
    # -----------------------------------------------------------------------

    # Input audio capture settings (must match the client-side recorder)
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))   # Hz
    CHANNELS:    int = int(os.getenv("CHANNELS",    "1"))        # Mono

    # Silence detection: trigger final STT after this many milliseconds of silence
    SILENCE_THRESHOLD: int = int(os.getenv("SILENCE_THRESHOLD", "500"))  # ms

    # STT partial-transcript tuning
    # How often (ms) to request a partial transcript from the STT engine
    STT_PARTIAL_INTERVAL_MS: int = int(
        os.getenv("STT_PARTIAL_INTERVAL_MS", "300")
    )
    # Rolling window (seconds) sent to STT for partial transcripts.
    # 3 s gives Hinglish/Hindi conversational pacing enough context for
    # stable partial decoding without excessive compute.
    STT_PARTIAL_WINDOW_SECONDS: int = int(
        os.getenv("STT_PARTIAL_WINDOW_SECONDS", "3")
    )

    # STT provider abstraction layer (backend/app/services/stt/)
    # 'elevenlabs'    → ElevenLabs Scribe v2 primary, FasterWhisper fallback
    # 'faster_whisper' → FasterWhisper only (offline / no API key required)
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "elevenlabs")

    # When True, STTManager will try FasterWhisper if the primary fails.
    STT_FALLBACK_ENABLED: bool = (
        os.getenv("STT_FALLBACK_ENABLED", "true").lower() == "true"
    )

    # -----------------------------------------------------------------------
    # SERVER SETTINGS
    # -----------------------------------------------------------------------

    WEBSOCKET_HOST: str = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    WEBSOCKET_PORT: int = int(os.getenv("WEBSOCKET_PORT", "8766"))
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8000"))

    # -----------------------------------------------------------------------
    # RAG CONFIGURATION
    # -----------------------------------------------------------------------

    # Number of Gita verses to retrieve per query
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

    # Minimum cosine-similarity score to include a verse in context
    # Range 0.0–1.0; lower values → broader retrieval, higher → stricter
    RAG_SIMILARITY_THRESHOLD: float = float(
        os.getenv("RAG_SIMILARITY_THRESHOLD", "0.29")
    )

    # -----------------------------------------------------------------------
    # CONVERSATION SETTINGS
    # -----------------------------------------------------------------------

    # Maximum number of (user + assistant) messages kept in rolling history
    # Older messages are dropped to stay within LLM context limits
    MAX_CONVERSATION_HISTORY: int = int(
        os.getenv("MAX_CONVERSATION_HISTORY", "20")
    )

    # -----------------------------------------------------------------------
    # VOICE ACTIVITY DETECTION (VAD)
    # -----------------------------------------------------------------------

    # Master switch — set VAD_ENABLED=false to bypass Silero entirely.
    VAD_ENABLED: bool = os.getenv("VAD_ENABLED", "true").lower() == "true"

    # Probability threshold above which a frame is classified as speech (0.0–1.0).
    VAD_SPEECH_THRESHOLD: float = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.5"))

    # Minimum continuous speech duration (ms) before speech start is declared.
    VAD_MIN_SPEECH_MS: int = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))

    # Minimum continuous silence duration (ms) before speech end is declared.
    VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS", "450"))

    # VAD frame size in ms (Silero works with 10 / 20 / 30 ms frames).
    VAD_FRAME_MS: int = int(os.getenv("VAD_FRAME_MS", "32"))

    # Enable per-frame DEBUG logging (verbose — disable in production).
    VAD_DEBUG: bool = os.getenv("VAD_DEBUG", "false").lower() == "true"

    # -----------------------------------------------------------------------
    # SECURITY SETTINGS
    # -----------------------------------------------------------------------

    # Enable WebSocket authentication middleware (future feature)
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "True").lower() == "true"

    # Maximum requests per minute per connected client
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # Enable content-moderation check on user input before LLM call
    ENABLE_MODERATION: bool = (
        os.getenv("ENABLE_MODERATION", "True").lower() == "true"
    )

    # -----------------------------------------------------------------------
    # FEATURE FLAGS
    # -----------------------------------------------------------------------

    # Redis connection string for future distributed session / rate-limit state
    # If left as default the system runs in single-process (non-distributed) mode
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------

    @classmethod
    def validate(cls) -> bool:
        """
        Validate the active configuration on startup.

        Checks:
            - At least one LLM provider API key is present.
            - If USE_GROQ is True, the Groq key must be present.
            - If ElevenLabs is configured, its key must be present.
            - Audio values are within sane operating ranges.

        Returns:
            bool: True when all checks pass.

        Raises:
            ValueError: With a descriptive message on the first failed check.
        """
        _log = logging.getLogger("krishna.config")

        # --- LLM provider ---------------------------------------------------
        if not (cls.OPENAI_API_KEY or cls.GROQ_API_KEY):
            raise ValueError(
                "Configuration error: at least one LLM API key is required. "
                "Set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
            )

        if cls.USE_GROQ and not cls.GROQ_API_KEY:
            raise ValueError(
                "Configuration error: USE_GROQ is True but GROQ_API_KEY is "
                "not set. Either set GROQ_API_KEY or set USE_GROQ=false."
            )

        # --- TTS provider ---------------------------------------------------
        # ElevenLabs is optional; OpenAI TTS is used as fallback.
        # Only raise if neither key is available at all.
        if not cls.ELEVENLABS_API_KEYS and not cls.OPENAI_API_KEY:
            raise ValueError(
                "Configuration error: no TTS provider available. "
                "Set ELEVENLABS_API_KEYS/ELEVENLABS_API_KEY (preferred) or "
                "OPENAI_API_KEY (fallback)."
            )

        # --- STT provider ---------------------------------------------------
        # Non-crashing checks.  Missing ELEVENLABS_API_KEY is handled at
        # warmup() time: STTManager automatically falls back to FasterWhisper.
        _valid_stt_providers = ("elevenlabs", "faster_whisper", "whisper", "local")
        if cls.STT_PROVIDER.strip().lower() not in _valid_stt_providers:
            raise ValueError(
                f"Configuration error: STT_PROVIDER='{cls.STT_PROVIDER}' is not "
                f"recognised. Valid options: {_valid_stt_providers}."
            )

        _stt_primary = cls.STT_PROVIDER.strip().lower()
        if _stt_primary == "elevenlabs" and not cls.ELEVENLABS_API_KEYS:
            _log.warning(
                "⚠️  STT_PROVIDER=elevenlabs but ELEVENLABS_API_KEYS are not set. "
                "STTManager will fall back to FasterWhisper automatically at warmup."
            )

        _fallback_str = "FasterWhisper (auto)" if cls.STT_FALLBACK_ENABLED else "none"
        _log.info(
            "STT topology: primary=%s | fallback=%s | timeout=%ss",
            cls.STT_PROVIDER,
            _fallback_str,
            cls.ELEVENLABS_TIMEOUT_SECONDS,
        )

        # --- Audio sanity checks --------------------------------------------
        if cls.SAMPLE_RATE not in (8000, 16000, 22050, 44100, 48000):
            raise ValueError(
                f"Configuration error: SAMPLE_RATE={cls.SAMPLE_RATE} is not a "
                "standard value. Use one of: 8000, 16000, 22050, 44100, 48000."
            )

        if cls.CHANNELS not in (1, 2):
            raise ValueError(
                f"Configuration error: CHANNELS must be 1 (mono) or 2 (stereo), "
                f"got {cls.CHANNELS}."
            )

        if not (100 <= cls.SILENCE_THRESHOLD <= 3000):
            raise ValueError(
                f"Configuration error: SILENCE_THRESHOLD={cls.SILENCE_THRESHOLD}ms "
                "is outside the safe range of 100–3000 ms."
            )

        # --- RAG sanity checks ----------------------------------------------
        if not (1 <= cls.RAG_TOP_K <= 20):
            raise ValueError(
                f"Configuration error: RAG_TOP_K={cls.RAG_TOP_K} must be "
                "between 1 and 20."
            )

        if not (0.0 <= cls.RAG_SIMILARITY_THRESHOLD <= 1.0):
            raise ValueError(
                f"Configuration error: RAG_SIMILARITY_THRESHOLD="
                f"{cls.RAG_SIMILARITY_THRESHOLD} must be between 0.0 and 1.0."
            )

        # --- Conversation history -------------------------------------------
        if cls.MAX_CONVERSATION_HISTORY < 2:
            raise ValueError(
                f"Configuration error: MAX_CONVERSATION_HISTORY="
                f"{cls.MAX_CONVERSATION_HISTORY} must be at least 2."
            )

        _log.info(
            "✅ Configuration validated successfully. "
            "STT=%s | LLM=%s | TTS=%s",
            cls.STT_PROVIDER,
            "Groq" if cls.USE_GROQ else "OpenAI",
            "ElevenLabs" if cls.ELEVENLABS_API_KEYS else "OpenAI",
        )
        return True