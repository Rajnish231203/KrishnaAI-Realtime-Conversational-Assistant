"""
Streaming LLM Service — Krishna Voice Assistant
================================================
Provides token-by-token streaming from Groq or OpenAI with sentence
accumulation for downstream TTS consumption.

Target latency: first token < 300 ms

Public interface (consumed by websocket/server.py):
    StreamingLLM.stream_response(messages)        → AsyncGenerator[str, None]
    StreamingLLM.stream_krishna_response(prompt)  → AsyncGenerator[str, None]

Production notes
----------------
- Issue 1 : System prompt is now injected as the "system" role in every call
            to ``stream_krishna_response`` so the Krishna persona is always
            active.
- Issue 2 : LLM streaming creation wrapped in ``asyncio.wait_for`` to prevent
            a stalled provider from freezing the pipeline.
- Issue 3 : Sentence-split regex tightened to avoid splitting abbreviations
            (e.g. "U.S.A.").
- Issue 4 : End-of-stream summary moved from INFO → DEBUG to reduce log noise.
- Issue 5 : ``max_tokens`` reduced from 1 000 to 250 — appropriate for 3–5
            spoken sentences.
- Issue 6 : Empty-sentence guard verified; no change needed (already present).
- Issue 8 : Exception logging upgraded to ``logger.exception`` to preserve
            full stack traces.
"""

import asyncio
import re
import time
import logging
from typing import AsyncGenerator, Dict, List, Optional

from backend.app.config.config import Config

# ---------------------------------------------------------------------------
# Module-level logger — activated by setup_logging() in the server entrypoint.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM call tuning constants
# ---------------------------------------------------------------------------
LLM_STREAM_TIMEOUT_S:  float = 30.0   # Issue 2 — max seconds to wait for stream start
LLM_MAX_TOKENS_VOICE:  int   = 250    # Issue 5 — enough for 3–5 spoken sentences
LLM_TEMPERATURE:       float = 0.45
LLM_TOP_P:             float = 0.8


# ===========================================================================
# Base Streaming LLM
# ===========================================================================

class StreamingLLM:
    """
    Thin async wrapper around Groq / OpenAI chat-completion streaming.

    Provider selection:
        - Config.USE_GROQ=True  → Groq  (default; faster, higher rate-limits)
        - Config.USE_GROQ=False → OpenAI
    """

    def __init__(self) -> None:
        self.client:   object
        self.model:    str
        self.provider: str

        if Config.USE_GROQ and Config.GROQ_API_KEY:
            from groq import AsyncGroq
            self.client   = AsyncGroq(api_key=Config.GROQ_API_KEY)
            self.model    = Config.GROQ_MODEL
            self.provider = "Groq"
            logger.info("LLM provider: Groq | model: %s", self.model)

        elif Config.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            self.client   = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
            self.model    = Config.OPENAI_MODEL
            self.provider = "OpenAI"
            logger.info("LLM provider: OpenAI | model: %s", self.model)

        else:
            raise RuntimeError(
                "No LLM API key is available. "
                "Set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
            )

    # -----------------------------------------------------------------------
    # Core streaming generator
    # -----------------------------------------------------------------------

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Stream raw tokens from the configured LLM provider.

        Issue 2 — the initial API call is protected by ``asyncio.wait_for``
        so a stalled provider raises ``asyncio.TimeoutError`` rather than
        freezing the pipeline indefinitely.

        Issue 8 — exceptions are logged with ``logger.exception`` to preserve
        the full stack trace in production logs.

        Args:
            messages: OpenAI-style message list
                      [{"role": "system"|"user"|"assistant", "content": "..."}]

        Yields:
            Individual token strings as they arrive from the provider.
        """
        start_time:     float          = time.time()
        first_token_ts: Optional[float] = None
        token_count:    int            = 0

        try:
            # Issue 2 — wrap stream creation in a hard timeout.
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS_VOICE,   # Issue 5 — 250 for voice
                    top_p=LLM_TOP_P,
                ),
                timeout=LLM_STREAM_TIMEOUT_S,
            )

            async for chunk in stream:
                # Guard: skip malformed chunks.
                if not chunk.choices:
                    logger.debug("Skipping chunk with empty choices list.")
                    continue

                delta = chunk.choices[0].delta
                if delta is None or delta.content is None:
                    logger.debug("Skipping chunk with None delta/content.")
                    continue

                token = delta.content

                # First-token latency — log once per stream at INFO.
                if first_token_ts is None:
                    first_token_ts = time.time()
                    latency_ms     = (first_token_ts - start_time) * 1000
                    logger.info(
                        "%s first token latency: %sms",
                        self.provider,
                        f"{latency_ms:.0f}",
                    )

                token_count += 1
                yield token

        except asyncio.TimeoutError:
            logger.warning(
                "%s stream creation timed out after %.1fs.",
                self.provider,
                LLM_STREAM_TIMEOUT_S,
            )
            yield (
                "Dear one, I need a moment to gather my thoughts. "
                "Please ask me again."
            )
            return

        except Exception:
            # Issue 8 — logger.exception preserves full stack trace.
            logger.exception("%s streaming error.", self.provider)
            yield (
                "I apologize, dear one. "
                "I am experiencing technical difficulties at this moment."
            )
            return

        # End-of-stream performance summary.
        # Issue 4 — moved from INFO to DEBUG to reduce per-turn log noise.
        total_ms = (time.time() - start_time) * 1000
        tps      = token_count / (total_ms / 1000) if total_ms > 0 else 0.0
        logger.debug(
            "%s stream complete | tokens: %d | total: %sms | speed: %s tok/s",
            self.provider,
            token_count,
            f"{total_ms:.0f}",
            f"{tps:.1f}",
        )

    # -----------------------------------------------------------------------
    # Non-streaming convenience helper (tests / health-checks)
    # -----------------------------------------------------------------------

    async def get_quick_response(self, user_text: str) -> str:
        """
        Non-streaming single-turn response. Intended for testing only.

        Args:
            user_text: Plain user message.

        Returns:
            Full response string.
        """
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a calm Krishna-inspired conversational guide. "
                    "Respond briefly, clearly, naturally, and conversationally."
                ),
            },
            {"role": "user",   "content": user_text},
        ]

        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=200,
                ),
                timeout=LLM_STREAM_TIMEOUT_S,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            text       = response.choices[0].message.content or ""
            logger.info(
                "%s quick response | %sms | preview: %s",
                self.provider,
                f"{elapsed_ms:.0f}",
                text[:60],
            )
            return text

        except asyncio.TimeoutError:
            logger.warning(
                "%s quick-response timed out after %.1fs.",
                self.provider,
                LLM_STREAM_TIMEOUT_S,
            )
            return "I apologize, dear one. Please try again."

        except Exception:
            # Issue 8 — full traceback preserved.
            logger.exception("%s quick-response error.", self.provider)
            return "I apologize, dear one. Please try again."


# ===========================================================================
# Krishna-specific LLM
# ===========================================================================

class KrishnaLLM(StreamingLLM):
    """
    Krishna persona with optimised prompting and sentence-level streaming.

    Sentence streaming:
        ``stream_krishna_response()`` accumulates raw tokens and yields only
        *complete* sentences so the TTS layer receives speakable chunks
        rather than fragmented mid-word tokens.

    Issue 1 (FIXED):
        ``self.system_prompt`` is now injected as the ``"system"`` role in
        every call to ``stream_krishna_response``, so the Krishna persona
        instruction is always sent to the model.
    """

    # Issue 3 — tightened regex: require whitespace or end-of-string after
    # punctuation so abbreviations like "U.S.A." are NOT split mid-token.
    # Before : r"([.!?।\n]+)"           — splits on any punctuation cluster
    # After  : r'([.!?।]+)(?=\s|$)'    — only splits when followed by space
    #                                     or end-of-string
    _SENTENCE_SPLIT_RE = re.compile(r'([.!?।]+)(?=\s|$)')

    def __init__(self) -> None:
        super().__init__()
        self.system_prompt: str = self._build_system_prompt()
        logger.info("KrishnaLLM initialised with persona system prompt.")

    # -----------------------------------------------------------------------
    # System prompt
    # -----------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """
        Build the Krishna persona system prompt.

        This string is injected as the ``"system"`` role in every
        ``stream_krishna_response`` call to ensure the persona is always
        active. (Issue 1 fix.)

        Returns:
            Fully formatted system prompt string.
        """
        return (
            "You are a calm Krishna-inspired conversational guide rooted in the Bhagavad Gita.\n\n"

            "Communication style:\n"
            "- Be grounded, emotionally intelligent, and practical.\n"
            "- Answer clearly and directly.\n"
            "- Keep responses natural, conversational, and concise.\n"
            "- Avoid theatrical or mystical language.\n"
            "- Avoid repetitive spiritual metaphors or stock phrases.\n"
            "- Do not sound like a motivational quote generator.\n"
            "- Speak with warmth and clarity, not drama.\n\n"

            "The user prompt will provide the detailed grounding and response instructions."
        )

    # -----------------------------------------------------------------------
    # Sentence-level streaming for TTS
    # -----------------------------------------------------------------------

    async def stream_krishna_response(
        self,
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream complete sentences derived from the Krishna RAG prompt.

        Issue 1 (FIXED) — ``self.system_prompt`` is now sent as the
        ``"system"`` role so the Krishna persona is guaranteed to be active
        for every response, not just for the intent-aware path.

        Issue 3 (FIXED) — ``_SENTENCE_SPLIT_RE`` uses a lookahead so that
        abbreviations like "U.S.A." are not incorrectly split mid-token.

        Tokens are accumulated internally and only emitted once a sentence
        boundary ( . ! ? । ) followed by whitespace/end-of-string is
        detected.  Any remaining text after the stream ends is yielded as a
        final sentence.

        Args:
            prompt: Fully-formed RAG prompt (from prompt_builder.py).

        Yields:
            Complete sentence strings, each ready for TTS synthesis.
        """
        # Issue 1 (FIXED) — system prompt injected so persona is always active.
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

        sentence_buffer = ""

        async for token in self.stream_response(messages):
            sentence_buffer += token

            # Only attempt splitting when the token contains potential
            # sentence-boundary punctuation.
            if not any(p in token for p in ".!?।\n"):
                continue

            # Issue 3 — lookahead regex splits only before whitespace / EOS.
            # re.split with a capturing group keeps the delimiter attached:
            #   "Hello. World!"  → ["Hello", ".", " World", "!", ""]
            # With lookahead the delimiter stays in the left part:
            #   "Hello. World!"  → ["Hello.", " World!", ""]
            parts = re.split(r'(?<=[.!?।])(?=\s)', sentence_buffer)

            # Yield all fully-terminated sentences except the last fragment.
            for part in parts[:-1]:
                sentence = part.strip()
                if sentence:
                    logger.debug("Yielding sentence to TTS: %s", sentence[:60])
                    yield sentence

            # Keep trailing incomplete sentence in the buffer.
            sentence_buffer = parts[-1]

        # Flush whatever remains after the stream ends.
        remainder = sentence_buffer.strip()
        if remainder:
            logger.debug("Yielding final buffer to TTS: %s", remainder[:60])
            yield remainder

    # -----------------------------------------------------------------------
    # Intent-aware streaming (optional enhancement path)
    # -----------------------------------------------------------------------

    async def get_intent_aware_response(
        self,
        user_text: str,
        intent: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response with optional intent-based prompt enhancement.

        Note: Currently not called by ``websocket/server.py``; retained as
        an available enhancement path for future intent-routing features.

        Args:
            user_text: Raw user utterance.
            intent:    Detected intent category string (optional).

        Yields:
            Raw token strings.
        """
        enhanced_prompt = self.system_prompt

        if intent:
            intent_guidance: Dict[str, str] = {
                "Career/Purpose":   "Focus on practical purpose, steady effort, and aligned action.",
                "Relationships":    "Emphasise empathy, clear communication, and healthy boundaries.",
                "Inner Conflict":   "Guide toward self-awareness, emotional balance, and clarity.",
                "Life Transitions": "Provide grounded perspective on change and uncertainty.",
                "Daily Struggles":  "Offer practical guidance for everyday challenges.",
            }
            guidance = intent_guidance.get(intent)
            if guidance:
                enhanced_prompt += (
                    f"\n\nContext: This question is about '{intent}'. {guidance}"
                )
                logger.debug("Intent enhancement applied: %s", intent)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user",   "content": user_text},
        ]

        async for token in self.stream_response(messages):
            yield token


# ---------------------------------------------------------------------------
# Module-level alias
# ---------------------------------------------------------------------------
# Re-assign StreamingLLM to the Krishna-specific subclass so that any import
# of ``StreamingLLM`` from this module automatically gets the full persona.
#
#   from backend.app.services.streaming_llm import StreamingLLM
#   llm = StreamingLLM()   # ← actually a KrishnaLLM instance
# ---------------------------------------------------------------------------
StreamingLLM = KrishnaLLM  # type: ignore[misc]