"""
Streaming Orchestrator -- Krishna Voice Assistant.

Stateful per-connection coordinator for the realtime pipeline.
"""

import asyncio
import base64
import json
import logging
import time
from enum import Enum
from typing import Dict, List

import websockets
from websockets.server import WebSocketServerProtocol

from backend.app.config.config import Config
from backend.app.core.orchestrator.llm_pipeline import process_llm
from backend.app.core.orchestrator.stt_pipeline import process_stt
from backend.app.core.orchestrator.tts_pipeline import process_tts

# STEP 2 — Added SWITCH_CONVERSATION to the protocol import.
from backend.app.core.websocket.protocol import (
    AUDIO_CHUNK,
    END_OF_SPEECH,
    INTERRUPT,
    CHAT_MESSAGE,
    SWITCH_CONVERSATION,
)

from backend.app.services.rag.retriever import GitaRetriever
from backend.app.services.streaming_llm import StreamingLLM
from backend.app.services.stt.stt_manager import STTManager
from backend.app.services.streaming_tts import StreamingTTS
from backend.app.services.vad_service import VADService


logger = logging.getLogger("backend.app.core.streaming_server")

# ---------------------------------------------------------------------------
# Pipeline timing constants
# ---------------------------------------------------------------------------
SILENCE_MIN_CHECK_S = 0.05  # minimum delay between silence checks


class SpeechState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    USER_PAUSED = "user_paused"
    FINALIZING = "finalizing"
    THINKING = "thinking"
    ASSISTANT_SPEAKING = "assistant_speaking"
    INTERRUPTED = "interrupted"
    RESETTING = "resetting"


# ===========================================================================
# Streaming Orchestrator -- ONE INSTANCE PER CONNECTION
# ===========================================================================

class StreamingOrchestrator:
    """
    Stateful orchestrator that manages the full voice pipeline for a **single**
    WebSocket connection.

    Issue 1 (FIXED) -- Shared state across clients:
        ``StreamingOrchestrator`` is instantiated inside ``connection_handler``
        for every new connection, so all instance state is fully isolated per
        user. No state leaks between concurrent clients.

    Pipeline stages run as parallel asyncio tasks connected by in-memory
    queues so each stage begins processing as soon as data is available:

        receive_audio  ->[audio_buffer]->  process_stt
                                              |
                                      [transcript_buffer]
                                              |
                                         process_llm
                                              |
                                      [llm_token_buffer]
                                              |
                                         process_tts  ->  WebSocket audio
    """

    def __init__(self) -> None:
        # --- Service clients (one set per connection) --------------------
        self.stt: STTManager = STTManager()
        self.llm: StreamingLLM = StreamingLLM()
        self.tts: StreamingTTS = StreamingTTS()
        self.retriever: GitaRetriever = GitaRetriever()

        # --- Phase 2: VAD (one instance per connection) ------------------
        # VADService is a lightweight wrapper; the underlying Silero model
        # is a process-level singleton so this costs virtually nothing.
        self.vad: VADService = VADService() if Config.VAD_ENABLED else None
        # True while Silero reports the user is actively speaking (debounced).
        self._vad_speech_active: bool = False

        # --- Turn state --------------------------------------------------
        self._state: SpeechState = SpeechState.IDLE
        self._state_changed_at: float = time.monotonic()
        self._turn_owner_id: int = 0
        self._turn_txn_id: int = 0

        self._audio_activity_event: asyncio.Event = asyncio.Event()
        self._endpoint_request_event: asyncio.Event = asyncio.Event()
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Issue 8 (FIXED) — replace raw bool flag with asyncio.Event for
        # interrupt signalling so it is safe across concurrent tasks.
        self.interrupt_event: asyncio.Event = asyncio.Event()

        # ------------------------------------------------------------------
        # STEP 1 — Multi-conversation in-memory session state.
        #
        # Replaces the single flat `conversation_history` list so each
        # frontend conversation gets its own isolated history container.
        # All histories live in memory for the duration of the WebSocket
        # session and are discarded when the connection closes.
        #
        # conversation_id -> list of {"role": ..., "content": ...} dicts
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

        # Currently active voice conversation.
        # Voice streaming still assumes one active voice session at a time,
        # so we track which conversation ID owns the current voice turn.
        self.active_conversation_id: str | None = None
        # Frozen ownership for the currently active voice turn.
        # Prevents conversation switching from mutating ownership
        # mid-STT/finalization lifecycle.
        self.current_turn_conversation_id: str | None = None
        # ------------------------------------------------------------------

        self.last_audio_chunk_at: float = time.time()
        self._last_audio_monotonic: float = time.monotonic()
        self.current_turn_id: int = 0
        self.processed_turn_id: int = -1
        self._final_sent_for_turn: int = -1

        self.processing_lock: asyncio.Lock = asyncio.Lock()
        self.is_processing: bool = False

        # EOS / finalization guards (Fix 5 + Bug 2/3/4 fixes)
        # _eos_fired    : atomic gate — only first EOS path enqueues None sentinel
        # _turn_finalizing: broader flag readable by stt_pipeline to stop partials
        #                   the moment finalization begins, not just after STT ends
        self._eos_fired:       asyncio.Event = asyncio.Event()
        self._turn_finalizing: bool          = False

        # --- Per-turn timing metrics -------------------------------------
        self._turn_start_time: float = 0.0
        self._chunk_count: int = 0
        self._last_partial_stt_at: float = 0.0
        self.metrics: Dict[str, float] = self._blank_metrics()

        logger.info("StreamingOrchestrator created for new connection.")

    @property
    def state(self) -> SpeechState:
        return self._state

    def _set_state(self, new_state: SpeechState, reason: str) -> None:
        if self._state == new_state:
            return
        prev = self._state
        self._state = new_state
        self._state_changed_at = time.monotonic()
        logger.debug("State transition: %s -> %s (%s)", prev.value, new_state.value, reason)

    def _next_txn(self, reason: str) -> int:
        self._turn_txn_id += 1
        logger.debug("Turn transaction advanced to %d (%s)", self._turn_txn_id, reason)
        return self._turn_txn_id

    def _invalidate_turn(self, reason: str) -> None:
        self._next_txn(reason)
        self.interrupt_event.set()
        self._set_state(SpeechState.INTERRUPTED, reason)
        self._eos_fired.clear()
        self._turn_finalizing = False
        self._final_sent_for_turn = self.current_turn_id
        self._audio_activity_event.clear()
        self._endpoint_request_event.clear()
        logger.info("Turn %d invalidated (%s).", self.current_turn_id, reason)

    def _request_finalization(self, reason: str) -> bool:
        self._endpoint_request_event.set()
        if self._begin_turn_finalization():
            logger.debug("Finalization requested (%s).", reason)
            return True
        logger.debug("Finalization request ignored (%s).", reason)
        return False

    @property
    def is_speaking(self) -> bool:
        return self._state == SpeechState.ASSISTANT_SPEAKING

    @is_speaking.setter
    def is_speaking(self, value: bool) -> None:
        if value:
            self._set_state(SpeechState.ASSISTANT_SPEAKING, "tts_start")
        else:
            if self._state == SpeechState.ASSISTANT_SPEAKING:
                self._set_state(SpeechState.IDLE, "tts_end")

    # -----------------------------------------------------------------------
    # WebSocket entry-point
    # -----------------------------------------------------------------------

    async def handle_client(
        self,
        websocket: WebSocketServerProtocol,
        path: str,
    ) -> None:
        """
        Run the full pipeline for one WebSocket connection.

        Five parallel asyncio tasks are started:
            1. receive_audio   -- ingest base64 PCM chunks
            2. monitor_silence -- auto-trigger final STT after silence
            3. process_stt     -- partial + final transcription
            4. process_llm     -- RAG retrieval + LLM sentence streaming
            5. process_tts     -- TTS synthesis + audio streaming

        Issue 4 (FIXED) — cancelled tasks are now properly awaited via
        ``asyncio.gather`` to prevent "Task was destroyed but pending" warnings.
        """
        logger.info("Client connected: %s", websocket.remote_address)

        try:
            await self.stt.warmup()
        except Exception as exc:
            logger.exception("STT warmup failed; continuing without crash: %s", exc)

        # Fix 4 — bounded queues prevent unbounded memory growth when a
        # downstream stage (STT / LLM / TTS) is slower than its upstream
        # producer.  maxsize=100 provides ~4 s of audio back-pressure at
        # 40 ms chunks before the receive loop naturally throttles.
        audio_buffer:      asyncio.Queue = asyncio.Queue(maxsize=100)
        transcript_buffer: asyncio.Queue = asyncio.Queue(maxsize=100)
        llm_token_buffer:  asyncio.Queue = asyncio.Queue(maxsize=100)

        tasks = [
            asyncio.create_task(
                self.receive_audio(websocket, audio_buffer, transcript_buffer),
                name="receive_audio",
            ),
            asyncio.create_task(
                self.monitor_silence(audio_buffer, websocket),
                name="monitor_silence",
            ),
            asyncio.create_task(
                process_stt(self, audio_buffer, transcript_buffer, websocket),
                name="process_stt",
            ),
            asyncio.create_task(
                process_llm(self, transcript_buffer, llm_token_buffer, websocket),
                name="process_llm",
            ),
            asyncio.create_task(
                process_tts(self, llm_token_buffer, websocket),
                name="process_tts",
            ),
        ]

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

            # Issue 4 (FIXED) — await cancellation so tasks can clean up
            # and Python does not emit "Task destroyed but it is pending!".
            await asyncio.gather(*pending, return_exceptions=True)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected: %s", websocket.remote_address)
        except Exception as exc:
            logger.exception("Unhandled error in handle_client: %s", exc)
        finally:
            self._shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.stt.shutdown()
            except Exception as exc:
                logger.debug("STT shutdown error: %s", exc)
            logger.info("Connection closed: %s", websocket.remote_address)

    # -----------------------------------------------------------------------
    # Stage 1 — Audio ingestion
    # -----------------------------------------------------------------------

    async def receive_audio(
        self,
        websocket: WebSocketServerProtocol,
        audio_buffer: asyncio.Queue,
        transcript_buffer: asyncio.Queue,
    ) -> None:
        """
        Receive base64-encoded PCM audio chunks and forward raw bytes to the
        STT queue.

        ``audio_chunk``
            Decoded and queued for STT. Dropped while TTS is playing.

        ``interrupt``
            Barge-in handler. Sets ``interrupt_event``, flushes the audio
            buffer, then clears ``interrupt_event`` after the flush so
            subsequent turns are not affected.

        ``end_of_speech``
            Enqueues a ``None`` sentinel to trigger final transcription.

        ``switch_conversation``  [STEP 3 — NEW]
            Updates the active conversation ID so subsequent voice turns
            and LLM context lookups use the correct history container.

        ``chat_message``  [STEP 4 — updated]
            Now carries a conversation_id field for proper memory routing.
        """
        logger.info("Audio receiver started.")

        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == AUDIO_CHUNK:
                    if self.is_speaking:
                        # Assistant is speaking — feed VAD anyway so barge-in
                        # detection has up-to-date speech probability.
                        if self.vad:
                            audio_bytes_vad = base64.b64decode(data["audio"])
                            self._process_vad(audio_bytes_vad)
                        continue

                    if self.metrics["audio_received_at"] == 0:
                        # Bug 4 fix — centralized new-turn state reset.
                        self._begin_new_turn()

                    self._chunk_count += 1
                    self.last_audio_chunk_at = time.time()
                    self._last_audio_monotonic = time.monotonic()
                    self._audio_activity_event.set()
                    if self.state in (SpeechState.IDLE, SpeechState.LISTENING, SpeechState.USER_PAUSED):
                        self._set_state(SpeechState.USER_SPEAKING, "audio_chunk")

                    if self._chunk_count % 25 == 0:
                        logger.debug(
                            "Turn %d: %d audio chunks received.",
                            self.current_turn_id,
                            self._chunk_count,
                        )

                    audio_bytes = base64.b64decode(data["audio"])

                    # --- Phase 2: feed every chunk into VAD ---------------
                    if self.vad:
                        self._process_vad(audio_bytes)

                    await audio_buffer.put(audio_bytes)

                elif msg_type == INTERRUPT:
                    logger.info("Barge-in interrupt received.")

                    # Issue 8 (FIXED) — signal via Event (task-safe).
                    self._invalidate_turn("barge_in")

                    while not audio_buffer.empty():
                        audio_buffer.get_nowait()

                    self._reset_metrics()
                    logger.debug("Interrupt handled — state reset.")

                elif msg_type == END_OF_SPEECH:
                    logger.info("End-of-speech signal received.")
                    # Bug 2/3 fix — delegate to centralized finalization helper.
                    # _begin_turn_finalization() sets all finalization state
                    # atomically and returns True only for the first caller.
                    if self._request_finalization("client_eos"):
                        await audio_buffer.put(None)

                elif msg_type == CHAT_MESSAGE:
                    logger.info("Chat message received: %s", data.get("text", ""))
                    # Increment turn ID for typed input
                    self.current_turn_id += 1

                    # STEP 4A — Read conversation_id alongside the text.
                    conversation_id = data.get("conversation_id")
                    chat_text = data.get("text", "").strip()

                    if not chat_text:
                        logger.debug("Empty chat message — skipped.")
                        continue

                    # STEP 4B — Validate that a conversation_id was supplied.
                    # Without it we cannot route memory correctly.
                    if not conversation_id:
                        logger.debug("Missing conversation_id in chat message.")
                        continue

                    # STEP 4C — Ensure an isolated history container exists
                    # for this conversation, then mark it as the active one.
                    if conversation_id not in self.conversations:
                        self.conversations[conversation_id] = []

                    # Typed chat becomes the active conversation so that any
                    # immediately following voice turn inherits the same context.
                    self.active_conversation_id = conversation_id

                    # STEP 4D — Include conversation_id in the transcript dict
                    # so llm_pipeline.py can route to the correct history.
                    transcript = {
                        "type": "final",
                        "text": chat_text,
                        "turn_id": self.current_turn_id,
                        "timestamp": time.time(),
                        "source": "text",
                        "conversation_id": conversation_id,   # ← NEW
                    }
                    await transcript_buffer.put(transcript)
                    logger.debug(
                        "Enqueued chat transcript for turn %d (conversation: %s).",
                        self.current_turn_id,
                        conversation_id,
                    )

                # STEP 3 — Handle conversation switch events sent by the
                # frontend when the user clicks a different chat in the sidebar.
                # This updates active_conversation_id so the next voice turn
                # uses the correct history container immediately.
                elif msg_type == SWITCH_CONVERSATION:
                    conversation_id = data.get("conversation_id")

                    if not conversation_id:
                        logger.debug("Missing conversation_id in switch request.")
                        continue

                    self.active_conversation_id = conversation_id

                    # Create empty history container if this is the first time
                    # the backend has seen this conversation.
                    if conversation_id not in self.conversations:
                        self.conversations[conversation_id] = []

                    logger.info(
                        "Switched active conversation to: %s",
                        conversation_id,
                    )

                else:
                    logger.debug("Unknown message type ignored: %s", msg_type)

            except json.JSONDecodeError:
                logger.error("Invalid JSON received from client.")
            except Exception as exc:
                logger.exception("Error receiving audio: %s", exc)

    # -----------------------------------------------------------------------
    # Stage 2 — Silence monitor
    # -----------------------------------------------------------------------

    async def monitor_silence(
        self,
        audio_buffer: asyncio.Queue,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """
        Phase 2 — Hybrid endpoint validator.

        Combines VAD speech-inactivity signals with the existing silence
        timeout to make endpoint decisions.  The timeout remains as a
        safety fallback so the system stays robust when VAD is disabled
        or encounters noisy/low-volume environments.

        Decision logic:
            1. Silence duration < timeout  → keep waiting (no change).
            2. Silence duration >= timeout AND VAD says user still speaking
               → suppress finalization, log suppression, keep waiting.
            3. Silence duration >= timeout AND VAD says user NOT speaking
               (or VAD is unavailable/disabled)
               → trigger finalization as before.
        """
        logger.info("Hybrid endpoint validator started.")
        silence_timeout_s = Config.SILENCE_THRESHOLD / 1000.0

        try:
            while websocket.close_code is None and not self._shutdown_event.is_set():
                if (
                    self.metrics["audio_received_at"] == 0
                    or self.is_speaking
                    or self.is_processing
                    or self._final_sent_for_turn >= self.current_turn_id
                ):
                    try:
                        await asyncio.wait_for(
                            self._audio_activity_event.wait(),
                            timeout=silence_timeout_s,
                        )
                        self._audio_activity_event.clear()
                    except asyncio.TimeoutError:
                        pass
                    continue

                silence_duration = time.monotonic() - self._last_audio_monotonic
                if silence_duration >= silence_timeout_s:
                    # --- Phase 2: VAD veto gate --------------------------
                    # If VAD is active and reports the user is still speaking,
                    # suppress this finalization attempt and wait for another
                    # silence_timeout_s window before re-checking.
                    if self.vad and self.vad.is_available and self._vad_speech_active:
                        logger.info(
                            "Endpoint suppressed — VAD reports user still speaking "
                            "(silence=%.2fs, vad_active=True).",
                            silence_duration,
                        )
                        self._set_state(SpeechState.USER_SPEAKING, "vad_suppressed_endpoint")
                        try:
                            await asyncio.wait_for(
                                self._audio_activity_event.wait(),
                                timeout=silence_timeout_s,
                            )
                            self._audio_activity_event.clear()
                        except asyncio.TimeoutError:
                            pass
                        continue

                    # VAD agrees (or is unavailable) — safe to finalize.
                    vad_context = (
                        f"vad_active={self._vad_speech_active}"
                        if self.vad and self.vad.is_available
                        else "vad=disabled"
                    )
                    logger.info(
                        "Hybrid endpoint triggered (silence=%.2fs, %s).",
                        silence_duration,
                        vad_context,
                    )
                    if self._request_finalization("hybrid_endpoint"):
                        await audio_buffer.put(None)
                    continue

                timeout_s = max(SILENCE_MIN_CHECK_S, silence_timeout_s - silence_duration)
                try:
                    await asyncio.wait_for(self._audio_activity_event.wait(), timeout=timeout_s)
                    self._audio_activity_event.clear()
                except asyncio.TimeoutError:
                    continue

        except Exception as exc:
            logger.debug("Endpoint validator stopped: %s", exc)

    # -----------------------------------------------------------------------
    # Metrics helpers
    # -----------------------------------------------------------------------

    def _blank_metrics(self) -> Dict[str, float]:
        """Return a zeroed metrics dictionary for a new turn."""
        return {
            "audio_received_at": 0.0,
            "stt_first_partial_at": 0.0,
            "llm_first_token_at": 0.0,
            "tts_first_audio_at": 0.0,
        }

    def _reset_metrics(self) -> None:
        """Reset per-turn timing state ready for the next utterance."""
        self.metrics = self._blank_metrics()
        logger.debug("Metrics reset.")

    def _log_metrics(self) -> None:
        """Log a structured performance summary for the completed turn."""
        base = self.metrics.get("audio_received_at", 0.0)
        if base == 0.0:
            return

        lines = ["=" * 52, "  PERFORMANCE METRICS", "=" * 52]

        stt_t = self.metrics.get("stt_first_partial_at", 0.0)
        if stt_t:
            lines.append(f"  STT first partial : {(stt_t - base) * 1000:.0f}ms")

        llm_t = self.metrics.get("llm_first_token_at", 0.0)
        if llm_t:
            lines.append(f"  LLM first token   : {(llm_t - base) * 1000:.0f}ms")

        tts_t = self.metrics.get("tts_first_audio_at", 0.0)
        if tts_t:
            total_ms = (tts_t - base) * 1000
            lines.append(f"  TTS first audio   : {total_ms:.0f}ms")
            lines.append(f"  TOTAL LATENCY     : {total_ms:.0f}ms")

            if total_ms < 500:
                verdict = "EXCELLENT — feels instant"
            elif total_ms < 1000:
                verdict = "GREAT — conversational"
            elif total_ms < 1500:
                verdict = "ACCEPTABLE — could improve"
            else:
                verdict = "SLOW — needs optimisation"

            lines.append(f"  Verdict           : {verdict}")

        lines.append("=" * 52)
        logger.info("\n%s", "\n".join(lines))

    # -----------------------------------------------------------------------
    # Bug 2/3/4 fix — centralized turn lifecycle helpers
    # -----------------------------------------------------------------------

    def _begin_turn_finalization(self) -> bool:
        """
        Atomically mark the current turn as finalizing.

        Bug 2/3/4 fix — single method called by BOTH EOS paths
        (receive_audio END_OF_SPEECH and monitor_silence).  Ensures all
        finalization state is updated consistently and completely:

            * _eos_fired         — blocks the second EOS path (no-op)
            * _turn_finalizing   — signals stt_pipeline to stop partials
            * _final_sent_for_turn — blocks silence monitor outer condition
            * metrics reset      — prevents silence monitor re-entry

        Returns:
            True  — this call is the first (caller should enqueue None).
            False — finalization already started; caller must do nothing.
        """
        if self._eos_fired.is_set():
            logger.debug(
                "EOS guard: finalization already active for turn %d — skipping.",
                self.current_turn_id,
            )
            return False

        # --- Atomic finalization state update ----------------------------
        self._eos_fired.set()
        self._turn_finalizing        = True
        self._final_sent_for_turn    = self.current_turn_id
        self.metrics["audio_received_at"] = 0   # silence monitor re-entry guard
        self._set_state(SpeechState.FINALIZING, "eos")

        logger.info(
            "Turn %d finalization started (EOS guard set).",
            self.current_turn_id,
        )
        return True

    def _begin_new_turn(self) -> None:
        """
        Reset ALL per-turn state so the next utterance starts clean.

        Bug 4 fix — centralized cleanup.  Called from receive_audio when
        the first audio chunk of a new turn arrives.  Prevents stale state
        from one turn bleeding into the next.

        Resets:
            * EOS / finalization guards
            * turn ID counter
            * audio timing / silence state
            * partial STT rate-limiter
            * metrics
            * VAD speech-active flag + Silero hidden state
        """
        self.current_turn_id += 1

        # STEP 5 — Assign voice turns to the currently active conversation.
        # Voice audio does not carry a per-chunk conversation_id, so we rely
        # on active_conversation_id to own the turn. If no conversation has
        # been activated yet (e.g. user speaks before typing), we fall back
        # to a stable default key so history is never lost.
        if self.active_conversation_id is None:
            self.active_conversation_id = "default_voice_conversation"

        # Freeze conversation ownership for this voice turn.
        # Even if the frontend switches chats mid-stream,
        # STT finalization and downstream routing must stay
        # attached to the conversation that owned the turn
        # when recording began.
        self.current_turn_conversation_id = self.active_conversation_id

        if self.active_conversation_id not in self.conversations:
            self.conversations[self.active_conversation_id] = []

        self._turn_owner_id      = self.current_turn_id
        self._next_txn("new_turn")
        self._eos_fired.clear()
        self._turn_finalizing    = False
        self._chunk_count        = 0
        self._last_partial_stt_at = 0.0
        self._turn_start_time    = time.time()
        self._reset_metrics()
        self.metrics["audio_received_at"] = time.time()
        self.interrupt_event.clear()
        self._endpoint_request_event.clear()
        self._set_state(SpeechState.USER_SPEAKING, "new_turn")

        # Phase 2 — reset VAD so stale debounce/hysteresis from the previous
        # turn does not bleed into the new one.
        if self.vad:
            self._vad_speech_active = False
            self.vad.reset()

        logger.info(
            "New turn %d started (conversation: %s).",
            self.current_turn_id,
            self.active_conversation_id,
        )

    # -----------------------------------------------------------------------
    # Phase 2 — VAD event handler (called synchronously from receive_audio)
    # -----------------------------------------------------------------------

    def _process_vad(self, pcm16_bytes: bytes) -> None:
        """
        Feed a PCM16 chunk into VAD and handle speech-state events.

        Called inline inside ``receive_audio`` so VAD events are processed
        in the same asyncio iteration as audio ingestion — zero extra latency.

        Handles:
            speech_started → USER_SPEAKING state, clears pending endpoint.
            speech_ended   → USER_PAUSED state, marks VAD speech inactive.
        """
        for result in self.vad.feed(pcm16_bytes):
            if result.speech_started:
                self._vad_speech_active = True
                # Clear any stale endpoint-request from a previous micro-pause
                # so the silence monitor doesn't finalize mid-utterance.
                self._endpoint_request_event.clear()
                self._set_state(SpeechState.USER_SPEAKING, "vad_speech_started")
                logger.info(
                    "VAD ▶ Speech started (turn %d, prob=%.3f).",
                    self.current_turn_id,
                    result.speech_probability,
                )

            elif result.speech_ended:
                self._vad_speech_active = False
                self._set_state(SpeechState.USER_PAUSED, "vad_speech_ended")
                logger.info(
                    "VAD ■ Speech ended   (turn %d, prob=%.3f) — "
                    "awaiting hybrid endpoint validation.",
                    self.current_turn_id,
                    result.speech_probability,
                )