"""
Stage 3 — Speech-to-Text pipeline.

STT provider: STTManager  (services/stt/stt_manager.py)
    Primary  : ElevenLabs Scribe v2 (cloud, best Hindi/Hinglish accuracy)
    Fallback : FasterWhisper        (local CPU, zero cloud dependency)

Return-type contract (as of STT migration Phase 2):
    transcribe_partial() → str   empty string  "" on silence or skip
    transcribe_final()   → str   empty string  "" on failure

    Both "" and None are falsy in Python so all existing `if not partial_text`
    and `if final_text and ...` guards continue to behave correctly.
    Do NOT change these checks to `is not None` — that would break the
    empty-silence filtering that prevents silent turns from reaching the LLM.

This pipeline does NOT import STT providers directly.
All provider selection and fallback routing happens inside STTManager.
The pipeline only calls:
    await orchestrator.stt.transcribe_partial(window_bytes)  → str
    await orchestrator.stt.transcribe_final(audio_bytes)     → str
"""

import asyncio
import logging
import time

from websockets.server import WebSocketServerProtocol

from backend.app.config.config import Config
from backend.app.core.websocket.protocol import TRANSCRIPT_FINAL, TRANSCRIPT_PARTIAL
from backend.app.core.websocket.utils import send_json


logger = logging.getLogger("backend.app.core.streaming_server")

# ---------------------------------------------------------------------------
# Pipeline timing constants
# ---------------------------------------------------------------------------
STT_PARTIAL_MIN_INTERVAL = 0.25  # minimum seconds between partial STT calls
STT_MAX_UTTERANCE_SECONDS = 7.0  # hard cap — keeps realtime conversational feel

# STT rolling window: keep only the last N bytes for partial transcription.
# Formula: sample_rate × 2 bytes/sample × window_seconds
# Avoids O(n²) memory growth when the user speaks for a long time.
_STT_PARTIAL_WINDOW_BYTES = Config.SAMPLE_RATE * 2 * Config.STT_PARTIAL_WINDOW_SECONDS
_STT_MAX_UTTERANCE_BYTES = int(Config.SAMPLE_RATE * 2 * STT_MAX_UTTERANCE_SECONDS)


async def process_stt(
    orchestrator,
    audio_buffer,
    transcript_buffer,
    websocket: WebSocketServerProtocol,
) -> None:
    """
    Consume audio bytes; produce partial (UI) and final (LLM) transcripts.

    Issue 3 (FIXED) — O(n²) memory growth eliminated.
    Audio is accumulated in a ``bytearray`` for O(1) appends.
    Partial STT calls use only a rolling window
    (``_STT_PARTIAL_WINDOW_BYTES``) so the STT provider never receives
    the full 30-second buffer repeatedly.

    Partial transcripts:
        Rate-limited to once per ``STT_PARTIAL_MIN_INTERVAL`` second.
        Sent to the client for live display only — do NOT trigger LLM.

    Final transcripts:
        The full ``bytearray`` is sent when a ``None`` sentinel arrives.
    """
    logger.info("STT processor started.")

    # Issue 3 (FIXED) — bytearray for O(1) appends instead of list + join.
    audio_buf: bytearray = bytearray()
    partial_task: asyncio.Task | None = None
    final_task: asyncio.Task | None = None
    finalizing_turn_id: int | None = None
    truncated_turn_id: int | None = None

    async def _run_partial(window_bytes: bytes, scheduled_at: float, turn_id: int) -> None:
        start_at = time.time()
        schedule_delay_ms = (start_at - scheduled_at) * 1000

        if finalizing_turn_id == turn_id or turn_id != orchestrator.current_turn_id:
            logger.debug(
                "Partial STT skipped — turn %d is no longer active.",
                turn_id,
            )
            return

        try:
            partial_text = await orchestrator.stt.transcribe_partial(window_bytes)
            elapsed_ms = (time.time() - start_at) * 1000
            logger.info(
                "STT partial completed | %sms | schedule delay: %sms | window: %d bytes",
                f"{elapsed_ms:.0f}",
                f"{schedule_delay_ms:.0f}",
                len(window_bytes),
            )

            if not partial_text or orchestrator._turn_finalizing:
                return

            if turn_id != orchestrator.current_turn_id:
                logger.debug(
                    "Partial STT suppressed — turn %d no longer current.",
                    turn_id,
                )
                return

            if orchestrator.metrics["stt_first_partial_at"] == 0:
                orchestrator.metrics["stt_first_partial_at"] = start_at
                latency_ms = (start_at - orchestrator._turn_start_time) * 1000
                logger.info("STT latency: %sms", f"{latency_ms:.0f}")

            logger.debug("STT partial: %s", partial_text)

            await send_json(websocket, {
                "type": TRANSCRIPT_PARTIAL,
                "text": partial_text,
            })

        except asyncio.CancelledError:
            logger.debug("STT partial task cancelled.")
            return
        except Exception:
            logger.exception("STT partial task failed.")

    async def _run_final(audio_bytes: bytes, turn_id: int) -> None:
        nonlocal finalizing_turn_id
        try:
            logger.info(
                "STT finalising %d bytes for turn %d.",
                len(audio_bytes),
                turn_id,
            )

            final_text = await orchestrator.stt.transcribe_final(audio_bytes)

            if final_text and len(final_text.strip()) > 1:
                if turn_id != orchestrator.current_turn_id:
                    logger.debug(
                        "Final transcript suppressed — turn %d no longer current.",
                        turn_id,
                    )
                    return

                logger.info("STT final transcript: %s", final_text)
                orchestrator._final_sent_for_turn = turn_id

                await transcript_buffer.put({
                    "type": "final",
                    "text": final_text,
                    "turn_id": turn_id,
                    "timestamp": time.time(),
                    "conversation_id": orchestrator.current_turn_conversation_id,
                })

                await send_json(websocket, {
                    "type": TRANSCRIPT_FINAL,
                    "text": final_text,
                })
            else:
                logger.debug("STT final transcript empty or too short — ignored.")

        except asyncio.CancelledError:
            logger.debug("STT final task cancelled.")
            return
        except Exception:
            logger.exception("STT final task failed.")
        finally:
            if finalizing_turn_id == turn_id:
                orchestrator._turn_finalizing = False
                finalizing_turn_id = None

    try:
        while True:
            chunk = await audio_buffer.get()

            # ----------------------------------------------------------------
            # None sentinel — run final transcription
            # ----------------------------------------------------------------
            if chunk is None:
                if (
                    orchestrator._final_sent_for_turn == orchestrator.current_turn_id
                    and not audio_buf
                ):
                    logger.debug(
                        "Duplicate None for turn %d — skipping.",
                        orchestrator.current_turn_id,
                    )
                    continue

                if not audio_buf:
                    continue

                if finalizing_turn_id == orchestrator.current_turn_id:
                    logger.debug(
                        "Finalization already in progress for turn %d — skipping.",
                        orchestrator.current_turn_id,
                    )
                    continue

                finalizing_turn_id = orchestrator.current_turn_id
                orchestrator._turn_finalizing = True

                if partial_task and not partial_task.done():
                    partial_task.cancel()
                    partial_task = None
                    logger.debug("Cancelled in-flight partial before finalization.")

                audio_snapshot = bytes(audio_buf)
                audio_buf = bytearray()
                truncated_turn_id = None

                if final_task and not final_task.done():
                    final_task.cancel()
                    logger.debug("Cancelled stale final task before scheduling new final.")

                final_task = asyncio.create_task(
                    _run_final(audio_snapshot, orchestrator.current_turn_id)
                )
                continue

            # ----------------------------------------------------------------
            # Regular chunk — O(1) append
            # ----------------------------------------------------------------
            audio_buf.extend(chunk)

            if len(audio_buf) > _STT_MAX_UTTERANCE_BYTES:
                if truncated_turn_id != orchestrator.current_turn_id:
                    logger.warning(
                        "STT buffer capped at %d bytes for turn %d.",
                        _STT_MAX_UTTERANCE_BYTES,
                        orchestrator.current_turn_id,
                    )
                    truncated_turn_id = orchestrator.current_turn_id
                audio_buf = audio_buf[-_STT_MAX_UTTERANCE_BYTES:]

            now = time.time()
            enough_data = len(audio_buf) >= Config.SAMPLE_RATE * 2 * 0.2  # ~200 ms
            enough_time = (now - orchestrator._last_partial_stt_at) >= STT_PARTIAL_MIN_INTERVAL

            if enough_data and enough_time and not orchestrator.is_speaking:
                # Bug 3 fix — suppress partial STT once EOS / finalization has
                # started for this turn.  Without this guard, an in-flight partial
                # call would still run (wasting CPU) and subsequent loop iterations
                # kept scheduling new partials even after the None sentinel was
                # already in the queue, causing the "forever looping" log spam.
                if finalizing_turn_id == orchestrator.current_turn_id:
                    logger.debug(
                        "Partial STT suppressed — turn %d is finalizing.",
                        orchestrator.current_turn_id,
                    )
                else:
                    orchestrator._last_partial_stt_at = now

                    # Issue 3 (FIXED) — slice only the rolling window, not the
                    # full buffer, so the STT provider receives ≤ N seconds.
                    window_bytes = bytes(audio_buf[-_STT_PARTIAL_WINDOW_BYTES:])
                    if partial_task and not partial_task.done():
                        logger.debug("Partial STT skipped — previous task still running.")
                    else:
                        logger.debug(
                            "Scheduling partial STT | window: %d bytes | turn: %d",
                            len(window_bytes),
                            orchestrator.current_turn_id,
                        )
                        partial_task = asyncio.create_task(
                            _run_partial(window_bytes, now, orchestrator.current_turn_id)
                        )
    except asyncio.CancelledError:
        logger.info("STT processor cancelled — cleaning up tasks.")
        raise
    finally:
        if partial_task and not partial_task.done():
            partial_task.cancel()
        if final_task and not final_task.done():
            final_task.cancel()
