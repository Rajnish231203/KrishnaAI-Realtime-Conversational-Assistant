"""
Stage 5 — Text-to-Speech pipeline.
"""

import asyncio
import base64
import logging
import time

from websockets.server import WebSocketServerProtocol

from backend.app.core.websocket.protocol import (
    AUDIO_COMPLETE,
    AUDIO_RESPONSE_CHUNK,
    RESPONSE_COMPLETE,
    STATE,
)
from backend.app.core.websocket.utils import send_json


logger = logging.getLogger("backend.app.core.streaming_server")


async def process_tts(
    orchestrator,
    llm_token_buffer,
    websocket: WebSocketServerProtocol,
) -> None:
    """
    Consume LLM sentences, synthesise audio, stream PCM to the client.

    Issue 8 (FIXED) — interrupt detection uses ``interrupt_event.is_set()``
    instead of the raw boolean flag. ``asyncio.Event`` is safe to read
    and set across concurrent tasks with no race conditions.
    """
    logger.info("TTS processor started.")

    while True:
        text = await llm_token_buffer.get()

        if text is None:
            await send_json(websocket, {"type": RESPONSE_COMPLETE})
            orchestrator._log_metrics()
            logger.info("TTS: response_complete sent.")
            continue

        # Issue 8 (FIXED) — use Event instead of raw bool.
        if orchestrator.interrupt_event.is_set():
            logger.info("TTS: sentence skipped due to barge-in interrupt.")
            continue

        try:
            orchestrator.is_speaking = True
            await send_json(websocket, {
                "type": STATE,
                "message": "Krishna is speaking...",
                "status": "speaking",
            })

            logger.info("TTS synthesising: %s…", text[:60])
            first_chunk = True

            async for chunk in orchestrator.tts.stream_audio(text):
                if orchestrator.interrupt_event.is_set():
                    logger.info("TTS playback interrupted mid-sentence.")
                    break

                if not chunk:
                    logger.debug("Empty TTS chunk — skipped.")
                    continue

                if first_chunk:
                    orchestrator.metrics["tts_first_audio_at"] = time.time()
                    if orchestrator.metrics["llm_first_token_at"] > 0:
                        ttfa_ms = (
                            orchestrator.metrics["tts_first_audio_at"]
                            - orchestrator.metrics["llm_first_token_at"]
                        ) * 1000
                        logger.info(
                            "TTS first audio latency: %sms",
                            f"{ttfa_ms:.0f}",
                        )
                    first_chunk = False

                await send_json(websocket, {
                    "type": AUDIO_RESPONSE_CHUNK,
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                })

            await send_json(websocket, {"type": AUDIO_COMPLETE})
            # Fix 6 — removed unconditional 500 ms sleep that added artificial
            # latency after every TTS sentence (was: await asyncio.sleep(0.5)).
            orchestrator.is_speaking = False

            await send_json(websocket, {
                "type": STATE,
                "message": "Krishna finished speaking",
                "status": "success",
            })

        except Exception as exc:
            logger.exception("TTS error: %s", exc)
            orchestrator.is_speaking = False
            await send_json(websocket, {
                "type": STATE,
                "message": "idle",
                "status": "idle",
            })
