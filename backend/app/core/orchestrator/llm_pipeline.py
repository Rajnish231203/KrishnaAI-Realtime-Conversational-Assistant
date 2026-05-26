"""
Stage 4 — LLM (RAG + generation) pipeline.
"""

import logging
import time
import asyncio

from websockets.server import WebSocketServerProtocol

from backend.app.config.config import Config
from backend.app.core.websocket.protocol import (
    LLM_TOKEN,
    STATE,
    RESPONSE_COMPLETE,
)
from backend.app.core.websocket.utils import send_json
from backend.app.services.rag.prompt_builder import build_krishna_prompt


logger = logging.getLogger("backend.app.core.streaming_server")

SCRIPTURE_KEYWORDS = [
    "shlok",
    "shloka",
    "chapter",
    "verse",
    "adhyay",
    "gita me kya kaha",
    "geeta me kya kaha",
    "krishna ne kya kaha",
    "sanskrit",
]


async def process_llm(
    orchestrator,
    transcript_buffer,
    llm_token_buffer,
    websocket: WebSocketServerProtocol,
) -> None:
    """
    Consume final transcripts, run RAG retrieval, stream LLM sentences.

    Issue 7 (FIXED) — per-conversation ``conversation_history`` is now
    passed to ``build_krishna_prompt`` so each chat has isolated memory
    rather than sharing a global state.

    For each accepted transcript:
        1. Retrieve top-K Gita verses (``GitaRetriever``).
        2. Build a grounded prompt including conversation history.
        3. Stream LLM response sentence-by-sentence to TTS.
        4. If no verses found — use the fallback prompt (graceful degradation).
    """
    logger.info("LLM processor started.")

    while True:
        transcript_data = await transcript_buffer.get()

        user_text = transcript_data["text"].strip()
        source = transcript_data.get("source", "voice")
        conversation_id = transcript_data.get("conversation_id")

        if not conversation_id:
            logger.warning(
                "Transcript missing conversation_id — skipping turn %d.",
                transcript_data.get("turn_id", 0),
            )
            continue

        conversation_history = orchestrator.conversations.setdefault(
            conversation_id,
            [],
        )

        query_lower = user_text.lower()

        is_scripture_query = any(
            keyword in query_lower
            for keyword in SCRIPTURE_KEYWORDS
        )

        response_mode = (
            "scripture"
            if is_scripture_query
            else "general"
        )
        turn_id = transcript_data.get("turn_id", 0)

        if turn_id <= orchestrator.processed_turn_id:
            logger.debug(
                "Skipping duplicate turn %d (last processed: %d).",
                turn_id,
                orchestrator.processed_turn_id,
            )
            continue

        if not user_text:
            logger.debug("Empty/blank utterance — skipped.")
            continue

        async with orchestrator.processing_lock:
            if turn_id <= orchestrator.processed_turn_id:
                continue

            orchestrator.is_processing = True
            orchestrator.processed_turn_id = turn_id
            orchestrator._turn_start_time = time.time()

            logger.info("LLM processing turn %d: %s", turn_id, user_text)

            await send_json(websocket, {
                "type": STATE,
                "message": "Krishna is thinking...",
                "status": "processing",
            })

            assistant_response = ""

            try:
                logger.info("RAG retrieval for: %s", user_text)
                # Fix 3 — retrieve() calls SentenceTransformer.encode() and
                # FAISS search, both CPU-bound.  Running them in a worker thread
                # keeps the asyncio event loop free during retrieval (50–300 ms).
                verses = await asyncio.to_thread(
                    orchestrator.retriever.retrieve,
                    user_text,
                    Config.RAG_TOP_K,
                    Config.RAG_SIMILARITY_THRESHOLD,
                )

                if verses:
                    logger.info("RAG found %d verses.", len(verses))
                    # Issue 7 (FIXED) — pass conversation history so LLM
                    # has multi-turn context, not just the current query.
                    prompt = build_krishna_prompt(
                        user_text,
                        verses,
                        conversation_history=conversation_history,
                        response_mode=response_mode,
                    )
                else:
                    logger.info(
                        "No verses above threshold — using fallback prompt."
                    )
                    prompt = build_krishna_prompt(
                        user_text,
                        [],
                        conversation_history=conversation_history,
                        response_mode=response_mode,
                    )

                async for sentence in orchestrator.llm.stream_krishna_response(prompt):
                    # Fix 7 — stop consuming Groq/OpenAI tokens the moment the
                    # user barges in.  Without this check the LLM loop continued
                    # generating silently, wasting API budget and filling the
                    # llm_token_buffer with sentences that TTS would skip anyway.
                    if orchestrator.interrupt_event.is_set():
                        logger.info(
                            "LLM generation interrupted on turn %d — barge-in detected.",
                            turn_id,
                        )
                        break

                    if not sentence or not sentence.strip():
                        logger.debug("Empty sentence from LLM — skipped.")
                        continue

                    if orchestrator.metrics["llm_first_token_at"] == 0:
                        orchestrator.metrics["llm_first_token_at"] = time.time()
                        latency_ms = (
                            orchestrator.metrics["llm_first_token_at"]
                            - orchestrator._turn_start_time
                        ) * 1000
                        logger.info(
                            "LLM first token latency: %sms",
                            f"{latency_ms:.0f}",
                        )

                    assistant_response += sentence

                    # Still send tokens to frontend for internal accumulation.
                    # Frontend will reveal text progressively in sync with audio.
                    await send_json(websocket, {
                        "type": LLM_TOKEN,
                        "token": sentence,
                    })

                    # NOTE: No longer queuing individual sentences to TTS.
                    # Full response is queued ONCE after streaming completes.

                if source == "voice":
                    # Queue the ENTIRE accumulated response as ONE TTS item.
                    # This ensures only ONE ElevenLabs API call per response,
                    # dramatically reducing free-tier quota usage.
                    if assistant_response.strip():
                        logger.info(
                            "Queuing full response for single TTS synthesis (%d chars).",
                            len(assistant_response),
                        )
                        await llm_token_buffer.put(assistant_response.strip())

                    # The None sentinel tells tts_pipeline.py that
                    # generation finished, and TTS later emits RESPONSE_COMPLETE.
                    await llm_token_buffer.put(None)

            except Exception as exc:
                logger.exception(
                    "LLM/RAG error on turn %d: %s", turn_id, exc
                )
                fallback = (
                    "I'm having trouble responding right now. "
                    "Please try speaking again."
                )
                if source == "voice":
                    await llm_token_buffer.put(fallback)
                    await llm_token_buffer.put(None)
                orchestrator.is_processing = False
                continue

            # --- Update conversation history --------------------------
            conversation_history.append(
                {"role": "user", "content": user_text}
            )
            conversation_history.append(
                {"role": "assistant", "content": assistant_response.strip()}
            )

            max_history = Config.MAX_CONVERSATION_HISTORY
            if len(conversation_history) > max_history:
                orchestrator.conversations[conversation_id] = (
                    conversation_history[-max_history:]
                )
                conversation_history = orchestrator.conversations[conversation_id]
                logger.debug(
                    "Conversation history trimmed to %d entries.", max_history
                )

            orchestrator._log_metrics()
            orchestrator._reset_metrics()
            orchestrator.is_processing = False

            # Text mode has no TTS/playback lifecycle,
            # so emit RESPONSE_COMPLETE only after
            # backend cleanup fully finishes.
            if source != "voice":
                await send_json(websocket, {
                    "type": RESPONSE_COMPLETE,
                })
