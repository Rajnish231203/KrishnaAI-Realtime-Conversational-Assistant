"""
WebSocket server entrypoint for the Krishna Voice Assistant.
"""

import asyncio
import logging
import os

import websockets
from websockets.server import WebSocketServerProtocol

from backend.app.config.config import Config, setup_logging
from backend.app.core.orchestrator.orchestrator import StreamingOrchestrator
from backend.app.services.stt.faster_whisper_provider import prewarm_model as _prewarm_whisper_model


logger = logging.getLogger("backend.app.core.streaming_server")

# ---------------------------------------------------------------------------
# WebSocket server tuning constants
# ---------------------------------------------------------------------------
WS_HOST = os.getenv("HOST", Config.WEBSOCKET_HOST)
WS_PORT = int(os.getenv("PORT", str(Config.WEBSOCKET_PORT)))
WS_MAX_SIZE = 10_000_000   # 10 MB — accommodates large PCM payloads
WS_PING_INTERVAL = 20      # seconds between server keepalive pings
WS_PING_TIMEOUT = 20       # seconds before a non-responsive client is dropped
WS_MAX_QUEUE = 32          # max pending connections (DoS protection)


# ===========================================================================
# Connection factory (Issue 1 FIX — per-connection orchestrator)
# ===========================================================================

async def connection_handler(
    websocket: WebSocketServerProtocol,
    path: str,
) -> None:
    """
    Top-level WebSocket handler registered with ``websockets.serve``.

    Issue 1 (FIXED) — creates a **fresh** ``StreamingOrchestrator`` for
    every connection so that all per-user state (is_speaking, history,
    metrics, interrupt_event ...) is completely isolated. Concurrent clients
    never share any mutable state.
    """
    orchestrator = StreamingOrchestrator()
    await orchestrator.handle_client(websocket, path)


# ===========================================================================
# Server entry-point
# ===========================================================================

async def main() -> None:
    """
    Initialise structured logging and start the WebSocket server.

    Issue 9  (FIXED) — ``asyncio.Event().wait()`` replaces ``asyncio.Future()``.
    Issue 10 (FIXED) — ``compression=None`` disables per-frame compression
                        which wastes CPU on already-encoded PCM audio.
    Issue 11 (FIXED) — ``max_queue=WS_MAX_QUEUE`` limits pending connections
                        to protect against connection-flood DoS attacks.

    Server parameters:
        ``max_size``      10 MB  — handles large PCM audio payloads.
        ``ping_interval`` 20 s   — keepalive pings.
        ``ping_timeout``  20 s   — disconnect non-responsive clients.
        ``compression``   None   — disabled; audio payloads gain nothing.
        ``max_queue``     32     — connection backlog cap.
    """
    setup_logging()

    logger.info("=" * 60)
    logger.info("  KRISHNA REAL-TIME VOICE ASSISTANT")
    logger.info("  Optimised for < 1 second latency")
    logger.info("  WebSocket: ws://%s:%d", WS_HOST, WS_PORT)
    logger.info("  Per-connection orchestrators: enabled")
    logger.info("  STT primary  : %s", Config.STT_PROVIDER)
    logger.info(
        "  STT fallback : %s",
        "FasterWhisper (auto)" if Config.STT_FALLBACK_ENABLED else "disabled",
    )
    logger.info("=" * 60)

    # STT migration (Phase 2) — pre-warm FasterWhisper singleton before accepting
    # connections.  Model load costs 2–8 s on CPU.  Paying it here means the
    # very first user utterance gets normal partial latency instead of ~12 s.
    #
    # IMPORTANT: prewarm_model() is sourced from the NEW faster_whisper_provider
    # (services/stt/).  The old streaming_stt._get_whisper_model is intentionally
    # NOT imported here — doing so would trigger a SECOND WhisperModel singleton,
    # doubling RAM usage and causing CPU contention.
    logger.info("[FasterWhisper STT] Pre-warming singleton (new provider path)...")
    await _prewarm_whisper_model()
    logger.info("[FasterWhisper STT] Singleton ready — accepting connections.")

    # Issue 1  — connection_handler creates a new orchestrator per connection.
    # Issue 9  — asyncio.Event().wait() instead of asyncio.Future().
    # Issue 10 — compression=None.
    # Issue 11 — max_queue=WS_MAX_QUEUE.
    stop_event = asyncio.Event()

    async with websockets.serve(
        connection_handler,
        WS_HOST,
        WS_PORT,
        max_size=WS_MAX_SIZE,
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT,
        compression=None,
        max_queue=WS_MAX_QUEUE,
    ):
        logger.info("Server ready — waiting for connections.")
        await stop_event.wait()   

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
