"""
Main entry point for the Krishna Voice Assistant WebSocket server.

Responsibilities:
- Validate configuration on startup
- Start the async streaming server
- Handle graceful shutdown
- Log startup and runtime errors
"""

import asyncio
import logging

from backend.app.config.config import Config
from backend.app.core.websocket.server import main as start_streaming_server


logger = logging.getLogger(__name__)


async def run_server():
    """Validate config then run the streaming server."""
    # Validate all configuration at startup — surfaces mis-configuration early
    # (wrong STT_PROVIDER, missing API keys, bad audio settings, etc.) before
    # any connection is accepted.  Does NOT crash on missing ElevenLabs key —
    # STTManager warmup() handles that gracefully via fallback.
    Config.validate()

    logger.info("Starting WebSocket streaming server...")
    await start_streaming_server()


def main():
    """Application entrypoint."""

    try:
        asyncio.run(run_server())

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")

    except Exception:
        logger.exception("Fatal error in WebSocket server")


if __name__ == "__main__":
    main()