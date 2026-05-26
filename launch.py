"""
Launch script for Krishna Real-Time Voice Assistant.

Responsible for starting:
1. HTTP static server
2. WebSocket streaming server

Production features:
- structured logging
- subprocess lifecycle management
- environment validation
- graceful shutdown
"""

import logging
import subprocess
import sys
import os
import signal
from pathlib import Path

from backend.app.config.config import Config


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("launcher")


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

HTTP_SERVER_PATH = BASE_DIR / "backend" / "app" / "core" / "http_server.py"
WEBSOCKET_SERVER_PATH = BASE_DIR / "backend" / "main.py"


# -------------------------------------------------------------------
# Process helpers
# -------------------------------------------------------------------

def start_process(script_path: Path, name: str) -> subprocess.Popen:
    """
    Start a Python subprocess.

    Args:
        script_path: path to Python script
        name: human-readable process name

    Returns:
        subprocess.Popen
    """

    logger.info("Starting %s...", name)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)

    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env
    )

    logger.info("%s started with PID %s", name, process.pid)

    return process


# -------------------------------------------------------------------
# Environment validation
# -------------------------------------------------------------------

def validate_environment():
    """Validate runtime environment."""

    if not os.path.exists(".env"):
        logger.warning(".env file not found. API keys may be missing.")

    try:
        from backend.app.config.config import Config
        Config.validate()
        logger.info("Configuration validated successfully")

    except Exception:
        logger.exception("Configuration validation failed")
        raise


# -------------------------------------------------------------------
# Main launcher
# -------------------------------------------------------------------

def main():

    logger.info("=" * 60)
    logger.info("KRISHNA REAL-TIME VOICE ASSISTANT")
    logger.info("Target latency: < 1 second")
    logger.info("Architecture: Parallel streaming pipeline")
    logger.info("=" * 60)

    validate_environment()

    processes = []

    try:

        # Start HTTP server
        http_process = start_process(
            HTTP_SERVER_PATH,
            "HTTP server"
        )
        processes.append(http_process)

        # Start WebSocket server
        ws_process = start_process(
            WEBSOCKET_SERVER_PATH,
            "WebSocket server"
        )
        processes.append(ws_process)

        logger.info("=" * 60)
        logger.info("SERVERS READY")
        logger.info("WebSocket: ws://%s:%s", Config.WEBSOCKET_HOST, Config.WEBSOCKET_PORT)
        logger.info("Web Client: http://localhost:%s/client.html", Config.HTTP_PORT)
        logger.info("=" * 60)

        # Wait for servers
        for process in processes:
            process.wait()

    except KeyboardInterrupt:

        logger.info("Shutdown requested. Terminating servers...")

        for process in processes:
            process.terminate()

        for process in processes:
            process.wait()

        logger.info("All servers stopped")

    except Exception:
        logger.exception("Fatal launcher error")

        for process in processes:
            process.terminate()

        raise


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()