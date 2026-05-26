"""
HTTP server for serving the Krishna Voice Assistant frontend.

Responsibilities:
- Serve static frontend files
- Add CORS headers
- Log HTTP requests
"""

import http.server
import socketserver
import logging
from pathlib import Path

from backend.app.config.config import Config


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PORT = Config.HTTP_PORT

BASE_DIR = Path(__file__).resolve().parents[3]
FRONTEND_DIR = BASE_DIR / "frontend"


# -------------------------------------------------------------------
# Custom request handler
# -------------------------------------------------------------------

class FrontendHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP request handler for serving frontend assets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def end_headers(self):
        """Add CORS headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, format, *args):
        """Override default logging."""
        logger.info("HTTP request: %s", format % args)


# -------------------------------------------------------------------
# Server start function
# -------------------------------------------------------------------

def start_http_server():
    """Start HTTP server for frontend."""

    logger.info("Starting HTTP server")
    logger.info("Serving frontend from: %s", FRONTEND_DIR)

    try:
        socketserver.TCPServer.allow_reuse_address = True

        with socketserver.TCPServer(("", PORT), FrontendHTTPRequestHandler) as httpd:

            logger.info("HTTP server running at http://localhost:%s", PORT)
            logger.info("Open http://localhost:%s/client.html in your browser", PORT)

            httpd.serve_forever()

    except Exception:
        logger.exception("HTTP server failed to start")
        raise


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    start_http_server()