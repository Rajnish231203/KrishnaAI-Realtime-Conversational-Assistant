"""
WebSocket helpers for the streaming server.
"""

import json
import logging
from typing import Any, Dict

from websockets.server import WebSocketServerProtocol


logger = logging.getLogger("backend.app.core.streaming_server")


async def send_json(
    websocket: WebSocketServerProtocol,
    payload: Dict[str, Any],
) -> None:
    """
    Serialise *payload* to JSON and send it over *websocket*.

    Guards against sending to an already-closed socket before attempting
    the network call. Failures are logged via ``logger.exception`` and
    swallowed so that one bad send never crashes a pipeline stage.

    Args:
        websocket: Active WebSocket connection to the client.
        payload:   Dictionary to JSON-serialise and transmit.
    """
    # Issue 5 — check closed state before sending to avoid noisy exceptions.
    # Issue 6 — use close_code instead of deprecated websocket.closed.
    if websocket.close_code is not None:
        logger.debug(
            "send_json skipped — socket already closed (code=%s, type=%s).",
            websocket.close_code,
            payload.get("type"),
        )
        return

    try:
        await websocket.send(json.dumps(payload))
    except Exception as exc:
        logger.exception(
            "send_json failed (payload type=%s): %s",
            payload.get("type"),
            exc,
        )
