"""
backend.app.services.stt
========================
STT provider abstraction package.

Public surface consumed by the orchestrator integration phase:

    from backend.app.services.stt import STTManager
    from backend.app.services.stt import BaseSTTProvider, STTProviderError
"""

from backend.app.services.stt.base_stt import BaseSTTProvider, STTProviderError
from backend.app.services.stt.stt_manager import STTManager

__all__ = [
    "BaseSTTProvider",
    "STTProviderError",
    "STTManager",
]
