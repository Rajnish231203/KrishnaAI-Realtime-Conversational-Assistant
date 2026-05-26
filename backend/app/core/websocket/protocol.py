"""
WebSocket message protocol constants.
"""

AUDIO_CHUNK = "audio_chunk"
INTERRUPT = "interrupt"
END_OF_SPEECH = "end_of_speech"

TRANSCRIPT_PARTIAL = "transcript_partial"
TRANSCRIPT_FINAL = "transcript_final"
LLM_TOKEN = "llm_token"
AUDIO_RESPONSE_CHUNK = "audio_response_chunk"  # Fix 8 — was "audio_chunk" (collision with inbound)
CHAT_MESSAGE = "chat_message"
AUDIO_COMPLETE = "audio_complete"
RESPONSE_COMPLETE = "response_complete"
SWITCH_CONVERSATION = "switch_conversation"
STATE = "state"
