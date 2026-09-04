"""kotodama: two-stage JA->EN subtitle pipeline.

Stage 1: faster-whisper Japanese transcription (timestamps originate here).
Stage 2: JA->EN translation via an OpenAI-compatible endpoint.
"""

__version__ = "0.1.0"
