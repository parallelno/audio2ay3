"""Neural analysis stage: audio -> instrumental stems -> note/percussion transcription."""

from __future__ import annotations

from .load_audio import load_audio
from .model import Note, Percussion, Transcription
from .percussion_detect import detect_percussion
from .separate import SeparationResult, separate, separate_stems
from .transcribe import transcribe

__all__ = [
    "Note",
    "Percussion",
    "SeparationResult",
    "Transcription",
    "detect_percussion",
    "load_audio",
    "separate",
    "separate_stems",
    "transcribe",
]
