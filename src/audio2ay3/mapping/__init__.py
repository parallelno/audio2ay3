"""Mapping: arrange a transcription onto the AY's 3 tone channels + noise generator."""

from __future__ import annotations

from .percussion import apply_percussion, percussion_busy_frames
from .voices import (
    Voice,
    allocate_voices,
    is_breath_program,
    is_sustained_program,
    is_vibrato_program,
    n_frames_for,
    place_bass,
)

__all__ = [
    "Voice",
    "allocate_voices",
    "apply_percussion",
    "is_breath_program",
    "is_sustained_program",
    "is_vibrato_program",
    "n_frames_for",
    "percussion_busy_frames",
    "place_bass",
]
