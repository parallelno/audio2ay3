"""Mapping: arrange a transcription onto the AY's 3 tone channels + noise generator."""

from __future__ import annotations

from .percussion import apply_percussion
from .voices import Voice, allocate_voices, is_sustained_program, n_frames_for, place_bass

__all__ = [
    "Voice",
    "allocate_voices",
    "apply_percussion",
    "is_sustained_program",
    "n_frames_for",
    "place_bass",
]
