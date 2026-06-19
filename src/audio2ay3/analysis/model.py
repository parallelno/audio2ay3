"""Intermediate representation between analysis and arrangement.

A :class:`Transcription` is the neutral hand-off produced by the neural analysis stage
(separation + transcription) and consumed by the deterministic mapping/encode stage. It is
deliberately backend-agnostic: Basic Pitch, MT3, or Onsets-and-Frames all reduce to the same
note + percussion event lists, so the arrangement code never depends on a specific model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PercussionKind = Literal["kick", "snare", "hat"]


@dataclass(frozen=True)
class Note:
    """A single pitched note event (already vocals-free, post-separation)."""

    onset_s: float
    duration_s: float
    pitch_hz: float
    velocity: float = 1.0  # perceptual loudness, 0..1

    @property
    def offset_s(self) -> float:
        return self.onset_s + self.duration_s


@dataclass(frozen=True)
class Percussion:
    """A single drum hit, classified coarsely into the three AY-friendly buckets."""

    onset_s: float
    kind: PercussionKind = "snare"
    velocity: float = 1.0


@dataclass
class Transcription:
    """Everything the arranger needs: pitched notes, drum hits, and total length."""

    notes: list[Note] = field(default_factory=list)
    percussion: list[Percussion] = field(default_factory=list)
    duration_s: float = 0.0

    def sorted_notes(self) -> list[Note]:
        return sorted(self.notes, key=lambda n: n.onset_s)
