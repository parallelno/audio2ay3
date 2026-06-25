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
    # Per-frame loudness shape sampled from the source stem across the note's span, normalised
    # so 1.0 is the note's loudest frame. Frame ``k`` corresponds to ``onset + k`` at the
    # frame rate. Empty when no source envelope is available; the arranger then falls back to a
    # synthetic amplitude envelope. This is what lets a held note sustain and a pluck decay
    # like the original instead of every note sharing one fixed curve.
    amp_contour: tuple[float, ...] = ()
    # General-MIDI program (0-127) when the backend knows the instrument identity (MT3), else
    # ``None`` (Basic Pitch is pitch-only). The arranger uses it to keep a lead audible over a
    # pad when tone channels are scarce (see :func:`mapping.voices._program_rank`).
    program: int | None = None
    # Which separated source this note came from: ``"melody"`` (instrumental stem), ``"bass"``
    # (bass stem), or ``"vocals"`` (sung melody kept via ``--vocals``); ``None`` when unknown
    # (e.g. the multitrack backends, which don't separate stems). Lets the arranger scope
    # per-stem effects such as vibrato even when the backend gives no GM program.
    stem: str | None = None

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
    """Everything the arranger needs: pitched notes, drum hits, and total length.

    ``notes`` are the melodic/harmonic content (shared across the free tone channels).
    ``bass_notes`` come from the isolated bass stem and get a dedicated channel so the low
    end never has to fight the lead for a voice (see :func:`mapping.place_bass`).
    """

    notes: list[Note] = field(default_factory=list)
    percussion: list[Percussion] = field(default_factory=list)
    bass_notes: list[Note] = field(default_factory=list)
    duration_s: float = 0.0

    def sorted_notes(self) -> list[Note]:
        return sorted(self.notes, key=lambda n: n.onset_s)
