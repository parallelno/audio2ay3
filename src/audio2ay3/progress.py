"""A tiny, dependency-free stage progress bar for the long ``convert``/``preview`` runs.

The neural stages (separation, transcription) dominate wall-clock time, so the pipeline reports a
coarse, one-line-per-stage bar as each stage *starts*. Printing a fresh line per stage (rather
than animating a single ``\\r`` line) keeps it readable when a sub-tool prints its own progress
bar in between — Demucs and Basic Pitch both do. The bar goes to stderr so stdout keeps carrying
only the machine-readable ``ok:`` line.
"""

from __future__ import annotations

import sys
from typing import Protocol, TextIO


class Progress(Protocol):
    """The minimal surface the pipeline needs: advance one stage."""

    def step(self, label: str) -> None: ...


class NullProgress:
    """A do-nothing sink — the default when no reporter is supplied (e.g. library use)."""

    def step(self, label: str) -> None:
        return None


class ProgressReporter:
    """Render a coarse, stage-based progress bar to a stream (stderr by default).

    *total* is the number of :meth:`step` calls the run will make; each call advances the bar by
    one and prints a new line so it coexists with sub-tools that emit their own output.
    """

    def __init__(self, total: int, *, stream: TextIO | None = None, width: int = 24) -> None:
        self.total = max(1, int(total))
        self.width = max(1, int(width))
        self.stream = stream if stream is not None else sys.stderr
        self._done = 0

    def step(self, label: str) -> None:
        self._done = min(self._done + 1, self.total)
        frac = self._done / self.total
        filled = int(round(frac * self.width))
        bar = "#" * filled + "." * (self.width - filled)
        self.stream.write(
            f"audio2ay3: [{bar}] {int(frac * 100):3d}% [{self._done}/{self.total}] {label}\n"
        )
        self.stream.flush()
