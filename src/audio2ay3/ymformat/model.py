"""Normalised in-memory representation of a YM register-dump tune."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class YmSong:
    """A YM tune as an ``(n_frames, 16)`` uint8 register array plus metadata.

    Register columns are R0..R15. The model is format-agnostic: readers de-interleave and
    zero-extend (e.g. YM3's 14 registers) into this shape, and the writer serialises it back.
    """

    frames: np.ndarray  # (n_frames, 16) uint8
    master_clock: int = 1_773_400
    frame_rate: int = 50
    loop_frame: int = 0
    version: str = "YM6"
    name: str = ""
    author: str = ""
    comment: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / float(self.frame_rate) if self.frame_rate else 0.0
