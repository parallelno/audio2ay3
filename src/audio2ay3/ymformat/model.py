"""Normalised in-memory representation of a YM register-dump tune."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class YmSong:
    """A YM tune as an ``(n_frames, 16 * n_chips)`` uint8 register array plus metadata.

    A single AY occupies 16 register columns (R0..R15). A dual-AY tune (``n_chips == 2``) stores
    the second chip's 16 registers side-by-side, so the block is 32 columns wide: chip 0 in
    columns 0..15, chip 1 in columns 16..31. The model is format-agnostic: readers de-interleave
    and zero-extend (e.g. YM3's 14 registers) into this shape, and the writer serialises it back.
    """

    frames: np.ndarray  # (n_frames, 16 * n_chips) uint8
    master_clock: int = 1_773_400
    frame_rate: int = 50
    loop_frame: int = 0
    version: str = "YM6"
    name: str = ""
    author: str = ""
    comment: str = ""
    n_chips: int = 1  # 1 or 2 (dual-AY)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / float(self.frame_rate) if self.frame_rate else 0.0

    def per_chip_frames(self) -> list[np.ndarray]:
        """Split the register block into one ``(n_frames, 16)`` array per chip."""
        n = max(1, self.n_chips)
        width = self.frames.shape[1] // n
        return [self.frames[:, c * width:(c + 1) * width] for c in range(n)]

    def per_chip_songs(self) -> list[YmSong]:
        """One single-chip :class:`YmSong` per chip, sharing this song's metadata."""
        return [
            replace(self, frames=np.ascontiguousarray(block), n_chips=1)
            for block in self.per_chip_frames()
        ]
