"""Render a :class:`YmSong` to PCM and to audio files via the emulator."""

from __future__ import annotations

import numpy as np

from ..chip.ay3_8910 import Ay3Emulator
from ..config import ChipConfig
from .audio_out import write_audio


class Renderer:
    """Emulate a YM register stream and produce peak-safe PCM / audio files."""

    def __init__(self, render_sr: int = 44_100, oversample: int = 2,
                 headroom_db: float = -1.0) -> None:
        self.render_sr = int(render_sr)
        self.oversample = int(oversample)
        self.headroom = 10.0 ** (headroom_db / 20.0)

    def render(self, song, normalize: bool = True) -> np.ndarray:
        chip = ChipConfig(master_clock_hz=song.master_clock,
                          frame_rate_hz=song.frame_rate)
        emu = Ay3Emulator(chip=chip, render_sr=self.render_sr,
                          oversample=self.oversample)
        pcm = emu.render_song(song)
        if pcm.size:
            pcm = pcm - float(np.mean(pcm))  # DC block (equivalent to AC coupling)
            if normalize:
                peak = float(np.max(np.abs(pcm)))
                if peak > 1e-9:
                    pcm = pcm / peak * self.headroom
        return pcm.astype(np.float32)

    def render_to_file(self, song, path: str, bitrate_kbps: int = 192,
                       max_seconds: float | None = None) -> np.ndarray:
        pcm = self.render(song)
        if max_seconds is not None:
            pcm = pcm[: int(max_seconds * self.render_sr)]
        write_audio(path, pcm, self.render_sr, bitrate_kbps)
        return pcm
