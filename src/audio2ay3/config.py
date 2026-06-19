"""Typed configuration for the chip target and a conversion/render run.

The analysis defaults are neural (Demucs separation + Basic Pitch transcription); there is no
classical-DSP note-detection path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ChipConfig:
    """Describes the target PSG configuration."""

    master_clock_hz: int = 1_773_400  # ZX Spectrum default
    frame_rate_hz: int = 50
    n_chips: int = 1  # 1 or 2 (dual-AY)
    tone_channels: int = 3  # per chip

    @property
    def total_tone_channels(self) -> int:
        return self.n_chips * self.tone_channels


@dataclass(frozen=True)
class RunConfig:
    """End-to-end run configuration."""

    chip: ChipConfig = field(default_factory=ChipConfig)
    use_gpu: bool = True  # auto-falls back to CPU
    threads: int = 0  # 0 = auto
    # Neural analysis stack (no DSP option by design):
    separation: Literal["demucs", "spleeter", "none"] = "demucs"
    transcription: Literal["basic-pitch", "mt3", "onsets-frames"] = "basic-pitch"
    render_sr: int = 44_100
    oversample: int = 2
    mp3_bitrate_kbps: int = 192
    seed: int = 0
