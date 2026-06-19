"""Register encoding: turn quantities into legal AY-3-8910 register frames."""

from __future__ import annotations

from .quantize import (
    frames_for_duration,
    hz_to_noise_period,
    hz_to_tone_period,
    quantize_tone,
    seconds_to_frame,
    tone_period_to_hz,
    velocity_to_amplitude,
)
from .register_stream import ENV_NO_RETRIGGER, RegisterStreamBuilder

__all__ = [
    "ENV_NO_RETRIGGER",
    "RegisterStreamBuilder",
    "frames_for_duration",
    "hz_to_noise_period",
    "hz_to_tone_period",
    "quantize_tone",
    "seconds_to_frame",
    "tone_period_to_hz",
    "velocity_to_amplitude",
]
