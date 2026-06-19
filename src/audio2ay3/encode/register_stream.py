"""Assemble per-frame AY-3-8910 register snapshots — the legality choke point.

All writes go through :class:`RegisterStreamBuilder`, which masks every field to its true bit
width (tone coarse = 4 bits, noise = 5 bits, amplitude = 4 bits + envelope flag, shape = 4 bits)
and manages the mixer so a register state a real chip cannot reach can never be emitted.

Register map (per chip): R0/R1 A tone fine/coarse, R2/R3 B, R4/R5 C, R6 noise period,
R7 mixer (0 = enabled), R8-R10 A/B/C amplitude, R11/R12 envelope period, R13 envelope shape
(0xFF = no retrigger this frame), R14/R15 I/O ports (unused).
"""

from __future__ import annotations

import numpy as np

from .quantize import (
    AMP_MAX,
    ENV_PERIOD_MAX,
    NOISE_PERIOD_MAX,
    TONE_PERIOD_MAX,
    TONE_PERIOD_MIN,
)

ENV_NO_RETRIGGER = 0xFF  # R13 sentinel: leave the envelope generator running untouched
_AMP_ENV_BIT = 0x10  # amplitude bit 4: take level from the envelope generator


class RegisterStreamBuilder:
    """Accumulate legal register frames, then emit a ``(n_frames, 16)`` uint8 array."""

    def __init__(self, n_frames: int) -> None:
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        self.n_frames = n_frames
        self._frames = np.zeros((n_frames, 16), dtype=np.uint8)
        self._frames[:, 13] = ENV_NO_RETRIGGER
        # Mixer bits 0-5 set = tone/noise disabled on every channel; bits 6-7 (I/O) stay 0.
        self._mixer = np.full(n_frames, 0x3F, dtype=np.uint8)

    def set_tone(
        self,
        frame: int,
        channel: int,
        tone_period: int,
        amplitude: int,
        *,
        use_envelope: bool = False,
    ) -> None:
        """Enable tone on *channel* with a clamped period and amplitude."""
        tp = int(tone_period)
        if tp < TONE_PERIOD_MIN:
            tp = TONE_PERIOD_MIN
        elif tp > TONE_PERIOD_MAX:
            tp = TONE_PERIOD_MAX
        self._frames[frame, channel * 2] = tp & 0xFF
        self._frames[frame, channel * 2 + 1] = (tp >> 8) & 0x0F
        amp = max(0, min(AMP_MAX, int(amplitude)))
        if use_envelope:
            amp |= _AMP_ENV_BIT
        self._frames[frame, 8 + channel] = amp
        self._mixer[frame] &= ~(1 << channel) & 0xFF  # clear tone-disable bit

    def set_amplitude(self, frame: int, channel: int, amplitude: int) -> None:
        """Set a channel amplitude without touching the mixer (e.g. a noise-only voice)."""
        self._frames[frame, 8 + channel] = max(0, min(AMP_MAX, int(amplitude)))

    def enable_noise(self, frame: int, channel: int, noise_period: int) -> None:
        """Route the shared noise generator into *channel* and set its (shared) period."""
        np_ = max(0, min(NOISE_PERIOD_MAX, int(noise_period)))
        self._frames[frame, 6] = np_
        self._mixer[frame] &= ~(1 << (3 + channel)) & 0xFF  # clear noise-disable bit

    def disable_tone(self, frame: int, channel: int) -> None:
        self._mixer[frame] |= (1 << channel) & 0xFF

    def set_envelope(self, frame: int, period: int, shape: int) -> None:
        """Program the envelope period and retrigger with *shape* (0-15)."""
        ep = max(0, min(ENV_PERIOD_MAX, int(period)))
        self._frames[frame, 11] = ep & 0xFF
        self._frames[frame, 12] = (ep >> 8) & 0xFF
        self._frames[frame, 13] = shape & 0x0F

    def finish(self) -> np.ndarray:
        """Fold the accumulated mixer column into R7 and return the frame array."""
        self._frames[:, 7] = self._mixer
        return self._frames
