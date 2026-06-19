"""Map physical quantities (Hz, loudness) to legal AY-3-8910 register values.

The chip pitches a tone with a 12-bit period ``TP`` so that ``f = clock / (16 * TP)`` — the same
convention the emulator uses, which keeps analysis -> encode -> emulate self-consistent. This
module is the arithmetic half of the "legality choke point": every value it returns is already
inside the hardware's range.
"""

from __future__ import annotations

import math

# 12-bit tone period. TP=0 behaves like TP=1 on hardware, so 1 is the usable floor.
TONE_PERIOD_MIN = 1
TONE_PERIOD_MAX = 4095

# 5-bit noise period.
NOISE_PERIOD_MIN = 0
NOISE_PERIOD_MAX = 31

# 16-bit envelope period.
ENV_PERIOD_MAX = 65_535

AMP_MAX = 15  # 4-bit amplitude


def hz_to_tone_period(freq_hz: float, master_clock_hz: int) -> int:
    """Nearest tone period for *freq_hz*; 0 if the frequency is non-positive."""
    if freq_hz <= 0.0:
        return 0
    return round(master_clock_hz / (16.0 * freq_hz))


def tone_period_to_hz(tone_period: int, master_clock_hz: int) -> float:
    """Inverse of :func:`hz_to_tone_period` (0 period -> 0 Hz)."""
    if tone_period <= 0:
        return 0.0
    return master_clock_hz / (16.0 * tone_period)


def quantize_tone(freq_hz: float, master_clock_hz: int) -> int:
    """Legal tone period for *freq_hz*, octave-folded into range to preserve pitch class.

    Notes below the chip's floor (period would exceed 4095) are folded **up** an octave at a
    time; notes above the ceiling (period < 1) are folded **down**. The pitch class is kept, so
    a too-low bass line reappears an octave higher rather than detuning or going silent.
    """
    tp = hz_to_tone_period(freq_hz, master_clock_hz)
    if tp <= 0:
        return 0
    while tp > TONE_PERIOD_MAX:
        tp //= 2  # raise an octave
    while tp < TONE_PERIOD_MIN:
        tp *= 2  # lower an octave
    return max(TONE_PERIOD_MIN, min(TONE_PERIOD_MAX, tp))


def velocity_to_amplitude(velocity: float) -> int:
    """Map a 0..1 perceptual loudness to a 0..15 amplitude (clamped)."""
    if velocity <= 0.0:
        return 0
    if velocity >= 1.0:
        return AMP_MAX
    return int(round(velocity * AMP_MAX))


def hz_to_noise_period(freq_hz: float, master_clock_hz: int) -> int:
    """Legal noise period whose centre frequency is nearest *freq_hz* (``f = clock/(16*NP)``)."""
    if freq_hz <= 0.0:
        return NOISE_PERIOD_MAX
    np_ = round(master_clock_hz / (16.0 * freq_hz))
    return max(NOISE_PERIOD_MIN, min(NOISE_PERIOD_MAX, np_))


def seconds_to_frame(t_s: float, frame_rate_hz: int) -> int:
    """Frame index for a time in seconds (rounded to nearest frame)."""
    return int(round(t_s * frame_rate_hz))


def frames_for_duration(duration_s: float, frame_rate_hz: int) -> int:
    """Number of 50 Hz frames needed to cover *duration_s* (at least one)."""
    return max(1, math.ceil(duration_s * frame_rate_hz))
