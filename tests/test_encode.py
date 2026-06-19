"""Tests for the encode stage: Hz/loudness quantisation and register assembly."""

from __future__ import annotations

import numpy as np

from audio2ay3.encode.quantize import (
    AMP_MAX,
    NOISE_PERIOD_MAX,
    TONE_PERIOD_MAX,
    frames_for_duration,
    hz_to_noise_period,
    hz_to_tone_period,
    quantize_tone,
    seconds_to_frame,
    tone_period_to_hz,
    velocity_to_amplitude,
)
from audio2ay3.encode.register_stream import ENV_NO_RETRIGGER, RegisterStreamBuilder

CLOCK = 1_773_400


def test_hz_tone_period_roundtrip_a4():
    tp = hz_to_tone_period(440.0, CLOCK)
    assert tp == 252  # round(1773400 / (16 * 440))
    assert abs(tone_period_to_hz(tp, CLOCK) - 440.0) < 1.0


def test_hz_to_tone_period_non_positive_is_zero():
    assert hz_to_tone_period(0.0, CLOCK) == 0
    assert hz_to_tone_period(-5.0, CLOCK) == 0


def test_quantize_tone_folds_subsonic_up_an_octave():
    # 20 Hz would need a period above the 12-bit ceiling; it must fold up, not clip to silence.
    raw = hz_to_tone_period(20.0, CLOCK)
    assert raw > TONE_PERIOD_MAX
    tp = quantize_tone(20.0, CLOCK)
    assert 1 <= tp <= TONE_PERIOD_MAX
    # Folded pitch is the original times a power of two (same pitch class), to within the
    # rounding error of integer period division (well under a cent here).
    ratio = tone_period_to_hz(tp, CLOCK) / 20.0
    octaves = np.log2(ratio)
    assert abs(octaves - round(octaves)) < 0.01
    assert round(octaves) >= 1  # actually folded up


def test_velocity_to_amplitude_endpoints_and_mid():
    assert velocity_to_amplitude(0.0) == 0
    assert velocity_to_amplitude(1.0) == AMP_MAX
    assert velocity_to_amplitude(2.0) == AMP_MAX  # clamped
    assert velocity_to_amplitude(0.5) == 8  # round(0.5 * 15)


def test_noise_period_clamped():
    assert 0 <= hz_to_noise_period(50.0, CLOCK) <= NOISE_PERIOD_MAX
    assert hz_to_noise_period(0.0, CLOCK) == NOISE_PERIOD_MAX  # silence -> lowest hiss


def test_seconds_and_duration_to_frames():
    assert seconds_to_frame(0.1, 50) == 5
    assert frames_for_duration(0.5, 50) == 25
    assert frames_for_duration(0.0, 50) == 1  # always at least one frame


def test_builder_defaults_are_silent_and_legal():
    frames = RegisterStreamBuilder(3).finish()
    assert frames.shape == (3, 16)
    assert np.all(frames[:, 13] == ENV_NO_RETRIGGER)
    assert np.all(frames[:, 7] == 0x3F)  # all tone+noise disabled
    assert np.all(frames[:, :7] == 0)


def test_set_tone_writes_period_amp_and_enables_mixer():
    b = RegisterStreamBuilder(1)
    b.set_tone(0, 0, 0x123, 15)
    frames = b.finish()
    assert frames[0, 0] == 0x23  # fine
    assert frames[0, 1] == 0x01  # coarse nibble
    assert frames[0, 8] == 15
    assert frames[0, 7] & 0x01 == 0  # channel A tone-enable bit cleared


def test_set_tone_clamps_and_masks_period():
    b = RegisterStreamBuilder(1)
    b.set_tone(0, 1, 9999, 99)  # over-range period and amplitude
    frames = b.finish()
    assert frames[0, 2] == 0xFF  # fine of clamped 0xFFF
    assert frames[0, 3] == 0x0F  # coarse nibble of 0xFFF
    assert frames[0, 9] == AMP_MAX


def test_set_tone_envelope_flag_sets_bit4():
    b = RegisterStreamBuilder(1)
    b.set_tone(0, 2, 100, 10, use_envelope=True)
    frames = b.finish()
    assert frames[0, 10] == (10 | 0x10)


def test_enable_noise_routes_and_sets_period():
    b = RegisterStreamBuilder(1)
    b.enable_noise(0, 2, 20)
    frames = b.finish()
    assert frames[0, 6] == 20
    assert frames[0, 7] & (1 << (3 + 2)) == 0  # noise-on-C bit cleared


def test_set_envelope_writes_period_and_shape():
    b = RegisterStreamBuilder(1)
    b.set_envelope(0, 0x1234, 8)
    frames = b.finish()
    assert frames[0, 11] == 0x34
    assert frames[0, 12] == 0x12
    assert frames[0, 13] == 8
