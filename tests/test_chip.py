"""Unit tests for the AY-3-8910 emulator core."""

from __future__ import annotations

import numpy as np

from audio2ay3.chip import AY_DAC, Ay3Emulator
from audio2ay3.config import ChipConfig

CLOCK = 1_773_400


def _tone_frames(tp: int, n: int = 30, chan: int = 0, level: int = 15) -> np.ndarray:
    frames = np.zeros((n, 16), dtype=np.uint8)
    frames[:, 0 + 2 * chan] = tp & 0xFF
    frames[:, 1 + 2 * chan] = (tp >> 8) & 0x0F
    mixer = 0x3F & ~(1 << chan)  # enable tone on this channel only (active-low)
    frames[:, 7] = mixer
    frames[:, 8 + chan] = level
    frames[:, 13] = 0xFF  # do not retrigger envelope
    return frames


def _peak_freq(pcm: np.ndarray, sr: int) -> float:
    pcm = pcm - pcm.mean()
    win = np.hanning(pcm.size)
    spec = np.abs(np.fft.rfft(pcm * win))
    freqs = np.fft.rfftfreq(pcm.size, 1.0 / sr)
    return float(freqs[np.argmax(spec)])


def test_tone_frequency_a4_within_a_few_hz():
    tp = round(CLOCK / (16 * 440.0))
    frames = _tone_frames(tp, n=40)
    emu = Ay3Emulator(ChipConfig(master_clock_hz=CLOCK), render_sr=44_100, oversample=2)
    pcm = emu.render_frames(frames, CLOCK, 50)
    assert abs(_peak_freq(pcm, 44_100) - 440.0) < 10.0


def test_tone_frequency_scales_with_period():
    # Halving the period should roughly double the frequency.
    tp = round(CLOCK / (16 * 220.0))
    emu = Ay3Emulator(ChipConfig(master_clock_hz=CLOCK), render_sr=44_100, oversample=2)
    f_low = _peak_freq(emu.render_frames(_tone_frames(tp, n=40), CLOCK, 50), 44_100)
    f_high = _peak_freq(emu.render_frames(_tone_frames(tp // 2, n=40), CLOCK, 50), 44_100)
    assert 1.8 < (f_high / f_low) < 2.2


def test_silence_is_silent():
    frames = np.zeros((10, 16), dtype=np.uint8)
    frames[:, 7] = 0x3F  # all tone + noise disabled
    frames[:, 13] = 0xFF
    pcm = Ay3Emulator().render_frames(frames, CLOCK, 50)
    assert np.max(np.abs(pcm)) < 1e-6


def test_disabled_tone_produces_no_oscillation():
    frames = _tone_frames(300, n=10)
    frames[:, 7] = 0x3F  # disable everything; channel A holds a constant DC level
    pcm = Ay3Emulator().render_frames(frames, CLOCK, 50)
    assert np.max(np.abs(pcm - pcm.mean())) < 1e-6


def test_dac_table_is_monotonic_and_normalised():
    assert AY_DAC[0] == 0.0
    assert AY_DAC[-1] == 1.0
    assert np.all(np.diff(AY_DAC) > 0)


def test_noise_channel_produces_broadband_output():
    frames = np.zeros((20, 16), dtype=np.uint8)
    frames[:, 6] = 5  # noise period
    frames[:, 7] = 0x3F & ~(1 << 3)  # enable noise on channel A (clear bit 3)
    frames[:, 8] = 15
    frames[:, 13] = 0xFF
    pcm = Ay3Emulator().render_frames(frames, CLOCK, 50)
    assert np.std(pcm) > 0.01  # noise has real variance
