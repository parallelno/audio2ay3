"""Tests for the --explain register-level diagnostics."""

from __future__ import annotations

import numpy as np

from audio2ay3.encode.register_stream import RegisterStreamBuilder
from audio2ay3.explain import describe_song, song_stats
from audio2ay3.ymformat.model import YmSong


def test_song_stats_summarises_voices_noise_and_dynamics():
    b = RegisterStreamBuilder(4)
    # Channel A: audible all 4 frames; period 100 then 200; amplitude 15 then 10.
    b.set_tone(0, 0, 100, 15)
    b.set_tone(1, 0, 100, 15)
    b.set_tone(2, 0, 200, 10)
    b.set_tone(3, 0, 200, 10)
    # Channel B: audible only frames 0-1.
    b.set_tone(0, 1, 300, 12)
    b.set_tone(1, 1, 300, 12)
    # Channel C: a noise-only hit on frame 0 (tone stays disabled).
    b.enable_noise(0, 2, 16)
    b.set_amplitude(0, 2, 14)
    song = YmSong(frames=b.finish())

    s = song_stats(song)
    assert s.frames == 4
    assert s.tone_on == (4, 2, 0)  # C carries no tone voice (noise only)
    assert s.poly == (0, 2, 2, 0)  # frames 0-1 have 2 tone voices, frames 2-3 have 1
    assert s.noise_frames == 1
    assert s.bass_distinct_periods == 2  # channel A moved 100 -> 200
    assert s.amp_changes[0] == 2  # onset (15) then the change to 10
    assert s.distinct_amp_levels == 3  # 15, 10 (A) and 12 (B); C's 14 isn't an audible tone


def test_describe_song_is_human_readable():
    b = RegisterStreamBuilder(2)
    b.set_tone(0, 0, 100, 15)
    b.set_tone(1, 0, 100, 15)
    text = describe_song(YmSong(frames=b.finish()))
    assert "explain:" in text
    assert "tone-on A/B/C" in text
    assert "polyphony" in text


def test_song_stats_handles_empty_song():
    song = YmSong(frames=np.zeros((0, 16), dtype=np.uint8))
    assert song_stats(song).frames == 0
    assert "empty" in describe_song(song)
