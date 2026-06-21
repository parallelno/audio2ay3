"""Tests for dual-AY (two-chip) support: config, allocation, arrange, split, render, explain.

All of these exercise the deterministic, pure-numpy path (no numba/torch), so they run on the
core suite. The real emulator core is covered elsewhere; the renderer split here mocks
``render_frames`` so the per-chip sum can be asserted without invoking the JIT.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio2ay3.analysis.model import Note, Percussion, Transcription
from audio2ay3.chip.ay3_8910 import Ay3Emulator
from audio2ay3.config import ChipConfig, RunConfig
from audio2ay3.explain import song_stats
from audio2ay3.mapping.voices import allocate_voices
from audio2ay3.pipeline import arrange
from audio2ay3.ymformat.model import YmSong


def test_chip_config_validates_n_chips():
    assert ChipConfig(n_chips=1).total_tone_channels == 3
    assert ChipConfig(n_chips=2).total_tone_channels == 6
    with pytest.raises(ValueError):
        ChipConfig(n_chips=3)
    with pytest.raises(ValueError):
        ChipConfig(n_chips=0)


def test_allocate_voices_spreads_over_six_channels():
    # Six notes sound together; with six channels every one gets a voice, with three only three.
    notes = [Note(0.0, 0.2, 200.0 + 50 * i, 1.0 - 0.05 * i) for i in range(6)]
    six = allocate_voices(notes, 50, 10, n_channels=6)
    three = allocate_voices(notes, 50, 10, n_channels=3)
    assert sum(v is not None for v in six[0]) == 6
    assert sum(v is not None for v in three[0]) == 3


def test_ymsong_per_chip_split():
    frames = np.zeros((4, 32), dtype=np.uint8)
    frames[:, :16] = 1  # chip 0 marker
    frames[:, 16:] = 2  # chip 1 marker
    song = YmSong(frames=frames, n_chips=2)

    blocks = song.per_chip_frames()
    assert len(blocks) == 2
    assert blocks[0].shape == (4, 16) and np.all(blocks[0] == 1)
    assert blocks[1].shape == (4, 16) and np.all(blocks[1] == 2)

    chip_songs = song.per_chip_songs()
    assert [cs.n_chips for cs in chip_songs] == [1, 1]
    assert chip_songs[0].frames.shape == (4, 16)
    assert np.all(chip_songs[1].frames == 2)


def _dense_transcription() -> Transcription:
    # Five melodic notes plus a bass note all sounding at once: a single chip (bass + 2 melody)
    # can voice three, dual-AY (bass + 5 melody) can voice all six.
    notes = [Note(0.0, 0.2, 300.0 + 60 * i, 1.0 - 0.05 * i) for i in range(5)]
    bass = [Note(0.0, 0.2, 80.0, 1.0)]
    return Transcription(notes=notes, bass_notes=bass, duration_s=0.2)


def test_arrange_single_chip_unchanged_shape():
    song = arrange(_dense_transcription(), RunConfig(chip=ChipConfig(n_chips=1)))
    assert song.n_chips == 1
    assert song.frames.shape[1] == 16
    assert max(range(len(song_stats(song).poly))) == 3  # poly indexed 0..3


def test_arrange_dual_chip_voices_more_notes():
    dual = arrange(_dense_transcription(), RunConfig(chip=ChipConfig(n_chips=2)))
    single = arrange(_dense_transcription(), RunConfig(chip=ChipConfig(n_chips=1)))

    assert dual.n_chips == 2
    assert dual.frames.shape[1] == 32

    s_dual = song_stats(dual)
    s_single = song_stats(single)
    # Dual reports across six channels (poly 0..6, one tone-on entry per channel).
    assert len(s_dual.poly) == 7
    assert len(s_dual.tone_on) == 6
    # The dense frame voices all six on the dual chip but only three on a single chip.
    assert max(i for i, c in enumerate(s_dual.poly) if c) == 6
    assert max(i for i, c in enumerate(s_single.poly) if c) == 3
    # The second chip's three channels actually carry tone.
    assert all(s_dual.tone_on[ch] > 0 for ch in (3, 4, 5))


def test_arrange_dual_isolates_percussion_on_second_chip():
    tr = Transcription(
        notes=[Note(0.0, 0.4, 440.0, 1.0)],
        percussion=[Percussion(0.1, "snare", 1.0)],
        duration_s=0.4,
    )
    song = arrange(tr, RunConfig(chip=ChipConfig(n_chips=2)))
    f = 5  # 0.1s * 50fps -> frame 5
    chip0_mixer = song.frames[f, 7]
    chip1_mixer = song.frames[f, 16 + 7]
    # The snare routes noise to chip 1 channel C (mixer noise-disable bit 5 cleared) and leaves
    # chip 0's noise generator untouched, so drums never steal a melodic channel.
    assert ((chip1_mixer >> 5) & 1) == 0
    assert ((chip0_mixer >> 5) & 1) == 1


def test_render_song_sums_chips(monkeypatch):
    frames = np.zeros((3, 32), dtype=np.uint8)
    frames[:, 0] = 10  # chip 0 marker in R0
    frames[:, 16] = 20  # chip 1 marker in R0
    song = YmSong(frames=frames, n_chips=2)

    emu = Ay3Emulator()

    def fake_render_frames(block, master_clock, frame_rate):
        return np.full(8, float(block[0, 0]), dtype=np.float32)

    monkeypatch.setattr(emu, "render_frames", fake_render_frames)
    pcm = emu.render_song(song)
    # (10 + 20) / 2 chips = 15 everywhere.
    assert np.allclose(pcm, 15.0)
