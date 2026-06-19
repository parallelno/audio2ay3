"""Tests for the deterministic arrange() core of the conversion pipeline.

These exercise the neural-free half end-to-end: a synthetic Transcription becomes a
hardware-legal YmSong that the emulator can render. The neural front-end (load/separate/
transcribe) needs the optional [neural] extra and is not unit-tested here.
"""

from __future__ import annotations

import numpy as np

from audio2ay3.analysis.model import Note, Percussion, Transcription
from audio2ay3.config import RunConfig
from audio2ay3.encode.quantize import quantize_tone
from audio2ay3.pipeline import arrange
from audio2ay3.render import Renderer

CLOCK = 1_773_400


def test_arrange_produces_legal_song_with_expected_length():
    tr = Transcription(
        notes=[Note(0.0, 0.1, 440.0, 1.0)], percussion=[], duration_s=0.0
    )
    song = arrange(tr, RunConfig(), name="x")
    assert song.frames.shape == (5, 16)  # ceil(0.1s * 50)
    assert song.master_clock == CLOCK
    assert song.frame_rate == 50
    assert song.version == "YM6"
    # Reserved/IO registers stay clean, envelope left untouched.
    assert np.all(song.frames[:, 13] == 0xFF)


def test_arrange_maps_pitch_to_correct_tone_period():
    tr = Transcription(notes=[Note(0.0, 0.1, 440.0, 1.0)])
    song = arrange(tr, RunConfig())
    expected_tp = quantize_tone(440.0, CLOCK)  # 252
    assert song.frames[0, 0] == (expected_tp & 0xFF)
    assert song.frames[0, 1] == ((expected_tp >> 8) & 0x0F)
    assert song.frames[0, 8] == 15  # velocity 1.0 -> full amplitude
    assert song.frames[0, 7] & 0x01 == 0  # tone A enabled


def test_arrange_is_renderable_and_audible():
    tr = Transcription(notes=[Note(0.0, 0.3, 330.0, 1.0)])
    song = arrange(tr, RunConfig())
    pcm = Renderer(render_sr=22_050, oversample=1).render(song)
    assert pcm.size == int(song.n_frames * 22_050 / 50)
    assert np.max(np.abs(pcm)) > 0.1  # not silent


def test_arrange_percussion_only_programs_noise():
    tr = Transcription(
        notes=[], percussion=[Percussion(0.0, "snare", 1.0)], duration_s=0.2
    )
    song = arrange(tr, RunConfig())
    # The snare frame routes noise to channel C (mixer bit 5 cleared).
    assert song.frames[0, 7] & (1 << 5) == 0
    assert song.frames[0, 6] > 0  # noise period set


def test_arrange_empty_transcription_is_one_silent_frame():
    song = arrange(Transcription(), RunConfig())
    assert song.n_frames == 1
    assert np.all(song.frames[:, 7] == 0x3F)  # everything disabled
