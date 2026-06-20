"""Tests for the per-note loudness contour extraction (analysis.dynamics).

Pure NumPy: the only neural dependency is the audio that production hands in, so the maths is
exercised directly here with synthetic signals.
"""

from __future__ import annotations

import numpy as np

from audio2ay3.analysis.dynamics import (
    attach_amp_contours,
    frame_rms_envelope,
)
from audio2ay3.analysis.model import Note


def test_frame_rms_envelope_tracks_a_decay():
    sr = 1000
    # A one-second signal whose amplitude ramps from loud to silent.
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (1.0 - t) * np.sin(2 * np.pi * 50 * t)
    rms = frame_rms_envelope(audio, sr=sr, frame_rate_hz=50)
    assert rms.size == 50
    # Monotone-ish downward trend: the start is clearly louder than the end.
    assert rms[0] > rms[-1]
    assert rms[5] > rms[40]


def test_frame_rms_envelope_empty_audio_is_empty():
    assert frame_rms_envelope(np.zeros(0), sr=44_100, frame_rate_hz=50).size == 0
    assert frame_rms_envelope(np.ones(10), sr=0, frame_rate_hz=50).size == 0


def test_attach_amp_contours_normalises_to_note_peak():
    sr = 1000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (1.0 - t) * np.sin(2 * np.pi * 50 * t)  # decaying
    note = Note(onset_s=0.0, duration_s=1.0, pitch_hz=200.0, velocity=0.9)
    [shaped] = attach_amp_contours([note], audio, sr=sr, frame_rate_hz=50)
    contour = shaped.amp_contour
    assert contour  # non-empty
    # Other fields are preserved.
    assert shaped.velocity == 0.9
    assert shaped.pitch_hz == 200.0
    # Normalised: the loudest frame reaches 1.0 and nothing exceeds it; the note decays.
    assert max(contour) == 1.0
    assert all(0.0 <= v <= 1.0 for v in contour)
    assert contour[0] > contour[-1]


def test_attach_amp_contours_silent_note_gets_no_contour():
    sr = 1000
    audio = np.zeros(sr)
    note = Note(onset_s=0.0, duration_s=0.5, pitch_hz=200.0, velocity=0.5)
    [shaped] = attach_amp_contours([note], audio, sr=sr, frame_rate_hz=50)
    assert shaped.amp_contour == ()


def test_attach_amp_contours_empty_audio_returns_notes_unchanged():
    note = Note(onset_s=0.0, duration_s=0.5, pitch_hz=200.0, velocity=0.5)
    out = attach_amp_contours([note], np.zeros(0), sr=44_100, frame_rate_hz=50)
    assert out == [note]
    assert out[0].amp_contour == ()
