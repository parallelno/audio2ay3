"""Tests for the MT3 multitrack backend's deterministic glue.

The heavy MT3/T5X/JAX stack is never imported here: we exercise the pure ``NoteSequence`` -> IR
converter with a synthetic, duck-typed note sequence, the GM drum-key bucketing, and the
dispatch/error surface of :func:`transcribe`. Actual MT3 inference is validated out-of-band on a
machine with the ``[mt3]`` extra installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from audio2ay3.analysis.transcribe import (
    _drum_kind_for_midi,
    _midi_to_hz,
    note_sequence_to_transcription,
    transcribe,
)


def _note(pitch, start, end, *, velocity=100, program=0, is_drum=False):
    return SimpleNamespace(
        pitch=pitch,
        start_time=start,
        end_time=end,
        velocity=velocity,
        program=program,
        is_drum=is_drum,
    )


def _ns(notes, total_time=0.0):
    return SimpleNamespace(notes=notes, total_time=total_time)


def test_drum_kind_for_midi_buckets_into_kick_snare_hat():
    assert _drum_kind_for_midi(35) == "kick"
    assert _drum_kind_for_midi(36) == "kick"
    assert _drum_kind_for_midi(38) == "snare"
    assert _drum_kind_for_midi(40) == "snare"
    assert _drum_kind_for_midi(42) == "hat"  # closed hi-hat
    assert _drum_kind_for_midi(49) == "hat"  # crash cymbal falls through to hat


def test_note_sequence_routes_drums_bass_and_melody():
    ns = _ns(
        [
            _note(60, 0.0, 0.5, program=0),  # melody (piano)
            _note(40, 0.0, 1.0, program=34),  # GM bass family -> bass_notes
            _note(36, 0.25, 0.30, is_drum=True),  # kick
            _note(38, 0.50, 0.55, is_drum=True),  # snare
            _note(42, 0.75, 0.80, is_drum=True),  # hat
        ],
        total_time=1.0,
    )
    tr = note_sequence_to_transcription(ns)

    assert [round(n.pitch_hz, 1) for n in tr.notes] == [round(_midi_to_hz(60), 1)]
    assert [round(n.pitch_hz, 1) for n in tr.bass_notes] == [round(_midi_to_hz(40), 1)]
    assert [p.kind for p in tr.percussion] == ["kick", "snare", "hat"]
    assert [round(p.onset_s, 2) for p in tr.percussion] == [0.25, 0.50, 0.75]
    assert tr.duration_s == 1.0  # the longest note offset


def test_note_sequence_maps_velocity_and_timing():
    ns = _ns([_note(69, 0.1, 0.6, velocity=64)], total_time=0.0)
    tr = note_sequence_to_transcription(ns)
    note = tr.notes[0]
    assert note.pitch_hz == pytest.approx(440.0)  # MIDI 69 = A4
    assert note.onset_s == pytest.approx(0.1)
    assert note.duration_s == pytest.approx(0.5)
    assert note.velocity == pytest.approx(64 / 127)
    # total_time is 0, so duration comes from the note's end.
    assert tr.duration_s == pytest.approx(0.6)


def test_note_sequence_zero_velocity_falls_back_to_full():
    # MT3's multi-instrument checkpoint uses a single velocity bin, so a 0/absent velocity must
    # not silence the note — it floors to full loudness.
    tr = note_sequence_to_transcription(_ns([_note(60, 0.0, 0.5, velocity=0)]))
    assert tr.notes[0].velocity == 1.0


def test_note_sequence_empty_is_safe():
    tr = note_sequence_to_transcription(_ns([], total_time=2.0))
    assert tr.notes == [] and tr.bass_notes == [] and tr.percussion == []
    assert tr.duration_s == 2.0


def test_transcribe_onsets_frames_still_reserved():
    with pytest.raises(NotImplementedError):
        transcribe(np.zeros(16, dtype=np.float32), 16000, "onsets-frames")


def test_transcribe_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        transcribe(np.zeros(16, dtype=np.float32), 16000, "nope")


def test_transcribe_mt3_without_extra_raises_actionable_runtime_error():
    # No JAX/T5X/mt3 installed here: the call must surface a clean RuntimeError naming the extra,
    # not a raw ImportError from deep inside the stack.
    with pytest.raises(RuntimeError, match="mt3"):
        transcribe(np.zeros(16, dtype=np.float32), 16000, "mt3")
