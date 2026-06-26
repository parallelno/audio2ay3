"""Tests for the pre-arrangement MIDI export (audio2ay3.analysis.midi_export)."""

from __future__ import annotations

import sys

import pytest

from audio2ay3.analysis.midi_export import _hz_to_midi, _vel_to_midi, write_transcription_midi
from audio2ay3.analysis.model import Note, Percussion, Transcription

mido = pytest.importorskip("mido")


def _messages(path):
    """Flatten every (track_name, message) pair from a saved MIDI file."""
    mid = mido.MidiFile(str(path))
    out = []
    for track in mid.tracks:
        name = ""
        for msg in track:
            if msg.type == "track_name":
                name = msg.name
            out.append((name, msg))
    return out


def test_hz_to_midi_maps_concert_a():
    assert _hz_to_midi(440.0) == 69
    assert _hz_to_midi(880.0) == 81
    assert _hz_to_midi(0.0) == 0  # silence-safe


def test_vel_clamps_to_audible_range():
    assert _vel_to_midi(1.0) == 127
    assert _vel_to_midi(0.0) == 1  # never a silent note-on
    assert _vel_to_midi(0.5) == 64


def test_writes_one_track_per_stem(tmp_path):
    tr = Transcription(
        notes=[
            Note(onset_s=0.0, duration_s=0.5, pitch_hz=440.0, stem="melody"),
            Note(onset_s=0.5, duration_s=0.5, pitch_hz=660.0, stem="vocals"),
        ],
        bass_notes=[Note(onset_s=0.0, duration_s=1.0, pitch_hz=110.0, stem="bass")],
        percussion=[Percussion(onset_s=0.0, kind="kick")],
        duration_s=1.0,
    )
    out = write_transcription_midi(tr, tmp_path / "song.mid", frame_rate_hz=50)
    assert out.exists()

    names = {name for name, _ in _messages(out)}
    assert {"Melody", "Vocals", "Bass", "Drums"} <= names


def test_note_pitch_and_velocity(tmp_path):
    tr = Transcription(
        notes=[Note(onset_s=0.0, duration_s=1.0, pitch_hz=440.0, velocity=1.0, stem="melody")],
        duration_s=1.0,
    )
    out = write_transcription_midi(tr, tmp_path / "a.mid", frame_rate_hz=50)
    note_ons = [m for _, m in _messages(out) if m.type == "note_on" and m.velocity > 0]
    assert len(note_ons) == 1
    assert note_ons[0].note == 69
    assert note_ons[0].velocity == 127


def test_amp_contour_becomes_cc11(tmp_path):
    tr = Transcription(
        notes=[
            Note(
                onset_s=0.0,
                duration_s=0.1,
                pitch_hz=440.0,
                amp_contour=(1.0, 0.5, 0.25, 0.1, 0.0),
                stem="melody",
            )
        ],
        duration_s=0.1,
    )
    out = write_transcription_midi(tr, tmp_path / "cc.mid", frame_rate_hz=50)
    cc11 = [m for _, m in _messages(out)
            if m.type == "control_change" and m.control == 11]
    # Distinct contour levels -> several expression events (repeats are de-duplicated).
    assert len(cc11) >= 4
    assert max(m.value for m in cc11) == 127


def test_program_change_emitted_when_known(tmp_path):
    tr = Transcription(
        notes=[Note(onset_s=0.0, duration_s=1.0, pitch_hz=440.0, program=42, stem="melody")],
        duration_s=1.0,
    )
    out = write_transcription_midi(tr, tmp_path / "p.mid", frame_rate_hz=50)
    pcs = [m for _, m in _messages(out) if m.type == "program_change"]
    assert any(m.program == 42 for m in pcs)


def test_no_program_change_when_unknown(tmp_path):
    tr = Transcription(
        notes=[Note(onset_s=0.0, duration_s=1.0, pitch_hz=440.0, program=None, stem="melody")],
        duration_s=1.0,
    )
    out = write_transcription_midi(tr, tmp_path / "np.mid", frame_rate_hz=50)
    pcs = [m for _, m in _messages(out) if m.type == "program_change"]
    assert pcs == []


def test_percussion_on_channel_10(tmp_path):
    tr = Transcription(
        percussion=[
            Percussion(onset_s=0.0, kind="kick"),
            Percussion(onset_s=0.1, kind="snare"),
            Percussion(onset_s=0.2, kind="hat"),
        ],
        duration_s=0.3,
    )
    out = write_transcription_midi(tr, tmp_path / "d.mid", frame_rate_hz=50)
    drums = [m for _, m in _messages(out)
             if m.type == "note_on" and m.velocity > 0]
    assert {m.note for m in drums} == {36, 38, 42}
    assert all(m.channel == 9 for m in drums)


def test_empty_transcription_still_valid(tmp_path):
    out = write_transcription_midi(Transcription(), tmp_path / "empty.mid", frame_rate_hz=50)
    assert out.exists()
    mido.MidiFile(str(out))  # parses without error


def test_missing_mido_raises_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mido", None)
    with pytest.raises(RuntimeError, match=r"audio2ay3\[midi\]"):
        write_transcription_midi(Transcription(), tmp_path / "x.mid", frame_rate_hz=50)
