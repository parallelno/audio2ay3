"""Tests for the voice-contention diagnostic (mapping/contention.py).

These are pure replays of the deterministic arrange-time policy on synthetic transcriptions —
no neural deps — so they can assert exact note-frame accounting for the "how many notes does the
3-channel budget drop, and would a 2nd AY recover them?" question.
"""

from __future__ import annotations

from audio2ay3.analysis.model import Note, Percussion, Transcription
from audio2ay3.config import RunConfig
from audio2ay3.mapping.contention import (
    DUAL_MELODIC_CHANNELS,
    describe_contention,
    voice_contention,
)


def test_single_note_has_no_contention():
    tr = Transcription(notes=[Note(0.0, 0.2, 440.0, 1.0)], duration_s=0.2)
    stats = voice_contention(tr, RunConfig())

    assert stats.frames == 10
    assert stats.melodic_notes == 1
    assert stats.notes_silenced == 0
    assert stats.demanded_note_frames == 10
    assert stats.sounded_note_frames == 10
    assert stats.dropped_note_frames == 0
    assert stats.contention_frames == 0
    assert stats.demand_hist == (0, 10, 0, 0, 0, 0)
    assert sum(stats.demand_hist) == stats.frames


def test_fourth_simultaneous_note_is_dropped_for_lack_of_channels():
    # Four notes sound together but the single chip has three tone channels and no bass/drums,
    # so the quietest note never reaches a voice.
    notes = [
        Note(0.0, 0.2, 400.0, 1.0),
        Note(0.0, 0.2, 500.0, 0.9),
        Note(0.0, 0.2, 600.0, 0.8),
        Note(0.0, 0.2, 700.0, 0.7),
    ]
    stats = voice_contention(Transcription(notes=notes, duration_s=0.2), RunConfig())

    assert stats.melodic_notes == 4
    assert stats.notes_silenced == 1  # the v=0.7 note is fully starved
    assert stats.demanded_note_frames == 40
    assert stats.sounded_note_frames == 30
    assert stats.dropped_capacity == 10
    assert stats.dropped_to_drums == 0
    assert stats.contention_frames == 10
    assert stats.demand_hist == (0, 0, 0, 0, 10, 0)
    # A second AY (4 melodic channels) would voice every note-frame.
    assert stats.dual_sounded_note_frames == 40
    assert stats.dual_notes_silenced == 0
    recovered = stats.dual_sounded_note_frames - stats.sounded_note_frames
    assert recovered == 10


def test_drum_overwrites_a_melodic_note_on_channel_c():
    # Three notes fill channels A/B/C; a kick steals channel C for its 4-frame decay, so the
    # note parked there is silenced exactly while the drum sounds.
    notes = [
        Note(0.0, 0.2, 400.0, 1.0),
        Note(0.0, 0.2, 500.0, 0.9),
        Note(0.0, 0.2, 600.0, 0.8),
    ]
    tr = Transcription(
        notes=notes, percussion=[Percussion(0.0, "kick")], duration_s=0.2
    )
    stats = voice_contention(tr, RunConfig())

    assert stats.drum_frames == 4
    assert stats.dropped_to_drums == 4
    assert stats.contention_frames == 4
    assert stats.dropped_capacity == 0  # the only loss is to the drum, not to capacity
    assert stats.notes_silenced == 0  # channel C's note still sounds once the drum decays


def test_bass_reserves_channel_a_and_squeezes_the_melody():
    # Bass owns channel A for its whole span, leaving only B/C for three simultaneous melodic
    # notes, so the lowest-priority note is dropped every frame.
    notes = [
        Note(0.0, 0.2, 400.0, 1.0),
        Note(0.0, 0.2, 500.0, 0.9),
        Note(0.0, 0.2, 600.0, 0.8),
    ]
    tr = Transcription(
        notes=notes, bass_notes=[Note(0.0, 0.2, 80.0, 1.0)], duration_s=0.2
    )
    stats = voice_contention(tr, RunConfig())

    assert stats.bass_frames == 10
    assert stats.notes_silenced == 1
    assert stats.demanded_note_frames == 30
    assert stats.sounded_note_frames == 20
    assert stats.dropped_capacity == 10
    # The dual-chip estimate frees a melodic channel per chip, recovering the squeezed note.
    assert stats.dual_sounded_note_frames == 30
    assert stats.dual_notes_silenced == 0


def test_empty_transcription_is_safe():
    stats = voice_contention(Transcription(), RunConfig())

    assert stats.melodic_notes == 0
    assert stats.demanded_note_frames == 0
    assert stats.notes_silenced == 0
    assert sum(stats.demand_hist) == stats.frames
    # describe_contention must not divide by zero on an empty song.
    text = describe_contention(stats)
    assert "Voice contention" in text
    assert "0.0%" in text


def test_describe_contention_reports_the_dual_chip_estimate():
    notes = [Note(0.0, 0.2, 400.0 + 50 * i, 1.0 - 0.1 * i) for i in range(5)]
    stats = voice_contention(Transcription(notes=notes, duration_s=0.2), RunConfig())

    text = describe_contention(stats)
    assert "2nd AY" in text
    assert "recovered vs single AY" in text
    # Five notes at once but only DUAL_MELODIC_CHANNELS voice under the estimate.
    assert stats.dual_sounded_note_frames == DUAL_MELODIC_CHANNELS * stats.frames


def test_chips_two_replays_the_dual_chip_budget():
    # With --chips 2 the arranger has six tone channels, so five simultaneous notes all sound
    # (vs. the single chip dropping the 4th/5th). The contention report must reflect that real
    # layout instead of the single-AY budget.
    from audio2ay3.config import ChipConfig

    notes = [Note(0.0, 0.2, 400.0 + 50 * i, 1.0 - 0.1 * i) for i in range(5)]
    cfg = RunConfig(chip=ChipConfig(n_chips=2))
    stats = voice_contention(Transcription(notes=notes, duration_s=0.2), cfg)

    assert stats.n_chips == 2
    assert stats.notes_silenced == 0          # six channels comfortably fit five notes
    assert stats.dropped_capacity == 0
    assert stats.contention_frames == 0
    assert stats.sounded_note_frames == stats.demanded_note_frames

    text = describe_contention(stats)
    assert "dual AY" in text
    # No forward-looking estimate at the 2-chip ceiling.
    assert "2nd AY" not in text
    assert "recovered vs single AY" not in text

