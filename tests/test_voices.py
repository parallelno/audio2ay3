"""Tests for instrument-aware voice priority (mapping/voices.py).

A quiet lead must win a scarce tone channel over a loud pad — this is what keeps the main theme
audible when bass and drums already hold two of the three voices. The GM program comes from MT3;
Basic Pitch leaves it ``None``, which must preserve the old loudness-only behaviour.
"""

from __future__ import annotations

from audio2ay3.analysis.model import Note
from audio2ay3.mapping.voices import _priority, _program_rank, _Span, allocate_voices


def test_program_rank_promotes_leads_and_demotes_pads():
    assert _program_rank(81) == 0  # Synth Lead (sawtooth)
    assert _program_rank(73) == 0  # Pipe (flute)
    assert _program_rank(66) == 0  # Reed (tenor sax)
    assert _program_rank(11) == 0  # Chromatic Percussion (vibraphone)
    assert _program_rank(89) == 2  # Synth Pad (warm pad)
    assert _program_rank(48) == 2  # Ensemble (string ensemble 1)
    assert _program_rank(0) == 1  # Piano -> ambiguous, stays neutral
    assert _program_rank(None) == 1  # Basic Pitch (unknown) -> neutral


def test_priority_ranks_a_quiet_lead_above_a_loud_pad():
    lead = _Span(0, 0, 1, 440.0, 0.2, program=81)
    pad = _Span(1, 0, 1, 440.0, 1.0, program=89)
    assert _priority(lead) < _priority(pad)


def test_unknown_program_falls_back_to_loudness_order():
    quiet = _Span(0, 0, 1, 440.0, 0.3, program=None)
    loud = _Span(1, 0, 1, 660.0, 0.9, program=None)
    assert _priority(loud) < _priority(quiet)


def test_allocator_keeps_the_lead_when_channels_are_scarce():
    # Channel A is reserved (bass owns it), leaving B/C for three simultaneous melodic notes.
    # The loud pads would win on loudness alone, but the quiet lead must survive on identity.
    lead = Note(0.0, 0.1, 660.0, 0.2, program=81)  # synth lead, quietest
    pad_loud = Note(0.0, 0.1, 440.0, 1.0, program=89)  # synth pad, loudest
    pad_mid = Note(0.0, 0.1, 550.0, 0.9, program=89)  # synth pad
    reserved = [0] * 5  # bass holds channel A every frame

    assignment = allocate_voices([pad_loud, pad_mid, lead], 50, 5, reserved=reserved)

    placed = {v.pitch_hz for v in assignment[0] if v is not None}
    assert 660.0 in placed  # the lead is kept
    assert 440.0 in placed  # the louder pad takes the other free channel
    assert 550.0 not in placed  # the quieter pad is the one dropped, not the lead
