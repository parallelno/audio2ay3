"""Tests for the mapping stage: voice allocation and percussion overlay."""

from __future__ import annotations

from audio2ay3.analysis.model import Note, Percussion
from audio2ay3.encode.register_stream import RegisterStreamBuilder
from audio2ay3.mapping.percussion import apply_percussion
from audio2ay3.mapping.voices import allocate_voices, n_frames_for, place_bass


def test_empty_notes_allocate_to_silence():
    assignment = allocate_voices([], frame_rate_hz=50, n_frames=4)
    assert len(assignment) == 4
    assert all(slot is None for frame in assignment for slot in frame)


def test_single_note_keeps_one_channel_for_its_duration():
    note = Note(onset_s=0.0, duration_s=0.2, pitch_hz=200.0, velocity=0.8)
    assignment = allocate_voices([note], frame_rate_hz=50, n_frames=10)
    # frames 0..9 sound on exactly one channel, and it is the SAME channel each frame.
    channels_used = {
        ch for f in range(10) for ch in range(3) if assignment[f][ch] is not None
    }
    assert len(channels_used) == 1
    for f in range(10):
        voices = [v for v in assignment[f] if v is not None]
        assert len(voices) == 1
        assert voices[0].note_id == 0


def test_continuity_does_not_hop_channels_when_a_voice_joins():
    held = Note(onset_s=0.0, duration_s=0.2, pitch_hz=200.0, velocity=0.9)
    late = Note(onset_s=0.1, duration_s=0.1, pitch_hz=400.0, velocity=0.5)
    assignment = allocate_voices([held, late], frame_rate_hz=50, n_frames=10)

    def channel_of(frame, note_id):
        for ch in range(3):
            v = assignment[frame][ch]
            if v is not None and v.note_id == note_id:
                return ch
        return None

    # The held note never changes channel across the whole span.
    held_channels = {channel_of(f, 0) for f in range(10)}
    assert held_channels != {None}
    assert len([c for c in held_channels if c is not None]) == 1


def test_more_than_three_simultaneous_notes_drop_the_quietest():
    notes = [
        Note(0.0, 0.1, 100.0, velocity=1.0),
        Note(0.0, 0.1, 200.0, velocity=0.9),
        Note(0.0, 0.1, 300.0, velocity=0.8),
        Note(0.0, 0.1, 400.0, velocity=0.7),  # quietest -> dropped
    ]
    assignment = allocate_voices(notes, frame_rate_hz=50, n_frames=5)
    placed = {v.note_id for v in assignment[0] if v is not None}
    assert placed == {0, 1, 2}
    assert 3 not in placed


def test_n_frames_for_covers_latest_offset():
    notes = [Note(0.0, 0.5, 220.0), Note(1.0, 0.5, 440.0)]  # ends at 1.5s
    assert n_frames_for(notes, frame_rate_hz=50, duration_s=0.0) == 75


def test_apply_percussion_overlays_noise_decay_on_channel_c():
    builder = RegisterStreamBuilder(8)
    hits = [Percussion(onset_s=0.0, kind="kick", velocity=1.0)]
    apply_percussion(builder, hits, frame_rate_hz=50, n_frames=8)
    frames = builder.finish()

    # Channel C noise enabled on the onset frame, tone disabled, amplitude decaying.
    assert frames[0, 7] & (1 << 5) == 0  # noise-on-C enabled
    assert frames[0, 7] & (1 << 2) != 0  # tone-on-C disabled
    assert frames[0, 6] > 0  # a noise period was programmed
    decay = [int(frames[f, 10]) for f in range(4)]
    assert decay[0] == 15
    assert decay == sorted(decay, reverse=True)  # monotonically non-increasing


def test_apply_percussion_scales_with_velocity():
    loud = RegisterStreamBuilder(4)
    soft = RegisterStreamBuilder(4)
    apply_percussion(loud, [Percussion(0.0, "snare", 1.0)], 50, 4)
    apply_percussion(soft, [Percussion(0.0, "snare", 0.4)], 50, 4)
    assert loud.finish()[0, 10] > soft.finish()[0, 10]


def test_place_bass_reserves_channel_a_and_picks_the_lowest_note():
    low = Note(onset_s=0.0, duration_s=0.2, pitch_hz=80.0, velocity=0.9)
    high = Note(onset_s=0.0, duration_s=0.2, pitch_hz=160.0, velocity=1.0)
    voices, reserved = place_bass([low, high], frame_rate_hz=50, n_frames=10)
    # Bass sounds on every frame of its span and reserves channel A there.
    for f in range(10):
        assert reserved[f] == 0
        assert voices[f] is not None
        assert voices[f].pitch_hz == 80.0  # fundamental wins over the higher overlap


def test_place_bass_leaves_silent_frames_free():
    note = Note(onset_s=0.0, duration_s=0.06, pitch_hz=60.0, velocity=1.0)
    voices, reserved = place_bass([note], frame_rate_hz=50, n_frames=10)
    # Spans only the first 3 frames; the rest of the channel stays free for melody.
    assert [r is not None for r in reserved] == [True, True, True] + [False] * 7
    assert all(v is None for v in voices[3:])


def test_allocate_voices_never_uses_a_reserved_channel():
    note = Note(onset_s=0.0, duration_s=0.2, pitch_hz=440.0, velocity=1.0)
    reserved = [0] * 10  # channel A owned by bass for the whole span
    assignment = allocate_voices([note], frame_rate_hz=50, n_frames=10, reserved=reserved)
    for f in range(10):
        assert assignment[f][0] is None  # melody kept off the reserved channel
        placed = [ch for ch in range(3) if assignment[f][ch] is not None]
        assert placed and all(ch in (1, 2) for ch in placed)


def test_allocate_voices_resolves_per_frame_amp_scale_from_contour():
    # A four-frame note carrying a source contour: each frame's voice exposes that frame's value.
    note = Note(
        onset_s=0.0, duration_s=0.08, pitch_hz=440.0, velocity=1.0,
        amp_contour=(1.0, 0.75, 0.5, 0.25),
    )
    assignment = allocate_voices([note], frame_rate_hz=50, n_frames=4)
    scales = []
    for f in range(4):
        voice = next(v for v in assignment[f] if v is not None)
        scales.append(voice.amp_scale)
    assert scales == [1.0, 0.75, 0.5, 0.25]


def test_allocate_voices_amp_scale_none_without_contour():
    note = Note(onset_s=0.0, duration_s=0.1, pitch_hz=440.0, velocity=1.0)
    assignment = allocate_voices([note], frame_rate_hz=50, n_frames=5)
    voice = next(v for v in assignment[0] if v is not None)
    assert voice.amp_scale is None


def test_place_bass_resolves_per_frame_amp_scale_from_contour():
    note = Note(
        onset_s=0.0, duration_s=0.06, pitch_hz=60.0, velocity=1.0,
        amp_contour=(1.0, 0.6, 0.2),
    )
    voices, _ = place_bass([note], frame_rate_hz=50, n_frames=5)
    assert [v.amp_scale for v in voices[:3]] == [1.0, 0.6, 0.2]


