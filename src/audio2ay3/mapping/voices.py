"""Allocate transcribed notes to the three AY tone channels, frame by frame.

The skeleton policy is a greedy allocator with **continuity**: a note already sounding on a
channel stays there while it lasts, so we avoid the frame-to-frame channel hopping that causes
audible warble. Free channels are filled by priority (louder first, with a mild bass bias), and
when more than three notes sound at once the quietest are dropped for that frame.

The output is intentionally a plain per-frame structure so the encode stage stays the only place
that touches registers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.model import Note
from ..encode.quantize import frames_for_duration, seconds_to_frame

N_CHANNELS = 3

# Channel A is the home of the dedicated bass voice; B and C carry lead/harmony (and C also
# hosts the percussion noise). Keeping bass on a fixed channel maximises its continuity.
BASS_CHANNEL = 0


@dataclass(frozen=True)
class Voice:
    """A note placed on a channel for one frame."""

    pitch_hz: float
    velocity: float
    note_id: int


@dataclass
class _Span:
    note_id: int
    start: int
    end: int  # exclusive
    pitch_hz: float
    velocity: float


def _priority(span: _Span) -> tuple[float, float]:
    # Louder first; ties broken toward lower pitch (bass bias keeps the foundation).
    return (-span.velocity, span.pitch_hz)


def _spans_from_notes(
    notes: list[Note], frame_rate_hz: int, n_frames: int
) -> list[_Span]:
    """Clip each note to a half-open ``[start, end)`` frame span (>=1 audible frame)."""
    spans: list[_Span] = []
    for note_id, note in enumerate(notes):
        start = seconds_to_frame(note.onset_s, frame_rate_hz)
        end = seconds_to_frame(note.offset_s, frame_rate_hz)
        if end <= start:
            end = start + 1  # guarantee at least one audible frame
        start = max(0, start)
        end = min(n_frames, end)
        if start < end:
            spans.append(_Span(note_id, start, end, note.pitch_hz, note.velocity))
    return spans


def _bucket_by_frame(spans: list[_Span], n_frames: int) -> list[list[_Span]]:
    """Bucket spans by the frames they touch for an O(notes + frames) sweep."""
    active_by_frame: list[list[_Span]] = [[] for _ in range(n_frames)]
    for span in spans:
        for f in range(span.start, span.end):
            active_by_frame[f].append(span)
    return active_by_frame


def place_bass(
    bass_notes: list[Note],
    frame_rate_hz: int,
    n_frames: int,
    channel: int = BASS_CHANNEL,
) -> tuple[list[Voice | None], list[int | None]]:
    """Lay the bass line onto one dedicated *channel*, one note per frame.

    Returns ``(bass_voices, reserved)`` where ``bass_voices[f]`` is the bass :class:`Voice`
    (or ``None`` when silent) and ``reserved[f]`` is *channel* on frames a bass note sounds —
    the signal the melodic allocator uses to leave that channel alone — or ``None`` when the
    channel is free. When bass notes overlap, the lowest pitch wins: the fundamental, never a
    transcribed harmonic.
    """
    bass_voices: list[Voice | None] = [None] * n_frames
    reserved: list[int | None] = [None] * n_frames
    active_by_frame = _bucket_by_frame(
        _spans_from_notes(bass_notes, frame_rate_hz, n_frames), n_frames
    )
    for f in range(n_frames):
        active = active_by_frame[f]
        if not active:
            continue
        s = min(active, key=lambda sp: sp.pitch_hz)
        bass_voices[f] = Voice(s.pitch_hz, s.velocity, s.note_id)
        reserved[f] = channel
    return bass_voices, reserved


def allocate_voices(
    notes: list[Note],
    frame_rate_hz: int,
    n_frames: int,
    reserved: list[int | None] | None = None,
) -> list[list[Voice | None]]:
    """Return ``assignment[frame][channel]`` of :class:`Voice` or ``None`` (silent).

    When *reserved* is given, ``reserved[f]`` names a channel that is off-limits to melodic
    notes in frame *f* (because :func:`place_bass` owns it that frame); the remaining channels
    absorb the melody, so a sustained lead still keeps its continuity on the channels it can use.
    """
    active_by_frame = _bucket_by_frame(
        _spans_from_notes(notes, frame_rate_hz, n_frames), n_frames
    )

    assignment: list[list[Voice | None]] = [
        [None, None, None] for _ in range(n_frames)
    ]
    prev_ids: list[int | None] = [None, None, None]

    for f in range(n_frames):
        blocked = reserved[f] if reserved is not None else None
        usable = [ch for ch in range(N_CHANNELS) if ch != blocked]
        active = active_by_frame[f]
        by_id = {s.note_id: s for s in active}
        current: list[Voice | None] = [None, None, None]
        taken: set[int] = set()

        # 1) Continuity: keep last frame's note on its (still-usable) channel if it sounds on.
        for ch in usable:
            pid = prev_ids[ch]
            if pid is not None and pid in by_id:
                s = by_id[pid]
                current[ch] = Voice(s.pitch_hz, s.velocity, s.note_id)
                taken.add(pid)

        # 2) Fill free usable channels with the highest-priority unplaced notes.
        free = [ch for ch in usable if current[ch] is None]
        remaining = sorted(
            (s for s in active if s.note_id not in taken), key=_priority
        )
        for ch, s in zip(free, remaining):
            current[ch] = Voice(s.pitch_hz, s.velocity, s.note_id)

        assignment[f] = current
        prev_ids = [v.note_id if v is not None else None for v in current]

    return assignment


def n_frames_for(notes: list[Note], frame_rate_hz: int, duration_s: float) -> int:
    """Frames needed to cover the notes (or *duration_s*, whichever is longer)."""
    span_end = max((n.offset_s for n in notes), default=0.0)
    return frames_for_duration(max(duration_s, span_end), frame_rate_hz)
