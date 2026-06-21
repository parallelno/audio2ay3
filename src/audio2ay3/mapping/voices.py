"""Allocate transcribed notes to the three AY tone channels, frame by frame.

The skeleton policy is a greedy allocator with **continuity**: a note already sounding on a
channel stays there while it lasts, so we avoid the frame-to-frame channel hopping that causes
audible warble. Free channels are filled by priority — instrument identity first (a lead beats a
pad even when quieter, see :func:`_program_rank`), then loudness, then a mild bass bias — and
when more notes sound at once than there are channels the lowest-priority are dropped that frame.

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
    # Source-derived loudness for this exact frame (0..1, relative to the note's peak), or
    # ``None`` when the note carries no contour and the arranger should use its synthetic
    # envelope instead.
    amp_scale: float | None = None
    # General-MIDI program of the source note (``None`` for Basic Pitch / synthetic), so the
    # arranger can hold a sustained instrument legato instead of imposing a struck decay.
    program: int | None = None


@dataclass
class _Span:
    note_id: int
    start: int
    end: int  # exclusive
    pitch_hz: float
    velocity: float
    contour: tuple[float, ...] = ()
    program: int | None = None


def _contour_scale(span: _Span, frame: int) -> float | None:
    """The note's loudness at *frame* (clamped to the contour's ends), or ``None`` if absent."""
    if not span.contour:
        return None
    idx = frame - span.start
    if idx < 0:
        idx = 0
    elif idx >= len(span.contour):
        idx = len(span.contour) - 1
    return span.contour[idx]


def _priority(span: _Span) -> tuple[int, float, float]:
    # Instrument identity first: a quiet lead must beat a loud pad when channels are scarce
    # (this is what keeps the main theme audible). Within a salience class, louder wins, then
    # ties break toward lower pitch (a mild bass bias keeps the foundation).
    return (_program_rank(span.program), -span.velocity, span.pitch_hz)


# General-MIDI families split into salience classes. Lower rank = more important to keep.
# Only the unambiguous families are moved off neutral: the clearest melodic/lead voices are
# promoted and the clearest sustained-backing/atmosphere voices are demoted, so loudness still
# decides among the genuinely ambiguous instruments (piano, organ, guitar, solo strings, brass).
# ``None`` (Basic Pitch, synthetic notes) stays neutral, preserving the loudness-only behaviour.
_LEAD_PROGRAMS = frozenset(
    set(range(8, 16))  # Chromatic Percussion (glockenspiel, vibraphone, marimba, music box...)
    | set(range(64, 72))  # Reed (sax, oboe, clarinet, bassoon...)
    | set(range(72, 80))  # Pipe (flute, piccolo, recorder, whistle, ocarina...)
    | set(range(80, 88))  # Synth Lead (square, sawtooth, calliope, charang, lead...)
)
_PAD_PROGRAMS = frozenset(
    set(range(48, 56))  # Ensemble (string/synth-string ensembles, choir, voice, orchestra hit)
    | set(range(88, 96))  # Synth Pad (new age, warm, polysynth, halo, sweep...)
    | set(range(96, 104))  # Synth Effects (atmosphere, brightness, soundtrack, sci-fi...)
    | set(range(120, 128))  # Sound Effects (breath, seashore, applause, gunshot...)
)


def _program_rank(program: int | None) -> int:
    """Salience class for a GM *program*: 0 = lead/foreground, 1 = neutral, 2 = pad/background."""
    if program is None:
        return 1
    if program in _LEAD_PROGRAMS:
        return 0
    if program in _PAD_PROGRAMS:
        return 2
    return 1


# GM families that ring on naturally (bowed/blown/electronically held), so a long note should
# stay at its source level instead of decaying like a plucked string: Organ (16-23) plus the
# continuous span Strings/Ensemble/Brass/Reed/Pipe/Synth-Lead/Synth-Pad/Synth-FX (40-103).
# Everything else (Piano, Chromatic Percussion, Guitar, Bass, Ethnic, Percussive) and ``None``
# (Basic Pitch) keeps the struck attack-and-decay the arranger applied before.
_SUSTAINED_PROGRAMS = frozenset(set(range(16, 24)) | set(range(40, 104)))


def is_sustained_program(program: int | None) -> bool:
    """Whether a GM *program* is a held/legato instrument (no natural per-note decay)."""
    return program is not None and program in _SUSTAINED_PROGRAMS


# Instruments that idiomatically vibrato — Organ (16-23), Strings (40-47), Reed/sax (64-71),
# Pipe/flute (72-79), Synth Lead (80-87). A few cents of pitch LFO makes a bare square read as a
# living tone; brass/pads/ensembles are intentionally left steady.
_VIBRATO_PROGRAMS = frozenset(
    set(range(16, 24)) | set(range(40, 48)) | set(range(64, 88))
)
# Breathy wind instruments — Reed (64-71) + Pipe/flute (72-79) — get a short noise "chiff" at the
# attack to imitate their air.
_BREATH_PROGRAMS = frozenset(range(64, 80))


def is_vibrato_program(program: int | None) -> bool:
    """Whether a GM *program* should get an idiomatic pitch vibrato."""
    return program is not None and program in _VIBRATO_PROGRAMS


def is_breath_program(program: int | None) -> bool:
    """Whether a GM *program* should get a breathy noise chiff at each note's attack."""
    return program is not None and program in _BREATH_PROGRAMS


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
            spans.append(
                _Span(
                    note_id,
                    start,
                    end,
                    note.pitch_hz,
                    note.velocity,
                    note.amp_contour,
                    note.program,
                )
            )
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
        # Negative note-id namespace keeps bass notes distinct from melodic notes, so a bass
        # note and a melodic note that share an index can never be mistaken for one held note
        # when channel A flips between them (the arranger keys its envelope off note identity).
        bass_voices[f] = Voice(
            s.pitch_hz, s.velocity, -(s.note_id + 1), _contour_scale(s, f), s.program
        )
        reserved[f] = channel
    return bass_voices, reserved


def allocate_voices(
    notes: list[Note],
    frame_rate_hz: int,
    n_frames: int,
    reserved: list[int | None] | None = None,
    *,
    n_channels: int = N_CHANNELS,
    arpeggiate: bool = False,
) -> list[list[Voice | None]]:
    """Return ``assignment[frame][channel]`` of :class:`Voice` or ``None`` (silent).

    *n_channels* is the number of tone channels to fill (3 for a single AY, 6 for dual-AY); the
    allocator's continuity/priority policy is identical regardless of width, it simply has more
    channels to spread the melody across.

    When *reserved* is given, ``reserved[f]`` names a channel that is off-limits to melodic
    notes in frame *f* (because :func:`place_bass` owns it that frame); the remaining channels
    absorb the melody, so a sustained lead still keeps its continuity on the channels it can use.

    When *arpeggiate* is set, notes that would otherwise be dropped because every channel is busy
    are folded into a fast cycle on the lowest-priority channel (the classic chiptune arpeggio),
    so squeezed chord tones are heard in turn rather than silenced.
    """
    active_by_frame = _bucket_by_frame(
        _spans_from_notes(notes, frame_rate_hz, n_frames), n_frames
    )

    assignment: list[list[Voice | None]] = [
        [None] * n_channels for _ in range(n_frames)
    ]
    prev_ids: list[int | None] = [None] * n_channels

    for f in range(n_frames):
        blocked = reserved[f] if reserved is not None else None
        usable = [ch for ch in range(n_channels) if ch != blocked]
        active = active_by_frame[f]
        by_id = {s.note_id: s for s in active}
        current: list[Voice | None] = [None] * n_channels
        taken: set[int] = set()

        # 1) Continuity: keep last frame's note on its (still-usable) channel if it sounds on.
        for ch in usable:
            pid = prev_ids[ch]
            if pid is not None and pid in by_id:
                s = by_id[pid]
                current[ch] = Voice(
                    s.pitch_hz, s.velocity, s.note_id, _contour_scale(s, f), s.program
                )
                taken.add(pid)

        # 2) Fill free usable channels with the highest-priority unplaced notes.
        free = [ch for ch in usable if current[ch] is None]
        remaining = sorted(
            (s for s in active if s.note_id not in taken), key=_priority
        )
        for ch, s in zip(free, remaining):
            current[ch] = Voice(
                s.pitch_hz, s.velocity, s.note_id, _contour_scale(s, f), s.program
            )
            taken.add(s.note_id)

        # 3) Arpeggio: any active note still unplaced would be dropped for lack of a channel.
        #    Instead, fold it (and its host channel's note) into a one-per-frame cycle on the
        #    lowest-priority usable channel, so every squeezed chord tone is heard in turn.
        if arpeggiate:
            overflow = [s for s in active if s.note_id not in taken]
            occupied = [ch for ch in usable if current[ch] is not None]
            if overflow and occupied:
                # Lowest priority = the largest _priority tuple (priority sorts ascending).
                arp_ch = max(
                    occupied, key=lambda ch: _priority(by_id[current[ch].note_id])
                )
                group = sorted(
                    [by_id[current[arp_ch].note_id], *overflow], key=lambda sp: sp.pitch_hz
                )
                s = group[f % len(group)]
                current[arp_ch] = Voice(
                    s.pitch_hz, s.velocity, s.note_id, _contour_scale(s, f), s.program
                )

        assignment[f] = current
        prev_ids = [v.note_id if v is not None else None for v in current]

    return assignment


def n_frames_for(notes: list[Note], frame_rate_hz: int, duration_s: float) -> int:
    """Frames needed to cover the notes (or *duration_s*, whichever is longer)."""
    span_end = max((n.offset_s for n in notes), default=0.0)
    return frames_for_duration(max(duration_s, span_end), frame_rate_hz)
