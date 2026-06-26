"""Export a :class:`Transcription` to a Standard MIDI File for quality review.

This writes the **pre-arrangement** transcription — everything the neural front-end detected,
*before* :func:`pipeline.arrange` squeezes it onto the AY's three tone channels. It exists so a
conversion's musical content (pitches, timing, per-note loudness, instrument labels) can be
auditioned in any DAW/player and compared against the source, without first being flattened into
the lossy register stream.

The mapping:

* ``pitch_hz``      -> nearest MIDI note number (``69 + 12*log2(f/440)``).
* ``velocity`` 0..1 -> note-on velocity 1..127 (the note's peak loudness).
* ``amp_contour``   -> per-frame **CC11 (Expression)** automation across the note, so the
  loudness *shape* (a held swell vs a plucked decay) is preserved, not just the strike level.
* ``program``       -> a ``program_change`` (when the backend knows the GM instrument; Basic
  Pitch leaves it ``None`` and the channel keeps the default program).
* ``stem``          -> a separate track/channel: Melody, Bass, Vocals. One voice per channel so
  each note's CC11 contour animates independently (CC is per-channel in MIDI).
* ``Percussion``    -> GM channel 10 drums (kick 36 / snare 38 / hat 42).

Uses ``mido`` (pure-Python); install with ``pip install "audio2ay3[midi]"``.
"""

from __future__ import annotations

import math
from pathlib import Path

from .model import Note, Percussion, Transcription

# MIDI channels (0-indexed). Channel 9 is GM percussion.
_CH_MELODY = 0
_CH_BASS = 1
_CH_VOCALS = 2
_CH_DRUMS = 9

# General MIDI percussion note numbers for our three coarse buckets.
_DRUM_NOTE = {"kick": 36, "snare": 38, "hat": 42}
# A drum hit has no duration in the transcription; give it a short fixed-length note so players
# render an audible tick.
_DRUM_LEN_S = 0.05

# Event ordering at an identical tick: release before (re-)programming, program before the new
# note, and expression automation after the note has started.
_ORD_NOTE_OFF = 0
_ORD_PROGRAM = 1
_ORD_NOTE_ON = 2
_ORD_CC = 3


def _require_mido():
    try:
        import mido  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via CLI message
        raise RuntimeError(
            "MIDI export needs the 'mido' package. Install it with "
            'pip install "audio2ay3[midi]".'
        ) from exc
    return mido


def _hz_to_midi(hz: float) -> int:
    """Nearest MIDI note number for *hz*, clamped to the valid 0..127 range."""
    if hz <= 0.0:
        return 0
    return max(0, min(127, int(round(69.0 + 12.0 * math.log2(hz / 440.0)))))


def _vel_to_midi(velocity: float) -> int:
    """Map a 0..1 perceptual loudness to a 1..127 MIDI velocity (never a silent 0)."""
    return max(1, min(127, int(round(velocity * 127.0))))


def _sec_to_ticks(seconds: float, ticks_per_second: float) -> int:
    return max(0, int(round(seconds * ticks_per_second)))


def _note_events(
    note: Note,
    channel: int,
    ticks_per_second: float,
    frame_rate_hz: int,
    mido,
) -> list[tuple[int, int, object]]:
    """Absolute-tick (tick, order, Message) events for one pitched *note*."""
    events: list[tuple[int, int, object]] = []
    pitch = _hz_to_midi(note.pitch_hz)
    on_tick = _sec_to_ticks(note.onset_s, ticks_per_second)
    off_tick = max(on_tick + 1, _sec_to_ticks(note.offset_s, ticks_per_second))

    if note.program is not None:
        events.append((
            on_tick, _ORD_PROGRAM,
            mido.Message("program_change", channel=channel,
                         program=max(0, min(127, note.program))),
        ))
    events.append((
        on_tick, _ORD_NOTE_ON,
        mido.Message("note_on", channel=channel, note=pitch,
                     velocity=_vel_to_midi(note.velocity)),
    ))
    # Per-frame loudness shape -> CC11 automation, skipping repeats to keep the file small.
    if note.amp_contour and frame_rate_hz > 0:
        last_value = -1
        for k, amp in enumerate(note.amp_contour):
            value = max(1, min(127, int(round(amp * 127.0))))
            if value == last_value:
                continue
            last_value = value
            t = _sec_to_ticks(note.onset_s + k / frame_rate_hz, ticks_per_second)
            t = max(on_tick, min(t, off_tick - 1))
            events.append((
                t, _ORD_CC,
                mido.Message("control_change", channel=channel, control=11, value=value),
            ))
    events.append((
        off_tick, _ORD_NOTE_OFF,
        mido.Message("note_off", channel=channel, note=pitch, velocity=0),
    ))
    return events


def _percussion_events(
    hit: Percussion,
    ticks_per_second: float,
    mido,
) -> list[tuple[int, int, object]]:
    note = _DRUM_NOTE.get(hit.kind, _DRUM_NOTE["snare"])
    on_tick = _sec_to_ticks(hit.onset_s, ticks_per_second)
    off_tick = max(on_tick + 1, _sec_to_ticks(hit.onset_s + _DRUM_LEN_S, ticks_per_second))
    return [
        (on_tick, _ORD_NOTE_ON,
         mido.Message("note_on", channel=_CH_DRUMS, note=note,
                      velocity=_vel_to_midi(hit.velocity))),
        (off_tick, _ORD_NOTE_OFF,
         mido.Message("note_off", channel=_CH_DRUMS, note=note, velocity=0)),
    ]


def _build_track(name: str, events: list[tuple[int, int, object]], mido):
    """Sort absolute-tick *events* and emit a delta-timed :class:`mido.MidiTrack`."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    events.sort(key=lambda e: (e[0], e[1]))
    prev_tick = 0
    for abs_tick, _order, msg in events:
        track.append(msg.copy(time=abs_tick - prev_tick))
        prev_tick = abs_tick
    return track


def write_transcription_midi(
    tr: Transcription,
    path: str | Path,
    frame_rate_hz: int,
    *,
    tempo_bpm: float = 120.0,
    ppq: int = 480,
) -> Path:
    """Write *tr* to a Standard MIDI File at *path* and return the path.

    *frame_rate_hz* is the analysis frame rate (``cfg.chip.frame_rate_hz``); it maps each
    ``amp_contour`` frame to a wall-clock time for the CC11 automation. *tempo_bpm* and *ppq*
    only set the file's tick grid — note times stay faithful in seconds regardless.
    """
    mido = _require_mido()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ticks_per_second = ppq * tempo_bpm / 60.0
    mid = mido.MidiFile(ticks_per_beat=ppq)

    # Split the melodic notes by source stem so each lands on its own channel/track.
    melody = [n for n in tr.notes if n.stem != "vocals"]
    vocals = [n for n in tr.notes if n.stem == "vocals"]

    tracks: list[tuple[str, list[Note], int]] = [
        ("Melody", melody, _CH_MELODY),
        ("Bass", tr.bass_notes, _CH_BASS),
        ("Vocals", vocals, _CH_VOCALS),
    ]

    first = True
    for name, notes, channel in tracks:
        if not notes:
            continue
        events: list[tuple[int, int, object]] = []
        for note in notes:
            events.extend(_note_events(note, channel, ticks_per_second, frame_rate_hz, mido))
        track = _build_track(name, events, mido)
        if first:
            track.insert(1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
            first = False
        mid.tracks.append(track)

    if tr.percussion:
        events = []
        for hit in tr.percussion:
            events.extend(_percussion_events(hit, ticks_per_second, mido))
        track = _build_track("Drums", events, mido)
        if first:
            track.insert(1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
            first = False
        mid.tracks.append(track)

    if not mid.tracks:
        # Nothing transcribed: still emit a valid, empty MIDI file with a tempo.
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
        mid.tracks.append(track)

    mid.save(str(out))
    return out
