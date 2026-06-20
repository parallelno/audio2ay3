"""Transcription: audio -> :class:`Transcription` via a neural model (Basic Pitch default).

Backends are imported lazily and reduce to the same neutral IR:

- ``basic-pitch`` (Spotify) — the lightweight default. Polyphonic pitched notes only; percussion
  is filled by a dedicated onset stage and bass by a second pass over the bass stem.
- ``mt3`` (Google Magenta) — heavy multitrack model. A single pass yields pitched notes, bass,
  and drums together with General-MIDI instrument identity, so it routes its own percussion and
  bass here (see :func:`note_sequence_to_transcription`). Reserved behind the ``[mt3]`` extra.
- ``onsets-frames`` (Google Magenta) — still reserved for the deeper-analysis phase.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .model import Note, Percussion, PercussionKind, Transcription

# Basic Pitch's defaults target general transcription, not a 50 Hz chip. Its 127.7 ms
# minimum-note-length silently drops fast passages (a 16th note at 160 BPM is ~94 ms), which is
# why fast piano runs went missing; we lower it to ~3 AY frames so short notes survive (the chip
# can't resolve anything briefer anyway). A slightly lower onset threshold also re-splits the
# fast repeated notes the default merges into one sustained tone.
_BP_ONSET_THRESHOLD = 0.45
_BP_MIN_NOTE_MS = 58.0


def transcribe(
    audio: np.ndarray, sr: int, mode: str = "basic-pitch", frame_rate_hz: int = 50
) -> Transcription:
    """Transcribe a mono signal into notes using the selected neural backend."""
    if mode == "basic-pitch":
        return _transcribe_basic_pitch(audio, sr)
    if mode == "mt3":
        return _transcribe_mt3(audio, sr, frame_rate_hz)
    if mode == "onsets-frames":
        raise NotImplementedError(
            "The 'onsets-frames' backend is planned for the deeper-analysis phase; "
            "use --transcription basic-pitch or mt3 for now."
        )
    raise ValueError(f"unknown transcription mode: {mode!r}")


def _basic_pitch_model_path():
    """Pick the most portable Basic Pitch model available.

    Basic Pitch can run several backends; on import it defaults to the heavy TensorFlow
    SavedModel whenever TensorFlow is installed. We prefer the ONNX model when onnxruntime is
    present — it is lighter, has no native TF dependency, and avoids TF's CPU/SIMD fragility on
    some Windows machines. Falls back to whatever Basic Pitch chose by default otherwise.
    """
    import basic_pitch

    if getattr(basic_pitch, "ONNX_PRESENT", False):
        return basic_pitch.build_icassp_2022_model_path(basic_pitch.FilenameSuffix.onnx)
    return basic_pitch.ICASSP_2022_MODEL_PATH


def _transcribe_basic_pitch(audio: np.ndarray, sr: int) -> Transcription:
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Basic Pitch transcription needs the 'neural' extra: "
            "pip install audio2ay3[neural]"
        ) from exc

    from ..render.audio_out import write_wav

    model_path = _basic_pitch_model_path()
    duration_s = float(audio.size) / sr if sr else 0.0
    # Basic Pitch reads a file path; hand it a temporary WAV of the (separated) signal.
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "in.wav")
        write_wav(wav_path, audio, sr)
        _, _, note_events = predict(
            wav_path,
            model_path,
            onset_threshold=_BP_ONSET_THRESHOLD,
            minimum_note_length=_BP_MIN_NOTE_MS,
        )

    notes: list[Note] = []
    for event in note_events:
        start_s, end_s, pitch_midi, amplitude = event[0], event[1], event[2], event[3]
        notes.append(
            Note(
                onset_s=float(start_s),
                duration_s=max(0.0, float(end_s) - float(start_s)),
                pitch_hz=_midi_to_hz(int(pitch_midi)),
                velocity=float(max(0.0, min(1.0, amplitude))),
            )
        )
    return Transcription(notes=notes, percussion=[], duration_s=duration_s)


def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# --- MT3 (Google Magenta) multitrack backend ------------------------------------------------
#
# MT3 emits a General-MIDI ``NoteSequence`` covering every instrument at once. We route each note
# by its identity so the arranger gets the richer hand-off it can't reconstruct from Basic Pitch:
# drums become percussion hits, the GM bass family feeds the dedicated bass channel, and the rest
# is shared melodic/harmonic content. The conversion is a pure function so it is unit-tested with
# a synthetic NoteSequence, no JAX/TensorFlow required.

# GM "Bass" program family (0-indexed programs 32-39: acoustic/electric/fretless/slap/synth bass).
_MT3_BASS_PROGRAMS = frozenset(range(32, 40))
# GM percussion (channel-10) key map, bucketed into the three AY-friendly drum voices.
_KICK_MIDI = frozenset({35, 36})
_SNARE_MIDI = frozenset({37, 38, 39, 40, 41, 43, 45, 47, 48, 50})  # snare/clap/stick + low toms


def _drum_kind_for_midi(pitch: int) -> PercussionKind:
    """Map a GM percussion key number to a kick/snare/hat bucket (hats/cymbals fall through)."""
    if pitch in _KICK_MIDI:
        return "kick"
    if pitch in _SNARE_MIDI:
        return "snare"
    return "hat"


def note_sequence_to_transcription(ns, frame_rate_hz: int = 50) -> Transcription:
    """Convert an MT3 (or Onsets-and-Frames) ``NoteSequence`` into the neutral IR.

    ``ns`` is duck-typed: it needs ``total_time`` and an iterable ``notes`` whose items expose
    ``pitch`` (MIDI), ``start_time``, ``end_time``, ``velocity`` (0-127), ``program`` (GM) and
    ``is_drum``. Drum notes become :class:`Percussion`; GM bass-family programs become
    ``bass_notes``; everything else is shared melodic content. ``frame_rate_hz`` is accepted for
    signature parity with the other backends.
    """
    _ = frame_rate_hz
    notes: list[Note] = []
    bass_notes: list[Note] = []
    percussion: list[Percussion] = []
    duration_s = float(getattr(ns, "total_time", 0.0) or 0.0)
    for n in getattr(ns, "notes", ()):
        start = float(n.start_time)
        end = float(n.end_time)
        velocity = max(0.0, min(1.0, float(getattr(n, "velocity", 100)) / 127.0)) or 1.0
        if getattr(n, "is_drum", False):
            percussion.append(
                Percussion(
                    onset_s=start, kind=_drum_kind_for_midi(int(n.pitch)), velocity=velocity
                )
            )
            duration_s = max(duration_s, start)
            continue
        note = Note(
            onset_s=start,
            duration_s=max(0.0, end - start),
            pitch_hz=_midi_to_hz(int(n.pitch)),
            velocity=velocity,
        )
        if int(getattr(n, "program", 0)) in _MT3_BASS_PROGRAMS:
            bass_notes.append(note)
        else:
            notes.append(note)
        duration_s = max(duration_s, end)
    return Transcription(
        notes=notes, percussion=percussion, bass_notes=bass_notes, duration_s=duration_s
    )


def _transcribe_mt3(audio: np.ndarray, sr: int, frame_rate_hz: int) -> Transcription:
    """Run MT3 multitrack transcription and route its NoteSequence into the IR.

    The heavy MT3/T5X/JAX stack and its inference glue live in :mod:`._mt3_infer`; it raises a
    clear ``RuntimeError`` when the ``[mt3]`` extra or model checkpoint is missing.
    """
    from . import _mt3_infer

    ns = _mt3_infer.transcribe_mt3(audio, sr)
    return note_sequence_to_transcription(ns, frame_rate_hz)
