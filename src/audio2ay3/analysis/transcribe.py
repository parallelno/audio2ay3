"""Transcription: audio -> :class:`Transcription` via a neural model (Basic Pitch default).

Backends are imported lazily and reduce to the same neutral IR. Basic Pitch is the lightweight
default; MT3 and Onsets-and-Frames are reserved for the deeper-analysis phase. Percussion is not
produced by Basic Pitch, so for now drum events come through empty here and are added by a
dedicated onset stage later.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .model import Note, Transcription

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
    if mode in ("mt3", "onsets-frames"):
        raise NotImplementedError(
            f"The {mode!r} backend is planned for the deeper-analysis phase; "
            "use --transcription basic-pitch for now."
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
