"""Per-note loudness contours: follow each note's real amplitude shape from its source stem.

Basic Pitch reports one amplitude (peak loudness) per note, so a synthetic envelope is the only
shape the arranger would otherwise have — and a single fixed curve makes every note decay
identically, flattening their character. Here we sample the actual per-frame RMS of the
(separated) signal across each note's span and normalise it to the note's own peak. The arranger
multiplies the note's velocity-derived amplitude by this shape, so a sustained note stays up and
a pluck decays the way it does in the original.

This module is pure NumPy — the only neural dependency is the separated audio handed in, so the
maths is exercised directly by unit tests with synthetic signals.
"""

from __future__ import annotations

import numpy as np

from .model import Note


def frame_rms_envelope(audio: np.ndarray, sr: int, frame_rate_hz: int) -> np.ndarray:
    """Per-frame RMS of *audio* sampled at *frame_rate_hz* (one value per output frame).

    Each frame's value is the RMS over a roughly two-frame window centred on the frame, which
    smooths sample-level noise without smearing across note boundaries.
    """
    if sr <= 0 or frame_rate_hz <= 0 or audio.size == 0:
        return np.zeros(0, dtype=np.float64)
    hop = sr / frame_rate_hz
    n = int(audio.size)
    n_frames = int(np.ceil(n / hop))
    # Prefix sums of the squared signal turn every windowed mean-square into an O(1) lookup.
    sq = np.square(audio.astype(np.float64))
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    half = max(1, int(round(hop)))  # ~one frame either side
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        center = int(round(i * hop))
        a = max(0, center - half)
        b = min(n, center + half)
        rms[i] = np.sqrt((csum[b] - csum[a]) / max(1, b - a))
    return rms


def _contour_for_span(rms: np.ndarray, onset_f: int, offset_f: int) -> tuple[float, ...]:
    """Normalised loudness shape over ``[onset_f, offset_f]`` (1.0 at the note's loudest frame)."""
    if rms.size == 0:
        return ()
    lo = max(0, onset_f)
    hi = min(int(rms.size), max(onset_f + 1, offset_f + 1))
    if hi <= lo:
        return ()
    seg = rms[lo:hi]
    peak = float(seg.max())
    if peak <= 0.0:
        return ()
    return tuple(float(min(1.0, v / peak)) for v in seg)


def attach_amp_contours(
    notes: list[Note], audio: np.ndarray, sr: int, frame_rate_hz: int
) -> list[Note]:
    """Return *notes* with :attr:`Note.amp_contour` filled from *audio*'s loudness envelope.

    Notes are returned unchanged when *audio* is empty or a note's span is silent, leaving the
    arranger free to fall back to its synthetic envelope for those.
    """
    rms = frame_rms_envelope(audio, sr, frame_rate_hz)
    if rms.size == 0:
        return list(notes)
    out: list[Note] = []
    for note in notes:
        onset_f = int(round(note.onset_s * frame_rate_hz))
        offset_f = int(round(note.offset_s * frame_rate_hz))
        contour = _contour_for_span(rms, onset_f, offset_f)
        out.append(
            Note(
                onset_s=note.onset_s,
                duration_s=note.duration_s,
                pitch_hz=note.pitch_hz,
                velocity=note.velocity,
                amp_contour=contour,
                program=note.program,
            )
        )
    return out
