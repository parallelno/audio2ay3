"""Detect drum hits from an isolated drum stem and bucket them for the AY noise channel.

This is the percussion counterpart to :mod:`.transcribe`. The neural separation upstream has
already done the hard part — isolating drums from everything else — so finding *when* hits occur
is lightweight signal processing: ``librosa`` spectral-flux onset detection on the clean stem.
Each onset is then classified into kick / snare / hat by spectral centroid (a robust low / mid /
high brightness proxy), which is exactly the bucket :func:`audio2ay3.mapping.apply_percussion`
needs to pick a noise period and decay envelope.

``librosa`` is imported lazily, matching the rest of the analysis stage: the package stays
importable (and the emulator/``validate`` path usable) without the ``[neural]`` extra installed.
"""

from __future__ import annotations

import numpy as np

from .model import Percussion, PercussionKind

# A hit is a *kick* when a large fraction of its energy sits in the sub-bass band. Centroid
# alone can't find kicks: in a real drum stem hi-hat/cymbal bleed pulls every onset's centroid
# up, so we test low-band energy share first, then split the rest snare/hat by brightness.
_KICK_BAND_HZ = 150.0  # energy below this counts toward the "kick" band
_KICK_LOW_RATIO = 0.25  # >= this share of energy below _KICK_BAND_HZ -> kick (bass-dominant)
_SNARE_MAX_HZ = 3000.0  # centroid below -> snare, above -> hi-hat/cymbal

# STFT hop for onset strength / spectral features. ~23 ms at 22.05 kHz — fine for 50 Hz frames.
_HOP = 512

# Window (seconds) after each onset over which spectral features are averaged for classification.
_CLASSIFY_WIN_S = 0.03

# Floor so a detected-but-soft hit still triggers an audible noise burst.
_MIN_VELOCITY = 0.3


def _classify(centroid_hz: float, low_ratio: float) -> PercussionKind:
    if low_ratio >= _KICK_LOW_RATIO:
        return "kick"
    if centroid_hz < _SNARE_MAX_HZ:
        return "snare"
    return "hat"


def detect_percussion(drums: np.ndarray, sr: int) -> list[Percussion]:
    """Return drum hits found in the mono *drums* stem, classified kick / snare / hat.

    Returns an empty list for empty/silent input or when no onsets are found.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Percussion detection needs the 'neural' extra (librosa): "
            "pip install audio2ay3[neural]"
        ) from exc

    y = np.ascontiguousarray(drums, dtype=np.float32)
    if y.size == 0 or not np.any(y):
        return []

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP, backtrack=True
    )
    if len(onset_frames) == 0:
        return []
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=_HOP)

    # One magnitude STFT feeds both the brightness (centroid) and low-band-energy features.
    spec = np.abs(librosa.stft(y, hop_length=_HOP))
    freqs = librosa.fft_frequencies(sr=sr)
    total = spec.sum(axis=0) + 1e-9
    centroid = (freqs[:, None] * spec).sum(axis=0) / total
    low_ratio = spec[freqs < _KICK_BAND_HZ].sum(axis=0) / total

    # Velocity from onset strength, normalised by a robust reference (90th percentile of the
    # hit strengths) so a single loud transient doesn't squash everything to the floor.
    strengths = onset_env[np.minimum(onset_frames, len(onset_env) - 1)]
    ref = float(np.percentile(strengths, 90)) or 1.0
    win = max(1, round(_CLASSIFY_WIN_S * sr / _HOP))
    n_cols = spec.shape[1]

    hits: list[Percussion] = []
    for frame, t, strength in zip(onset_frames, onset_times, strengths):
        lo = min(int(frame), n_cols - 1)
        hi = min(n_cols, lo + win)
        centroid_hz = float(np.mean(centroid[lo:hi]))
        lr = float(np.mean(low_ratio[lo:hi]))
        velocity = max(_MIN_VELOCITY, min(1.0, float(strength) / ref))
        hits.append(
            Percussion(onset_s=float(t), kind=_classify(centroid_hz, lr), velocity=velocity)
        )
    return hits
