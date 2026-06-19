"""Load an audio file to a mono float32 array (optional ``soundfile`` dependency).

Audio decoding is deliberately isolated here so the deterministic arrange/encode core never
imports a heavyweight audio stack. ``soundfile`` (libsndfile) handles WAV/FLAC/OGG; MP3 support
depends on the local libsndfile build.
"""

from __future__ import annotations

import numpy as np


def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return ``(mono_float32, sample_rate)`` for *path*.

    If *target_sr* is given and differs from the file rate, the signal is linearly resampled —
    adequate for analysis front-ends that resample internally anyway.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Reading audio needs the 'audio' extra: pip install audio2ay3[audio]"
        ) from exc

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix to mono

    if target_sr and target_sr != sr:
        mono = _resample_linear(mono, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(mono, dtype=np.float32), sr


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = np.linspace(0.0, x.size - 1, num=n_out, dtype=np.float64)
    idx = np.arange(x.size, dtype=np.float64)
    return np.interp(src, idx, x).astype(np.float32)
