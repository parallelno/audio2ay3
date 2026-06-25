"""Load an audio file to a mono float32 array (optional ``soundfile`` dependency).

Audio decoding is deliberately isolated here so the deterministic arrange/encode core never
imports a heavyweight audio stack. ``soundfile`` (libsndfile) handles WAV/FLAC/OGG; MP3 support
depends on the local libsndfile build. Formats libsndfile cannot decode (notably ``.m4a``/AAC)
fall back to an ``ffmpeg`` subprocess.
"""

from __future__ import annotations

import os

import numpy as np

# Extensions soundfile/libsndfile cannot decode; routed straight to the ffmpeg fallback.
_FFMPEG_ONLY_EXTS = (".m4a",)


def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Return ``(mono_float32, sample_rate)`` for *path*.

    If *target_sr* is given and differs from the file rate, the signal is linearly resampled —
    adequate for analysis front-ends that resample internally anyway.

    ``.m4a``/AAC inputs (which libsndfile cannot decode) are read via an ``ffmpeg`` subprocess.
    """
    if os.path.splitext(path)[1].lower() in _FFMPEG_ONLY_EXTS:
        mono, sr = _load_audio_ffmpeg(path, target_sr)
    else:
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


def _load_audio_ffmpeg(path: str, target_sr: int | None) -> tuple[np.ndarray, int]:
    """Decode *path* to mono float32 via an ``ffmpeg`` subprocess.

    ffmpeg resamples directly when *target_sr* is given, so the linear fallback in the caller is
    a no-op for these inputs.
    """
    import shutil
    import subprocess

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Decoding {os.path.splitext(path)[1]} files needs 'ffmpeg' on PATH "
            "(install it, e.g. `winget install ffmpeg` / `brew install ffmpeg`)."
        )

    sr = target_sr or 44100
    cmd = [
        ffmpeg, "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path!r}: {msg}")

    mono = np.frombuffer(proc.stdout, dtype=np.float32)
    return mono, sr


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = np.linspace(0.0, x.size - 1, num=n_out, dtype=np.float64)
    idx = np.arange(x.size, dtype=np.float64)
    return np.interp(src, idx, x).astype(np.float32)
