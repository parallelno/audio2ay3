"""Audio file output: WAV (stdlib) and optional MP3 (lameenc).

Only peak-safe gain / format conversion happens here — never tonal shaping (the project's
"no post-AY enhancement" rule). DC removal and normalisation live in the renderer.
"""

from __future__ import annotations

import wave

import numpy as np


def _to_int16_bytes(pcm: np.ndarray) -> bytes:
    clipped = np.clip(pcm, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def write_wav(path: str, pcm: np.ndarray, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(_to_int16_bytes(pcm))


def write_mp3(path: str, pcm: np.ndarray, sr: int, bitrate_kbps: int = 192) -> None:
    try:
        import lameenc
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "MP3 output requires the 'lameenc' package "
            "(pip install \"audio2ay3[mp3]\"). Use a .wav output path instead."
        ) from exc

    enc = lameenc.Encoder()
    enc.set_bit_rate(int(bitrate_kbps))
    enc.set_in_sample_rate(int(sr))
    enc.set_channels(1)
    enc.set_quality(2)
    data = enc.encode(_to_int16_bytes(pcm))
    data += enc.flush()
    with open(path, "wb") as fh:
        fh.write(data)


def write_audio(path: str, pcm: np.ndarray, sr: int, bitrate_kbps: int = 192) -> None:
    if str(path).lower().endswith(".mp3"):
        write_mp3(path, pcm, sr, bitrate_kbps)
    else:
        write_wav(path, pcm, sr)
