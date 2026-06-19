"""Write a :class:`YmSong` as an uncompressed YM5/YM6 file (interleaved, clean high bits)."""

from __future__ import annotations

import struct

import numpy as np

from .model import YmSong

_MAGIC = {"YM6": b"YM6!", "YM5": b"YM5!"}


def to_bytes(song: YmSong, version: str = "YM6") -> bytes:
    if version not in _MAGIC:
        raise ValueError(f"Unsupported write version: {version!r} (use 'YM6' or 'YM5')")

    frames = np.ascontiguousarray(song.frames, dtype=np.uint8)
    n = frames.shape[0]
    if frames.shape[1] < 16:
        pad = np.zeros((n, 16 - frames.shape[1]), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=1)

    out = bytearray()
    out += _MAGIC[version]
    out += b"LeOnArD!"
    out += struct.pack(">I", n)
    out += struct.pack(">I", 1)  # songAttributes: bit0 = interleaved
    out += struct.pack(">H", 0)  # nbDigidrums
    out += struct.pack(">I", int(song.master_clock))
    out += struct.pack(">H", int(song.frame_rate))
    out += struct.pack(">I", int(song.loop_frame))
    out += struct.pack(">H", 0)  # addSize
    out += song.name.encode("latin-1", "replace") + b"\x00"
    out += song.author.encode("latin-1", "replace") + b"\x00"
    out += song.comment.encode("latin-1", "replace") + b"\x00"
    # Interleaved register block: all frames of R0, then R1, ... R15.
    out += np.ascontiguousarray(frames.T).tobytes()
    out += b"End!"
    return bytes(out)


def write(song: YmSong, path: str, version: str = "YM6") -> None:
    with open(path, "wb") as fh:
        fh.write(to_bytes(song, version))
