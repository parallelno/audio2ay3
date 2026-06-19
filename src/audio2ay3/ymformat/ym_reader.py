"""Read YM register-dump files (YM2/YM3/YM3b/YM5/YM6) into a :class:`YmSong`.

LHA-packed inputs (the common on-disk form, ``-lh5-``) are depacked transparently; uncompressed
YM files are read directly.
"""

from __future__ import annotations

import struct

import numpy as np

from ._lha import depack_lha, is_lha
from .model import YmSong


def _read_cstr(data: bytes, pos: int) -> tuple[str, int]:
    end = data.index(b"\x00", pos)
    return data[pos:end].decode("latin-1", "replace"), end + 1


def from_bytes(data: bytes) -> YmSong:
    """Parse YM bytes into a :class:`YmSong`, transparently depacking LHA-wrapped files."""
    if is_lha(data):
        data = depack_lha(data)
    magic = data[:4]
    if magic in (b"YM6!", b"YM5!"):
        return _parse_ym56(data, magic.decode("ascii").rstrip("!"))
    if magic == b"YM3!":
        return _parse_ym3(data, loop=False, version="YM3")
    if magic == b"YM3b":
        return _parse_ym3(data, loop=True, version="YM3b")
    if magic == b"YM2!":
        return _parse_ym3(data, loop=False, version="YM2")
    raise ValueError(f"Unrecognised YM magic: {magic!r}")


def _parse_ym56(data: bytes, version: str) -> YmSong:
    n = struct.unpack_from(">I", data, 12)[0]
    attrs = struct.unpack_from(">I", data, 16)[0]
    interleaved = bool(attrs & 1)
    nb_dd = struct.unpack_from(">H", data, 20)[0]
    clock = struct.unpack_from(">I", data, 22)[0]
    rate = struct.unpack_from(">H", data, 26)[0]
    loop = struct.unpack_from(">I", data, 28)[0]
    add_size = struct.unpack_from(">H", data, 32)[0]

    pos = 34 + add_size
    for _ in range(nb_dd):  # skip embedded digidrum samples (unsupported on write)
        sz = struct.unpack_from(">I", data, pos)[0]
        pos += 4 + sz

    name, pos = _read_cstr(data, pos)
    author, pos = _read_cstr(data, pos)
    comment, pos = _read_cstr(data, pos)

    block = np.frombuffer(data, dtype=np.uint8, count=16 * n, offset=pos)
    if interleaved:
        frames = block.reshape(16, n).T.copy()
    else:
        frames = block.reshape(n, 16).copy()

    return YmSong(
        frames=frames, master_clock=clock or 1_773_400, frame_rate=rate or 50,
        loop_frame=loop, version=version, name=name, author=author, comment=comment,
    )


def _parse_ym3(data: bytes, loop: bool, version: str) -> YmSong:
    body = data[4:]
    loop_frame = 0
    if loop and len(body) >= 4:
        loop_frame = struct.unpack_from(">I", body, len(body) - 4)[0]
        body = body[:-4]
    n = len(body) // 14
    block = np.frombuffer(body, dtype=np.uint8, count=14 * n)
    regs14 = block.reshape(14, n).T  # YM3 data is interleaved
    frames = np.zeros((n, 16), dtype=np.uint8)
    frames[:, :14] = regs14
    return YmSong(
        frames=frames, master_clock=2_000_000, frame_rate=50,
        loop_frame=loop_frame, version=version,
    )


def load(path: str) -> YmSong:
    with open(path, "rb") as fh:
        return from_bytes(fh.read())
