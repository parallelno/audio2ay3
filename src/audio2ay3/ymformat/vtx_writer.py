"""Write a YmSong as a VTX file (original ZLib variant, V_Soft / Vortex Project).

Header layout (15 bytes, all little-endian):

  Offset  Size  Field
   0       2    id        b"ay" (AY-3-8910/12) or b"ym" (YM2149)
   2       1    stereo    0=MONO 1=ABC 2=ACB 3=BAC 4=BCA 5=CAB 6=CBA
   3       1    loop      1=loop enabled, 0=no loop
   4       2    loopStart loop frame index (0 = beginning of melody)
   6       2    freq      playback frequency in Hz (50 for PAL)
   8       1    chipType  1=single AY, 2=turboAY (dual chip)
   9       2    year      year of composition (0 = unknown)
  11       1    playerClock  0=1 773 400 Hz ZX Spectrum, 1=2 000 000 Hz Atari ST
  12       3    reserved  must be 0x00 0x00 0x00

After header: 5 x null-terminated strings: title, author, from, tracker, comment
After strings: ZLib-compressed frame data

Frame data layout (column-major, same as YM3):
  chipType=1: R0[0..N-1], R1[0..N-1], ..., R13[0..N-1]           (14 x N bytes)
  chipType=2: chip-0 block (14 x N bytes) then chip-1 block (14 x N bytes)

Supported by ZXTune, Vortex Tracker II, and other modern ZX players.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from .model import YmSong

# ---- constants ---------------------------------------------------------------

_ID_AY: bytes = b"ay"   # AY-3-8910 / AY-3-8912
_ID_YM: bytes = b"ym"   # YM2149

STEREO_MONO = 0
STEREO_ABC  = 1
STEREO_ACB  = 2
STEREO_BAC  = 3
STEREO_BCA  = 4
STEREO_CAB  = 5
STEREO_CBA  = 6

_CHIP_SINGLE = 1
_CHIP_TURBO  = 2   # turboAY / dual chip

_CLOCK_ZX    = 0   # 1 773 400 Hz (ZX Spectrum)
_CLOCK_ATARI = 1   # 2 000 000 Hz (Atari ST)

_CLOCK_MAP: dict[int, int] = {
    1_773_400: _CLOCK_ZX,
    1_750_000: _CLOCK_ZX,
    2_000_000: _CLOCK_ATARI,
}

_REGS_PER_CHIP = 14   # R0..R13; R14/R15 (I/O ports) not stored

# 15-byte header struct (all LE):
#   2s id | B stereo | B loop | H loopStart | H freq |
#   B chipType | H year | B playerClock | B B B reserved
_HDR_FMT = "<2sBBHHBHBBBB"
assert struct.calcsize(_HDR_FMT) == 15


# ---- public API --------------------------------------------------------------

def _song_to_bytes(song: YmSong, *, stereo: int = STEREO_MONO) -> bytes:
    """Serialise *song* to VTX bytes.

    Single-chip songs produce chipType=1 (14 x N frame data).
    Dual-chip songs produce chipType=2 (chip-0 block then chip-1 block, 28 x N).
    """
    n_chips = max(1, song.n_chips)
    chip_type = _CHIP_TURBO if n_chips == 2 else _CHIP_SINGLE
    clock_id  = _CLOCK_MAP.get(int(song.master_clock), _CLOCK_ZX)

    # Build frame data: one 14xN column-major block per chip, chips concatenated.
    per_chip = song.per_chip_songs()
    blocks: list[bytes] = []
    for cs in per_chip[:n_chips]:
        regs14 = np.ascontiguousarray(cs.frames[:, :_REGS_PER_CHIP], dtype=np.uint8)
        blocks.append(np.ascontiguousarray(regs14.T).tobytes())
    frame_data = b"".join(blocks)

    compressed = zlib.compress(frame_data, level=9)

    # Header
    loop_start = max(0, int(song.loop_frame))
    freq       = max(1, min(65535, int(song.frame_rate)))

    header = struct.pack(
        _HDR_FMT,
        _ID_AY,
        stereo,
        1,           # loop always enabled
        loop_start,
        freq,
        chip_type,
        0,           # year unknown
        clock_id,
        0, 0, 0,     # reserved
    )

    # 5 null-terminated metadata strings (immediately after header)
    def _nul(s: str) -> bytes:
        return s.encode("latin-1", "replace") + b"\x00"

    strings = (
        _nul(song.name)
        + _nul(song.author)
        + b"\x00"
        + b"audio2ay3\x00"
        + _nul(song.comment)
    )

    return header + strings + compressed


def to_bytes(song: YmSong, *, stereo: int = STEREO_MONO) -> bytes:
    """Serialise *song* to VTX bytes (single or dual chip in one file)."""
    return _song_to_bytes(song, stereo=stereo)


def write(song: YmSong, path: str, **kwargs) -> None:
    """Write *song* to *path* as a VTX file."""
    with open(path, "wb") as fh:
        fh.write(_song_to_bytes(song, **kwargs))
