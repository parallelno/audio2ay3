"""Write a :class:`YmSong` as a VTX file (Vortex Tracker register-dump format).

VTX format (V_Soft / Vortex Project), as expected by AY_Emul 2.x and described in
its README.TXT:

Header (16 bytes, all little-endian)
-------------------------------------
Offset  Size  Type    Field
 0       2    word    ID: b"ay" (AY-3-8910/12) or b"ym" (YM2149)
 2       1    byte    stereo/loop byte — bits 0-2 = stereo mode:
                        0=MONO  1=ABC  2=ACB  3=BAC  4=BCA  5=CAB  6=CBA
 3       2    word    loopStart — loop VBL frame index (0 = beginning of melody)
 5       4    dword   chipFreq  — AY clock in Hz (ZX=1 773 400, Atari=2 000 000)
 9       1    byte    playerFreq — player interrupt rate in VBL/sec (50 for PAL)
10       2    word    year      — year of composition (0 = unknown)
12       4    dword   unpackedSize — size of uncompressed frame data in bytes

Strings (immediately after header, 5 × null-terminated)
---------------------------------------------------------
  title, author, source-program, tracker-name, comment

Compressed data (immediately after strings)
--------------------------------------------
  Raw LHA -lh5- compressed payload — no LHA archive file header, no end marker.
  Decompresses to unpackedSize bytes of YM3-compatible frame data:
      R0[0..N-1], R1[0..N-1], …, R13[0..N-1]   (14 regs × N frames, column-major)

Dual-AY songs
-------------
VTX is inherently single-chip.  For ``n_chips == 2`` callers should iterate
:meth:`YmSong.per_chip_songs` and call :func:`write` once per chip (handled by
:func:`audio2ay3.cli._write_song`).  :func:`write` and :func:`to_bytes` always
encode the first 14 register columns (chip 0) only.
"""

from __future__ import annotations

import struct

import numpy as np

from .model import YmSong

# ── header constants ───────────────────────────────────────────────────────────
_ID_AY: bytes = b"ay"   # AY-3-8910 / AY-3-8912
_ID_YM: bytes = b"ym"   # YM2149

STEREO_MONO = 0
STEREO_ABC  = 1
STEREO_ACB  = 2
STEREO_BAC  = 3
STEREO_BCA  = 4
STEREO_CAB  = 5
STEREO_CBA  = 6

# VTX stores R0..R13 only; R14/R15 (I/O ports) are absent
_REGS_PER_CHIP = 14

# Fixed header struct (all LE): 2s id, B stereo, H loopStart, I chipFreq,
#                                B playerFreq, H year, I unpackedSize
# Sizes: 2+1+2+4+1+2+4 = 16 bytes
_HDR_FMT = "<2sBHIBHI"
assert struct.calcsize(_HDR_FMT) == 16


# ── LHA -lh5- store compressor ────────────────────────────────────────────────

def _lha_lh5_store(data: bytes) -> bytes:
    """Encode *data* as a valid LHA ``-lh5-`` stream using store-only (no LZ77) coding.

    Every input byte is emitted as a literal using a flat 8-bit Huffman code
    (symbol k → 8-bit canonical code = k).  Output is valid ``-lh5-`` and slightly
    larger than the input (~6 bytes of header per 65 535-byte block).

    Returns raw compressed-stream bytes — **no** LHA archive file header and **no**
    archive end-marker (0x00), exactly what the VTX format expects.
    """
    _MAX_BLOCK = 65_535  # max code-words per LH5 block (16-bit count field)

    out = bytearray()
    buf = 0   # MSB-first bit accumulator
    cnt = 0   # pending bits in buf

    def emit(n: int, v: int) -> None:
        nonlocal buf, cnt
        buf = (buf << n) | (v & ((1 << n) - 1))
        cnt += n
        while cnt >= 8:
            cnt -= 8
            out.append((buf >> cnt) & 0xFF)
            buf &= (1 << cnt) - 1

    i = 0
    total = len(data)
    while i < total:
        chunk_end = min(i + _MAX_BLOCK, total)
        block_size = chunk_end - i   # number of literal code-words

        # ── block count (16 bits) ────────────────────────────────────────────
        emit(16, block_size)

        # ── T-tree (encodes C-tree lengths) ─────────────────────────────────
        # _read_pt_len(NT=19, TBIT=5, 3): getbits(5) = n; if n==0: single=getbits(5)
        # We want single=10 so every C-tree length = 10-2 = 8.
        emit(5, 0)    # T n=0
        emit(5, 10)   # T single=10

        # ── C-tree (literal/length Huffman, 256 symbols all length 8) ───────
        # _read_c_len: getbits(CBIT=9) = n; then for each symbol: decode T-tree.
        # T-tree is single=10 → zero bits consumed per symbol → all lengths = 8.
        emit(9, 256)  # C n=256  (no additional bits for the 256 lengths)

        # ── P-tree (positions, never used since we emit only literals) ───────
        # _read_pt_len(NP=14, PBIT=4, -1): getbits(4)=n; if n==0: single=getbits(4)
        emit(4, 0)    # P n=0
        emit(4, 0)    # P single=0

        # ── literal data ─────────────────────────────────────────────────────
        # The flat C-tree gives canonical code for symbol k = 8-bit value k.
        for j in range(i, chunk_end):
            emit(8, data[j])

        i = chunk_end

    # Flush remaining bits (zero-padded on the right)
    if cnt > 0:
        out.append(buf << (8 - cnt))

    return bytes(out)


# ── public API ────────────────────────────────────────────────────────────────

def _song_to_bytes(song: YmSong, *, stereo: int = STEREO_MONO) -> bytes:
    """Serialise the first chip of *song* to VTX bytes."""
    frames = np.ascontiguousarray(song.frames, dtype=np.uint8)
    n_frames = frames.shape[0]

    # Chip 0: columns 0..13; R14/R15 absent in VTX
    regs14 = frames[:, :_REGS_PER_CHIP]  # (n_frames, 14)

    # Column-major layout: R0[0..N-1], R1[0..N-1], …, R13[0..N-1]  (= YM3 layout)
    frame_data = np.ascontiguousarray(regs14.T).tobytes()
    unpacked_size = len(frame_data)

    compressed = _lha_lh5_store(frame_data)

    # ── fixed header ──────────────────────────────────────────────────────────
    loop_start  = max(0, int(song.loop_frame))
    chip_freq   = max(1, int(song.master_clock))   # Hz
    player_freq = max(1, min(255, int(song.frame_rate)))  # VBL/sec

    header = struct.pack(
        _HDR_FMT,
        _ID_AY,
        stereo,
        loop_start,
        chip_freq,
        player_freq,
        0,              # year unknown
        unpacked_size,
    )

    # ── 5 null-terminated metadata strings ───────────────────────────────────
    def _nul(s: str) -> bytes:
        return s.encode("latin-1", "replace") + b"\x00"

    strings = (
        _nul(song.name)      # title
        + _nul(song.author)  # author
        + b"\x00"            # source program (rip origin — not tracked)
        + b"audio2ay3\x00"   # tracker / editor
        + _nul(song.comment) # comment
    )

    return header + strings + compressed


def to_bytes(song: YmSong, *, stereo: int = STEREO_MONO) -> bytes:
    """Serialise *song* (chip 0) to VTX bytes.

    For dual-chip songs only chip 0 is written.  Use :func:`audio2ay3.cli._write_song`
    to obtain one VTX file per chip.
    """
    return _song_to_bytes(song, stereo=stereo)


def write(song: YmSong, path: str, **kwargs) -> None:
    """Write *song* (chip 0) to *path* as a VTX file."""
    with open(path, "wb") as fh:
        fh.write(_song_to_bytes(song, **kwargs))
