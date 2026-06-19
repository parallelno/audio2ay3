"""Transparent LHA depacking for YM files.

On-disk YM tunes are almost always wrapped in a single-member LHA archive using the ``-lh5-``
method (13-bit sliding dictionary + dynamic Huffman). This module unwraps that archive in pure
Python so :mod:`ym_reader` can read packed files directly, with no external tools.

The decoder is a faithful port of the classic *LHa for UNIX* algorithm (Haruhiko Okumura /
Masaru Oki, public domain). Canonical Huffman codes are decoded bit-by-bit (puff.c style),
which is simple and exact; throughput is irrelevant for the few-KB streams in a YM file.
"""

from __future__ import annotations

# --- lh5 parameters -----------------------------------------------------------------------
_THRESHOLD = 3            # shortest encoded match
_NC = 510                 # char/length codes: UCHAR_MAX(255) + MAXMATCH(256) + 2 - THRESHOLD
_CBIT = 9                 # bit width of the c-tree size field
_NP = 14                  # position codes: DICBIT(13) + 1
_PBIT = 4                 # bit width of the position-tree size field
_NT = 19                  # temp-tree codes: CODE_BIT(16) + 3
_TBIT = 5                 # bit width of the temp-tree size field
_DICBIT = 13
_DICSIZ = 1 << _DICBIT    # 8192
_DICMASK = _DICSIZ - 1


def _u16le(data: bytes, off: int) -> int:
    return data[off] | (data[off + 1] << 8)


def _u32le(data: bytes, off: int) -> int:
    return data[off] | (data[off + 1] << 8) | (data[off + 2] << 16) | (data[off + 3] << 24)


def is_lha(data: bytes) -> bool:
    """True if *data* looks like an LHA archive (``-lhX-`` method id at offset 2)."""
    return len(data) > 21 and data[2:5] == b"-lh"


class _BitIn:
    """MSB-first bit reader over a byte string; reads past the end as zero bits."""

    __slots__ = ("_data", "_len", "_pos", "_buf", "_cnt")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._len = len(data)
        self._pos = 0
        self._buf = 0
        self._cnt = 0

    def _need(self, n: int) -> None:
        while self._cnt < n:
            if self._pos < self._len:
                b = self._data[self._pos]
                self._pos += 1
            else:
                b = 0
            self._buf = (self._buf << 8) | b
            self._cnt += 8

    def getbits(self, n: int) -> int:
        if n == 0:
            return 0
        if self._cnt < n:
            self._need(n)
        self._cnt -= n
        val = (self._buf >> self._cnt) & ((1 << n) - 1)
        self._buf &= (1 << self._cnt) - 1
        return val

    def getbit(self) -> int:
        if self._cnt == 0:
            self._need(1)
        self._cnt -= 1
        val = (self._buf >> self._cnt) & 1
        self._buf &= (1 << self._cnt) - 1
        return val


class _Huff:
    """Canonical Huffman decoder built from per-symbol code lengths.

    ``single >= 0`` marks a degenerate table (one symbol, zero-length codes) that decodes to a
    constant without consuming any bits — the LHA "n == 0" special case.
    """

    __slots__ = ("count", "symbols", "maxbits", "single")

    def __init__(self, lengths: list[int] | None, n: int, single: int = -1) -> None:
        self.single = single
        if single >= 0:
            self.count = []
            self.symbols = []
            self.maxbits = 0
            return
        assert lengths is not None
        maxbits = 0
        for i in range(n):
            if lengths[i] > maxbits:
                maxbits = lengths[i]
        count = [0] * (maxbits + 1)
        for i in range(n):
            count[lengths[i]] += 1
        # Symbols sorted by (code length, symbol index) — canonical order.
        offs = [0] * (maxbits + 2)
        for length in range(1, maxbits + 1):
            offs[length + 1] = offs[length] + count[length]
        symbols = [0] * n
        for i in range(n):
            length = lengths[i]
            if length != 0:
                symbols[offs[length]] = i
                offs[length] += 1
        self.count = count
        self.symbols = symbols
        self.maxbits = maxbits

    def decode(self, bitin: _BitIn) -> int:
        if self.single >= 0:
            return self.single
        code = 0
        first = 0
        index = 0
        count = self.count
        getbit = bitin.getbit
        for length in range(1, self.maxbits + 1):
            code |= getbit()
            cnt = count[length]
            if code - cnt < first:
                return self.symbols[index + (code - first)]
            index += cnt
            first = (first + cnt) << 1
            code <<= 1
        raise ValueError("invalid Huffman code in LHA stream")


class _Lh5Decoder:
    """Decode an ``-lh5-`` compressed stream into exactly *orig_size* bytes."""

    def __init__(self, comp: bytes, orig_size: int) -> None:
        self.bitin = _BitIn(comp)
        self.orig_size = orig_size
        self.blocksize = 0
        self.c_huff: _Huff | None = None
        self.pt_huff: _Huff | None = None

    def _read_len_code(self) -> int:
        # Lengths 0..6 are a literal 3-bit value; 7+ is '111' then unary ones then a 0.
        c = self.bitin.getbits(3)
        if c == 7:
            while self.bitin.getbit() == 1:
                c += 1
        return c

    def _read_pt_len(self, nn: int, nbit: int, i_special: int) -> None:
        b = self.bitin
        n = b.getbits(nbit)
        if n == 0:
            self.pt_huff = _Huff(None, 0, single=b.getbits(nbit))
            return
        pt_len = [0] * nn
        i = 0
        while i < n:
            pt_len[i] = self._read_len_code()
            i += 1
            if i == i_special:
                z = b.getbits(2)
                while z > 0:
                    if i < nn:
                        pt_len[i] = 0
                    i += 1
                    z -= 1
        self.pt_huff = _Huff(pt_len, nn)

    def _read_c_len(self) -> None:
        b = self.bitin
        n = b.getbits(_CBIT)
        if n == 0:
            self.c_huff = _Huff(None, 0, single=b.getbits(_CBIT))
            return
        assert self.pt_huff is not None
        c_len = [0] * _NC
        i = 0
        while i < n:
            c = self.pt_huff.decode(b)
            if c <= 2:
                if c == 0:
                    run = 1
                elif c == 1:
                    run = b.getbits(4) + 3
                else:
                    run = b.getbits(_CBIT) + 20
                while run > 0 and i < _NC:
                    c_len[i] = 0
                    i += 1
                    run -= 1
            else:
                if i < _NC:
                    c_len[i] = c - 2
                i += 1
        self.c_huff = _Huff(c_len, _NC)

    def _decode_c(self) -> int:
        if self.blocksize == 0:
            self.blocksize = self.bitin.getbits(16)
            self._read_pt_len(_NT, _TBIT, 3)   # temp tree (used to code the c-tree lengths)
            self._read_c_len()                 # c tree, decoded via the temp tree
            self._read_pt_len(_NP, _PBIT, -1)  # position tree (overwrites pt_huff)
        self.blocksize -= 1
        assert self.c_huff is not None
        return self.c_huff.decode(self.bitin)

    def _decode_p(self) -> int:
        assert self.pt_huff is not None
        j = self.pt_huff.decode(self.bitin)
        if j != 0:
            j = (1 << (j - 1)) + self.bitin.getbits(j - 1)
        return j

    def run(self) -> bytes:
        out = bytearray()
        dic = bytearray(_DICSIZ)
        loc = 0
        target = self.orig_size
        append = out.append
        while len(out) < target:
            c = self._decode_c()
            if c < 256:
                dic[loc] = c
                loc = (loc + 1) & _DICMASK
                append(c)
            else:
                length = c - 256 + _THRESHOLD
                i = (loc - self._decode_p() - 1) & _DICMASK
                for _ in range(length):
                    b = dic[i]
                    dic[loc] = b
                    append(b)
                    i = (i + 1) & _DICMASK
                    loc = (loc + 1) & _DICMASK
                    if len(out) >= target:
                        break
        return bytes(out[:target])


def depack_lha(data: bytes) -> bytes:
    """Return the single member of an LHA archive, decompressing ``-lh5-``/``-lh0-`` members.

    Supports header levels 0, 1, and 2 (level 0 is what virtually all YM files use). Raises
    :class:`NotImplementedError` for compression methods other than ``-lh5-`` (stored ``-lh0-``
    / ``-lhd-`` are copied through).
    """
    if not is_lha(data):
        raise ValueError("not an LHA archive")
    method = bytes(data[2:7])
    level = data[20]
    orig_size = _u32le(data, 11)

    if level == 0:
        data_start = 2 + data[0]
    elif level == 1:
        pos = 2 + data[0]
        while pos + 2 <= len(data):
            ext = _u16le(data, pos)
            if ext == 0:
                pos += 2
                break
            pos += ext
        data_start = pos
    elif level == 2:
        data_start = _u16le(data, 0)
    else:
        raise ValueError(f"unsupported LHA header level {level}")

    comp = data[data_start:]
    if method in (b"-lh0-", b"-lhd-"):  # stored / directory: no compression
        return bytes(comp[:orig_size])
    if method != b"-lh5-":
        raise NotImplementedError(
            f"LHA method {method.decode('latin-1')!r} is not supported (only -lh5-/-lh0-). "
            "Depack with an external tool (e.g. 7-Zip) first."
        )

    out = _Lh5Decoder(comp, orig_size).run()
    if len(out) != orig_size:
        raise ValueError(
            f"LHA depack size mismatch: got {len(out)} bytes, header declares {orig_size}"
        )
    return out
