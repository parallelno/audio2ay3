# 03 — YM File Format Reference

The YM format (ST-Sound, by Arnaud Carré / "Leonard") is a **register-dump** format: it
stores the per-frame AY/YM2149 register values that the original play routine produced. It
is *not* executable code. That is exactly what we want — our converter produces register
frames, and our emulator consumes them.

We **read** YM2/YM3/YM3b/YM5/YM6 (for the validator). We **write** YM5 or YM6 (for the
converter), defaulting to **YM6** uncompressed for simplicity, with optional LHA packing.

> Source of truth: the ST-Sound format page (leonard.oxg.free.fr/ymformat.html) and the
> ST-Sound reference loader (`YmMusic.cpp`). Byte offsets below are validated against those
> during implementation.

## 3.1 Outer container: LHA packing

Most `.ym` files found in the wild are **LHA archives** (method `-lh5-`) containing a single
member whose decompressed bytes start with the YM magic (`YM5!`, `YM6!`, …). ST-Sound
depacks transparently.

Reader algorithm:

```
data = read_file(path)
if looks_like_lha(data):          # member header, '-lh5-' method id
    data = lha_unpack(data)       # single member
parse_ym(data)                    # now starts with 'YMx!'
```

Writer: emit raw YM bytes; optionally wrap in LHA `-lh5-`. For v1 we write **uncompressed**
YM (valid and widely supported) and add LHA packing as a later enhancement. See
[12-tech-stack-dependencies.md](12-tech-stack-dependencies.md) for the depack library choice.

## 3.2 YM5 / YM6 layout (what we write)

All multi-byte integers are **big-endian**. YM5 and YM6 share this header; YM6 only differs
in how the special-effect bits in the virtual registers are interpreted.

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | File ID | `"YM5!"` or `"YM6!"` |
| 4 | 8 | Check string | `"LeOnArD!"` (literal) |
| 12 | 4 | `nbFrames` | number of register frames |
| 16 | 4 | `songAttributes` | bit0 = **interleaved** data block |
| 20 | 2 | `nbDigidrums` | number of embedded samples (we write 0) |
| 22 | 4 | `masterClock` | chip clock in Hz (e.g. 1773400) |
| 26 | 2 | `frameRate` | replay rate in Hz (e.g. 50) |
| 28 | 4 | `loopFrame` | frame index to loop to (0 if none) |
| 32 | 2 | `addSize` | size of extra header data (we write 0) |
| 34 | addSize | extra data | skipped if `addSize`>0 |
| … | varies | digidrum block | per sample: 4-byte size + 4-bit PCM (none for us) |
| … | n+1 | song name | NUL-terminated string |
| … | n+1 | author name | NUL-terminated string |
| … | n+1 | comment | NUL-terminated string |
| … | 16×nbFrames | register data | layout depends on `songAttributes` bit0 |
| end | 4 | end marker | `"End!"` |

### Register data block

- **Interleaved (`songAttributes` bit0 = 1):** stored register-major. All `nbFrames` values
  of R0, then all of R1, …, through R15. This packs better and is the ST-Sound default.

  ```
  R0[0] R0[1] ... R0[n-1]  R1[0] ... R1[n-1]  ...  R15[0] ... R15[n-1]
  ```

- **Non-interleaved (bit0 = 0):** stored frame-major (R0..R15 for frame 0, then frame 1, …).

We will **write interleaved** (matches ST-Sound, friendlier to LHA) and **read both**.

### Virtual registers / special effects (why we keep high bits clean)

In YM5/YM6, ST-Sound overloads spare bits of some registers to drive Atari timer effects
(digidrums, sync-buzzer, sid-voice, sinus-SID). Examples: high bits of R1/R3 select digidrum
voice & timer predivisor; R6/R14/R15 carry sample number and timer count, etc. Because we run
at a strict 50 Hz with no sample playback, **we leave every such bit at 0** and emit only the
plain PSG state. This guarantees the file is a "classic" tune that any AY player reproduces
identically. (Full effect tables live in the ST-Sound docs; we reference but do not use them.)

## 3.3 YM3 / YM3b (what we mainly read)

Simpler, very common for ZX/Atari rips:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | `"YM3!"` (YM3) or `"YM3b"` |
| 4 | 14×nbFrames | register data, **interleaved**, 14 registers (R0–R13) |
| end (YM3b only) | 4 | loop frame (big-endian) |

YM3 implies `frameRate = 50 Hz` and `masterClock = 2_000_000 Hz` (Atari ST) unless the player
is told otherwise. `nbFrames = (dataLen) / 14`. R14/R15 are treated as 0.

## 3.4 YM2

Legacy Atari format, treat like YM3 (14 interleaved registers) for the reader with a couple of
fixed-effect quirks. Low priority; supported best-effort for the validator.

## 3.5 In-memory model

A single normalised structure represents any loaded or to-be-written tune:

```python
@dataclass
class YmSong:
    version: str            # "YM6", "YM3", ...
    frames: np.ndarray      # shape (nb_frames, 16), dtype=uint8  (R0..R15)
    master_clock: int       # Hz
    frame_rate: int         # Hz (50 default)
    loop_frame: int         # index, 0 if none
    name: str = ""
    author: str = ""
    comment: str = ""
    # digidrums intentionally unsupported on the write path
```

- The **reader** fills `frames` by de-interleaving as needed and zero-extending YM3's 14
  registers to 16.
- The **emulator** consumes `frames` row-by-row at `frame_rate`.
- The **writer** serialises `frames` (interleaved) into YM6 with the header above.

This isolates every other module from format quirks: analysis/mapping/encoding only ever deal
with the `(nb_frames, 16)` array plus the clock/rate metadata.

## 3.6 Reader/writer responsibilities

| Component | Reads | Writes | Notes |
|-----------|-------|--------|-------|
| `ymformat.ym_reader` | YM2/3/3b/5/6 (+LHA) | — | Normalises to `YmSong`. |
| `ymformat.ym_writer` | — | YM6 (raw; LHA optional) | Interleaved, clean high bits. |
| `ymformat.lha` | LHA `-lh5-` | (optional) | Depack for reader; pack later. |

## 3.7 Validation hooks

- **Round-trip test:** write a `YmSong`, read it back, assert byte-identical `frames` and
  metadata.
- **Cross-player test:** emitted files must load in an *independent* implementation
  (ST-Sound CLI or `libayemu`/`ayemu`) without error — proof of hardware-legality at the
  format level. See [10-testing-validation.md](10-testing-validation.md).
- **Golden headers:** unit tests assert exact header bytes for a known small `YmSong`.
