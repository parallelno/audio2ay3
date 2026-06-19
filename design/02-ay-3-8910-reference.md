# 02 — AY-3-8910 Hardware Reference

This is the engineering model the emulator and encoder must obey. Values that vary by
implementation (notably the DAC volume table) are flagged; treat the ST-Sound and MAME
reference implementations as the tie-breakers during development.

## 2.1 Block diagram

```
            +-----------+      +--------+
 clock --->| /16 tone   |---->| Tone A |--+
           | dividers   |---->| Tone B |--+--+
           +-----------+      | Tone C |--+  |     +--------+        +-----+
                              +--------+  |  +---->| Mixer  |--vol-->| DAC |--> A out
            +-----------+      +--------+ |  |     | (R7)   |        +-----+
 clock --->| /16 noise  |---->| Noise  |-+--+---->|        |--vol-->| DAC |--> B out
           | divider    |     | (LFSR) |    |     |        |--vol-->| DAC |--> C out
           +-----------+      +--------+    |     +--------+        +-----+
                                            |          ^
            +-----------+   +-----------+   |          | amplitude select
 clock --->| /256 env  |-->| Envelope  |---+----------+  (R8/R9/R10 bit 4)
           | divider   |   | generator |
           +-----------+   +-----------+
```

Three independent tone generators, one noise generator, and one envelope generator feed a
per-channel mixer. Each channel's amplitude is either a fixed 4-bit level or the shared
envelope output. Three independent DACs produce three analog outputs (usually summed to
mono, or panned ABC for "stereo" on some machines).

## 2.2 Register map

The chip exposes **16 registers** (R0–R15). The AY-3-8910 uses R0–R13 for sound; R14/R15
are GPIO I/O ports (irrelevant to audio but present in the YM stream). YM2/YM3 carry 14
registers (R0–R13); YM5/YM6 carry 16.

| Reg | Bits | Function |
|-----|------|----------|
| R0 | 8 | Channel A tone period, fine (low 8 bits) |
| R1 | 4 | Channel A tone period, coarse (high 4 bits) |
| R2 | 8 | Channel B tone period, fine |
| R3 | 4 | Channel B tone period, coarse |
| R4 | 8 | Channel C tone period, fine |
| R5 | 4 | Channel C tone period, coarse |
| R6 | 5 | Noise period (0–31) |
| R7 | 8 | Mixer / I/O enables (see §2.5) |
| R8 | 5 | Channel A amplitude (bits 0–3 level, bit 4 = use envelope) |
| R9 | 5 | Channel B amplitude |
| R10 | 5 | Channel C amplitude |
| R11 | 8 | Envelope period, fine (low 8 bits) |
| R12 | 8 | Envelope period, coarse (high 8 bits) |
| R13 | 4 | Envelope shape/cycle (see §2.7) |
| R14 | 8 | I/O port A data (non-audio) |
| R15 | 8 | I/O port B data (non-audio) |

> **Encoder rule:** unused high bits must be written as 0. R1/R3/R5 use only bits 0–3;
> R6 only bits 0–4; R8/R9/R10 only bits 0–4; R13 only bits 0–3. In YM5/YM6 some of these
> high bits are *repurposed* by ST-Sound for special effects — we must **not** set them
> unless we deliberately emit a digidrum/effect (we don't at 50 Hz). See
> [03-ym-format-reference.md](03-ym-format-reference.md).

## 2.3 Clock and master timing

```
f_master  = chip clock in Hz   (e.g. 1_773_400 ZX Spectrum, 2_000_000 Atari ST)
frame_rate = 50 Hz baseline    (writes per second to the register file)
```

Audio sample rate of the *emulator* output is independent (we render at 44.1 kHz or
48 kHz). The chip's internal generators run off `f_master`; the play routine only updates
registers once per frame.

## 2.4 Tone generation

Each tone channel has a 12-bit period `TP` built from coarse:fine:

```
TP = ((Rcoarse & 0x0F) << 8) | Rfine          # 0 .. 4095
```

The square-wave output frequency:

$$ f_{tone} = \frac{f_{master}}{16 \times TP} $$

Hardware detail: `TP = 0` behaves like `TP = 1` (the divider never fully stops). The square
wave toggles each time the /16 down-counter underflows.

**Inverse (encoder):** given a desired note frequency `f`, the nearest legal period is

$$ TP = \mathrm{round}\!\left(\frac{f_{master}}{16 f}\right), \quad TP \in [1, 4095] $$

### Pitch resolution & quantisation error

Because `TP` is an integer, the achievable frequencies are quantised. Resolution worsens at
high pitch (small `TP`). Worked examples at `f_master = 1_773_400`:

| Note | Ideal f (Hz) | TP | Actual f (Hz) | Cents error |
|------|--------------|----|---------------|-------------|
| A2 | 110.00 | 1008 | 109.97 | −0.4 |
| A4 | 440.00 | 252 | 439.83 | −0.7 |
| A5 | 880.00 | 126 | 879.66 | −0.7 |
| A6 | 1760.00 | 63 | 1759.33 | −0.7 |
| A7 | 3520.00 | 31 | 3575.40 | +27 |

The encoder must understand this: above ~A6 the chip cannot hit equal-tempered pitches
accurately, which is a key reason to keep leads in the chip's "sweet" mid register and to
prefer octave-folding very high content. See
[07-sound-quality-strategy.md](07-sound-quality-strategy.md).

## 2.5 Mixer (R7)

R7 enables tone and/or noise per channel. **A 0-bit ENABLES the source** (active-low).

| Bit | Meaning (0 = enabled) |
|-----|-----------------------|
| 0 | Tone on channel A |
| 1 | Tone on channel B |
| 2 | Tone on channel C |
| 3 | Noise on channel A |
| 4 | Noise on channel B |
| 5 | Noise on channel C |
| 6 | I/O port A direction (1 = output) — keep 0 for audio safety |
| 7 | I/O port B direction (1 = output) — keep 0 |

A channel may mix **both** tone and noise (e.g., tone+noise for a snare-ish timbre). When a
channel's amplitude is 0 it is silent regardless of mixer bits.

> **Encoder rule:** to play a clean tone on A, clear bit 0 and set bit 3. For a pure noise
> hit on C, set bit 2 and clear bit 5. Default safe mixer = `0b00111111` (all off).

## 2.6 Noise generation

A single 17-bit Linear Feedback Shift Register (LFSR) generates pseudo-random noise shared
by all three channels. Taps are bits 0 and 3 (XOR); the output bit gates the channel
amplitude. The 5-bit noise period `NP` (R6, 0–31) sets the rate:

$$ f_{noise} = \frac{f_{master}}{16 \times NP} $$

`NP = 0` behaves like `NP = 1`. Lower `NP` → brighter/hissier noise; higher `NP` → coarser,
lower-pitched noise useful for toms/bass-drum bodies.

> **Encoder note:** there is exactly **one** noise generator. All percussion that uses noise
> shares this single period at any instant — a core constraint for drum design
> (§[07](07-sound-quality-strategy.md)).

## 2.7 Envelope generator

One shared envelope generator produces a time-varying amplitude. Its 16-bit period `EP`:

```
EP = (R12 << 8) | R11          # 0 .. 65535
```

Per-step frequency and full-cycle frequency (the AY ramp has 16 steps; the YM2149 DAC is
finer but the ramp count is the same model we use):

$$ f_{env\_step} = \frac{f_{master}}{256 \times EP}, \qquad f_{env\_cycle} = \frac{f_{master}}{256 \times 16 \times EP} $$

A channel uses the envelope when **bit 4 of its amplitude register is set** (R8/R9/R10);
bits 0–3 are then ignored for that channel.

### Envelope shapes (R13)

Four control bits select the ramp behaviour:

| Bit | Name | Effect |
|-----|------|--------|
| 0 | HOLD | stop after first cycle |
| 1 | ALTERNATE | reverse direction each cycle |
| 2 | ATTACK | start rising (1) vs falling (0) |
| 3 | CONTINUE | keep cycling (1) vs go to 0 and hold (0) |

Useful shapes:

| R13 | Shape | Description | Musical use |
|-----|-------|-------------|-------------|
| `0x08` | `\|\|\|\|` | repeating saw-down | buzzy sustained bass |
| `0x0A` | `/\/\` | repeating triangle | vibrato-ish "buzzer" tone |
| `0x0C` | `////` | repeating saw-up | bright buzz |
| `0x0E` | `\/\/` | repeating triangle (alt phase) | smooth buzzer |
| `0x09` | `\___` | one decay then silence | **percussion decay**, plucks |
| `0x0D` | `‾‾‾‾` | attack then hold high | swells |
| `0x00`–`0x07`, `0x0F` | one-shot decay/hold | transients |

> Writing R13 **restarts** the envelope (re-triggers from the start of the shape). This is how
> we re-strike percussive/plucked sounds each frame without sub-frame writes. Repeatedly
> writing R13 every frame is legal and common.

## 2.8 DAC volume / amplitude table

The 4-bit amplitude (0–15) maps to output via a **non-linear, roughly logarithmic** DAC.
The exact normalised values are device-measured and differ slightly between AY-3-8910 and
YM2149. The emulator must use a measured table, **not** a linear ramp.

Representative normalised 16-entry table (AY-3-8910, MAME-style; final implementation should
adopt the exact table from the chosen reference):

```
level :  0      1      2      3      4      5      6      7
amp   : 0.0000 0.0076 0.0110 0.0158 0.0231 0.0344 0.0519 0.0764
level :  8      9      10     11     12     13     14     15
amp   : 0.1170 0.1632 0.2392 0.3536 0.5043 0.6261 0.8071 1.0000
```

Two consequences for the **encoder**:

1. **Loudness is logarithmic.** Mapping a linear amplitude envelope to levels 0–15 sounds
   wrong; map perceptual loudness (≈ dB) to the table by nearest-amplitude lookup.
2. **Low levels are very quiet and sparse.** Levels 1–4 are tightly bunched near silence;
   most musical dynamics live in levels 8–15. Dynamic compression *into* this range happens
   upstream (legal — it shapes registers, not rendered audio).

> The YM2149 in envelope mode exposes a finer 5-bit (32-step) ramp internally; our model
> uses the 16-step amplitude resolution for fixed levels, which is what the AY-3-8910 offers.

## 2.9 Per-channel mixing to output

For each output sample the emulator computes, per channel `c ∈ {A,B,C}`:

```
tone_on_c  = (R7 bit c   == 0)
noise_on_c = (R7 bit c+3 == 0)
gate_c     = (tone_on_c  ? tone_level_c : 1) AND (noise_on_c ? noise_bit : 1)
level_c    = (R[8+c] bit4) ? envelope_level : (R[8+c] & 0x0F)
out_c      = DAC[level_c] * gate_c
```

The three `out_c` are summed (and optionally panned) and scaled to the render bit depth. The
precise gating semantics (tone level is the square's 0/1, noise is the LFSR bit, and a
disabled source contributes a constant 1 so it does not mute the channel) follow the standard
PSG model and are validated against reference renders.

## 2.10 Reference clocks

| Machine | Clock (Hz) | Notes |
|---------|-----------|-------|
| ZX Spectrum 128 | 1_773_400 | Project default. |
| Atari ST (YM2149) | 2_000_000 | YM3 assumes this. |
| Amstrad CPC | 1_000_000 | Coarser pitch grid. |
| MSX | 1_789_772 | NTSC colourburst. |

The chosen clock is written into the YM header (§[03](03-ym-format-reference.md)) so players
reproduce pitch correctly. The encoder computes all periods from the *configured* clock.

## 2.11 Hard limits the encoder must enforce

- Tone period ∈ [1, 4095]; clamp and warn on out-of-range pitch.
- Noise period ∈ [1, 31]; one global noise pitch at a time.
- Amplitude level ∈ [0, 15]; envelope is a per-channel boolean override.
- Envelope period ∈ [1, 65535]; one global envelope at a time.
- Exactly one register write set per frame (50 Hz baseline). **No sub-frame writes.**
- Reserved/high bits = 0 unless intentionally encoding a documented effect (we do not).
