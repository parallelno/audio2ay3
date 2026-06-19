# 01 — Overview, Goals & Scope

## 1.1 Objective

Convert **instrumental audio** into a **50 Hz AY-3-8910 register stream** that, when
played on real hardware (or an accurate emulator), reproduces the original music as
faithfully as the chip allows.

The project is equal parts **research** (what mapping strategies sound best within a
3-voice + 1-noise budget?) and **engineering** (a modular, testable, accelerated
pipeline that produces hardware-legal `.ym` files).

## 1.2 Scope restriction

**Only instrumental music is supported.**

- Excluded: singing, speech, rap, narration, choirs, vocal pads, any sung/spoken content.
- Included: any combination of musical instruments, **including percussion and drums**.
- The system **may assume** input contains no vocals. We do not need a vocal detector to
  *reject* input, though an optional advisory check is listed as a stretch goal.

Rationale: monophonic-per-channel square-wave synthesis cannot represent the formant-rich,
continuously-pitched nature of the human voice without sounding like a detuned lead. Removing
vocals from scope lets every design decision optimise for pitched instruments and percussion.

## 1.3 Target platform

| Aspect | Specification |
|--------|---------------|
| Sound chip | AY-3-8910 (standard) / YM2149 compatible |
| Tone channels | 3 square-wave (A, B, C) |
| Noise | 1 shared pseudo-random generator |
| Envelope | 1 shared hardware envelope generator |
| Extra audio hardware | **None** — chip only |
| Output | Hardware-compatible register stream (`.ym`) |
| Baseline update rate | 50 Hz (20 ms per frame) |
| Default master clock | 1.7734 MHz (ZX Spectrum); configurable (2.0 MHz Atari ST, 1.0 MHz CPC, 1.7897 MHz MSX) |

See [02-ay-3-8910-reference.md](02-ay-3-8910-reference.md) for the full hardware model.

## 1.4 Guiding principle

> **Every processing decision must respect AY-3-8910 hardware limits, so the output is
> both high quality and accurately playable on real hardware.**

Operationally this means three rules:

1. **Upstream-only quality.** No equaliser, reverb, compressor, or sample-rate trickery is
   permitted *after* the emulator. If a tune sounds better, it is because the register
   stream is better — not because we polished the rendered audio.
2. **Legal registers only.** The encoder may never emit a register state a real chip cannot
   reach (e.g., out-of-range periods, sub-frame updates at 50 Hz, digidrum sample playback
   that requires >50 Hz writes).
3. **Emulator is ground truth.** The same emulator validates YM files *and* renders previews,
   so "what we measure" equals "what a chip produces."

## 1.5 What "high fidelity" means here

Fidelity is **perceptual**, not spectral-exact. Square waves at 4-bit volume will never match
a recording's spectrum. The goals, in priority order:

1. **Correct melody & harmony** — the listener recognises the tune and its chords.
2. **Stable, jitter-free tone** — no warbling from frame-to-frame pitch/volume thrash.
3. **Solid, consistent percussion** — drums read as drums, with steady timing and weight.
4. **Musical voice-leading** — when >3 notes sound at once, the *right* 3 survive.
5. **Full chip utilisation** — tone channels, noise, and the envelope generator all earn
   their keep instead of sitting idle.

See [07-sound-quality-strategy.md](07-sound-quality-strategy.md) for how each is achieved.

## 1.6 Success criteria

| Criterion | Measure |
|-----------|---------|
| Emulator accuracy | Renders reference YM files; A/B against ST-Sound / MAME output within tight perceptual tolerance (§[10](10-testing-validation.md)). |
| Round-trip legality | 100% of emitted `.ym` files load in an independent player (e.g., ST-Sound, libayemu) without error. |
| Melody recognisability | Blind listener test: tune identifiable on ≥90% of curated instrumental samples. |
| Jitter | No audible pitch warble on sustained notes; quantified by frame-to-frame period-change rate staying under a tuned threshold. |
| Percussion | Drum onsets from the source land within ±1 frame in the output on the drum-loop fixtures. |
| Performance | A 3-minute track converts in well under real-time on a CUDA GPU; emulation renders faster than real-time on CPU. |

## 1.7 Out of scope (initial release)

- Vocal/melodic transcription of *singing*.
- MIDI import/export (a possible later convenience format).
- Real-time / streaming conversion (batch only at first).
- Non-AY chips (SID, POKEY, SN76489) — though the architecture should not actively prevent
  a future backend.
- Digidrum sample playback (requires >50 Hz register writes; see
  [07-sound-quality-strategy.md](07-sound-quality-strategy.md#72-percussion)).

## 1.8 Glossary

| Term | Meaning |
|------|---------|
| **PSG** | Programmable Sound Generator — the AY-3-8910 family. |
| **Frame** | One register snapshot; at 50 Hz a frame is 20 ms. |
| **Register stream** | The ordered sequence of 14/16-register frames driving the chip. |
| **YM** | ST-Sound register-dump file format (YM2…YM6). See [03](03-ym-format-reference.md). |
| **Tone period (TP)** | 12-bit divisor setting a tone channel's pitch. |
| **Voice allocation** | Assigning detected notes to the 3 tone channels over time. |
| **HPSS** | Harmonic/Percussive Source Separation. |
| **Digidrum** | AY technique playing 4-bit PCM via rapid volume writes (out of scope at 50 Hz). |
