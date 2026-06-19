# audio2ay3 — Design & Implementation Plan

High-fidelity conversion of **instrumental audio** (MP3, WAV, FLAC, OGG, …) into a
**50 Hz AY-3-8910 register stream**, rendered back to audio for validation.

This folder is the authoritative design record for the project. It is the single
deliverable required before implementation begins, and it doubles as the living
architecture reference once code lands.

---

## How to read this folder

Read top-to-bottom for a full picture, or jump to the document that matches your task.

| # | Document | Purpose |
|---|----------|---------|
| 00 | [README.md](README.md) | This index. |
| 01 | [01-overview-goals-scope.md](01-overview-goals-scope.md) | Objective, scope restriction, guiding principles, success criteria. |
| 02 | [02-ay-3-8910-reference.md](02-ay-3-8910-reference.md) | Hardware reference: registers, timing, DAC, noise/envelope. |
| 03 | [03-ym-format-reference.md](03-ym-format-reference.md) | YM2/YM3/YM5/YM6 file format, LHA packing, register semantics. |
| 04 | [04-system-architecture.md](04-system-architecture.md) | Module boundaries, data flow, repository layout. |
| 05 | [05-emulator-design.md](05-emulator-design.md) | **Milestone 1** — accurate emulator + YM→MP3 validator. |
| 06 | [06-conversion-pipeline.md](06-conversion-pipeline.md) | Audio→YM pipeline: analysis, mapping, encoding stages. |
| 07 | [07-sound-quality-strategy.md](07-sound-quality-strategy.md) | Fidelity, jitter-free output, percussion, AY feature usage. |
| 08 | [08-cli-design.md](08-cli-design.md) | CLI commands: `validate`, `convert`, `preview`. |
| 09 | [09-performance-acceleration.md](09-performance-acceleration.md) | Multithreading and CUDA/GPU acceleration. |
| 10 | [10-testing-validation.md](10-testing-validation.md) | Test strategy and the YM-playback validation workflow. |
| 11 | [11-scalability.md](11-scalability.md) | Dual-AY configurations and 100 Hz update rates. |
| 12 | [12-tech-stack-dependencies.md](12-tech-stack-dependencies.md) | Python 3.12, `.venv`, libraries, tooling. |
| 13 | [13-implementation-roadmap.md](13-implementation-roadmap.md) | Phased milestones, deliverables, risks. |

---

## The one-paragraph summary

We build an **accurate AY-3-8910 / YM2149 emulator** first, and a tool that renders
existing **YM files to MP3**. That gives us a trusted "ear" to measure everything else
against. We then build the **converter**: it decodes instrumental audio, separates
harmonic and percussive content, estimates the most musically important pitches per
20 ms frame, allocates them across the chip's three tone channels and single noise
generator, quantises pitch/amplitude to the chip's coarse grid, and serialises the
result as a hardware-faithful **YM register stream**. The **preview** command chains the
converter and the emulator so a user can hear the AY result without real hardware. No
post-AY audio sweetening is ever allowed — every quality decision happens *upstream* of
register generation so the `.ym` output plays identically on a real chip.

---

## Non-negotiable constraints (read before designing anything)

1. **Instrumental only.** Vocals/speech/choir are out of scope and may be assumed absent.
2. **No post-AY enhancement.** Once registers are emulated, the audio is final. All
   fidelity work happens before register-stream generation.
3. **Hardware-faithful output.** Every emitted `.ym` must be playable on a real
   AY-3-8910 at the declared frame rate with no illegal register states.
4. **50 Hz baseline.** Default update rate is 50 Hz (20 ms/frame). The architecture must
   scale to 100 Hz and to dual-AY (6 tone channels) without redesign.

---

## Status

| Item | State |
|------|-------|
| Design docs | In progress (this folder) |
| Emulator (Milestone 1) | Not started |
| Converter | Not started |
| CLI | Not started |

> Implementation has **not** started. This folder must be reviewed and accepted first.
