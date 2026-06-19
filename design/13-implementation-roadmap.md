# 13 — Implementation Roadmap

A phased plan that front-loads the **trusted emulator** (so everything afterward is measurable),
then builds the converter outward from a minimal end-to-end skeleton, tuning fidelity last. Each
phase has concrete deliverables and exit criteria.

```mermaid
flowchart LR
    P0[P0 Scaffold] --> P1[P1 Emulator + validate]
    P1 --> P2[P2 YM I/O complete]
    P2 --> P3[P3 Skeleton convert/preview]
    P3 --> P4[P4 Analysis depth]
    P4 --> P5[P5 Arrangement + quality]
    P5 --> P6[P6 Acceleration]
    P6 --> P7[P7 Scalability]
    P7 --> P8[P8 Hardening]
```

> Phases are ordered by dependency and risk, not by calendar. No time estimates are given by
> design; each phase ends when its exit criteria are met.

---

## Phase 0 — Project scaffold

**Goal:** a runnable, tested skeleton.

- `pyproject.toml`, `src/audio2ay3/` packages, `.venv`, `ruff`/`mypy`/`pytest` wired.
- `config.py` (`ChipConfig`, `RunConfig`), `cli.py` with three no-op subcommands + `--help`.
- `utils/` (audio_io, gpu detection, logging), `build/` gitignored.

**Exit:** `audio2ay3 --help` works; empty test suite green; lint/type clean.

---

## Phase 1 — Emulator + `validate` (**Milestone 1**)

**Goal:** the trusted ear. *This is the foundational milestone from the brief.*

- `chip/` tone, noise (17-bit LFSR), envelope (all shapes), measured DAC table, mixer.
- `Ay3Emulator.render_song/render_frames/step_frame`; oversample + decimate anti-aliasing.
- `render/renderer.py` + `render/mp3.py`; numba-accelerate the inner loop.
- Minimal `ym_reader` (enough to load test YMs) + `validate` CLI (YM → MP3/WAV).

**Exit (acceptance):**
- Renders public YM2/3/5/6 (+LHA) without error.
- Single-tone pitch < 1 cent (mid register); noise/envelope shapes match reference.
- A/B perceptual diff vs ST-Sound/MAME under threshold on the reference set
  (§[10](10-testing-validation.md#103-ym-playback-validation-workflow-hardware-faithfulness)).
- Generator unit tests green.

---

## Phase 2 — YM I/O complete

**Goal:** read everything we need, write clean YM6.

- `ymformat/model.py` (`YmSong`), full `ym_reader` (YM2/3/3b/5/6, de-interleave, LHA depack).
- `ym_writer` (interleaved YM6, clean high bits, golden-byte tests).
- Round-trip + cross-player legality tests.

**Exit:** write→read→write byte-stable; emitted files load in an independent player.

---

## Phase 3 — End-to-end skeleton `convert` / `preview`

**Goal:** thinnest possible full pipeline producing a *legal* (if not yet pretty) `.ym`.

- `analysis/decode.py`; a **monophonic** pitch path (e.g. pYIN) + simple onset detector.
- Trivial mapping: 1 tone channel for the lead, noise for onsets.
- `encode/quantize.py` + `encode/register_stream.py` (the legality choke point).
- `pipeline.py` wiring; `convert` writes `.ym`, `preview` emulates to MP3.

**Exit:** `convert`/`preview` run on `samples/short/*`; output is hardware-legal and recognisably
follows the input's melody + hits, end-to-end on the real toolchain.

---

## Phase 4 — Analysis depth (research-heavy)

**Goal:** real polyphony and percussion understanding.

- `analysis/hpss.py` (harmonic/percussive split).
- `analysis/features.py` (STFT/CQT, torch/GPU-ready).
- `analysis/pitch.py` **multi-F0** (CQT-salience default; NMF/Klapuri pluggable).
- `analysis/onsets.py` drum detection + kick/snare/hat classification.
- `analysis/dynamics.py` perceptual loudness.

**Exit:** on synthetic chords, the correct 3 fundamentals are reported per frame; on the
drum-loop fixture, onsets/kinds detected within ±1 frame.

---

## Phase 5 — Arrangement + sound quality (the fidelity payoff)

**Goal:** make it *sound good* within constraints — all upstream of registers.

- `mapping/voices.py` (assignment problem: Hungarian/greedy, continuity, bass bias, salience).
- `mapping/percussion.py` (drum recipes, single-noise collision policy, envelope re-trigger).
- `mapping/smoothing.py` (hysteresis, pitch debounce, slew limiting, channel-swap damping).
- Envelope/buzzer timbres; octave-folding & sweet-spot bias in quantise.
- Named profiles (`quality`/`balanced`/`fast`); `--explain` debug artefacts.

**Exit (quality gates, §[10](10-testing-validation.md#104-perceptual--musical-regression-metrics)):**
jitter below baseline on sustained notes; onsets ±1 frame; melody recognisable on ≥90% of the
curated set in blind listening; zero illegal registers.

---

## Phase 6 — Performance & acceleration

**Goal:** practical speed; meet the perf targets.

- torch/CUDA path for spectral + multi-F0; `--no-gpu` fallback verified.
- Thread/process pools for I/O, segment-parallel analysis, chunked emulation.
- `--profile-run` stage timing; optimise only profiled hotspots.

**Exit:** emulate ≥10× real-time (CPU); convert a 3-min track well under real-time on GPU;
CPU-only CI passes fully.

---

## Phase 7 — Scalability

**Goal:** prove the config-only growth paths.

- `--frame-rate 100` (ms-based smoothing windows).
- `--chips 2` dual-AY: 6-channel allocation, dual noise/env, two-`.ym` output, summed emulation.
- Smoke tests keep both alive.

**Exit:** dual-AY @ 100 Hz runs end-to-end and audibly improves density without code redesign.

---

## Phase 8 — Hardening & docs

**Goal:** robustness and usability.

- Edge-case inputs (silence, clipping, odd sample rates, very long tracks).
- Full CLI UX (progress, exit codes, helpful errors), README/usage docs.
- Regression goldens + perceptual metrics locked in CI; determinism check.

**Exit:** green CI (CPU-only), documented usage, stable profiles, no known legality violations.

---

## Deliverables summary

| Deliverable (brief) | Where |
|---------------------|-------|
| Detailed implementation plan | this `design/` folder |
| Documented system architecture & stages | [04](04-system-architecture.md), [05](05-emulator-design.md), [06](06-conversion-pipeline.md) |
| Validation workflow via YM playback | [10](10-testing-validation.md#103-ym-playback-validation-workflow-hardware-faithfulness) |
| AY Validator (YM→MP3) | Phase 1, [08](08-cli-design.md#82-validate--ym-playback-milestone-1-deliverable) |
| Converter (audio→YM) | Phases 3–5, [08](08-cli-design.md#83-convert--audio--ym) |
| Preview (audio→MP3) | Phases 3–5, [08](08-cli-design.md#84-preview--audio--emulated-mp3) |

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Multi-F0 accuracy on dense mixes | High | High | pluggable estimators; optional NN stems; salience-weighted allocation tolerates imperfect detection |
| 3-voice budget loses too much music | Med | High | strong voice-leading cost function; dual-AY path for headroom |
| Jitter despite smoothing | Med | Med | ms-based hysteresis/slew tuning; objective jitter metric in CI |
| Emulator inaccuracy | Low | High | cross-validate vs ST-Sound/MAME before depending on it |
| Percussion realism with one noise gen | Med | Med | curated drum recipes + collision policy; dual-AY gives a 2nd noise gen |
| Performance on long tracks | Med | Med | segment streaming, GPU spectral path, chunked emulation |
| Dependency/CUDA friction | Med | Low | CPU-only default works; heavy ML behind extras |

## Definition of done (v1)

- All three CLI commands work on the provided `samples/`.
- Emitted `.ym` files are hardware-legal and load in an independent player.
- Quality gates met on the curated set; CPU-only CI green; deterministic with `--seed`.
- `design/` reflects the final architecture (kept in sync with code).
