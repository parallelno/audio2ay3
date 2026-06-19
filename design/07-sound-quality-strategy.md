# 07 — Sound Quality Strategy

The brief's primary objective is **maximising perceived audio fidelity within AY constraints**.
This document specifies *how* each quality goal is achieved — always **upstream** of register
generation, because no post-AY processing is permitted.

The goals, restated with their owning mechanisms:

| Goal | Primary mechanism | Owning stage |
|------|-------------------|--------------|
| Stable, jitter-free tone | hysteresis, debounce, slew limiting | mapping/smoothing |
| Solid, consistent percussion | noise+envelope drum models, collision policy | mapping/percussion |
| Full chip utilisation | tone×3 + noise + envelope arrangement rules | mapping/arrangement |
| Recognisable melody/harmony | salience-weighted voice allocation | mapping/voices |
| Tasteful high-pitch handling | octave folding, sweet-spot bias | encode/quantise |

---

## 7.1 Jitter-free output

"Jitter" here = audible warble/zipper artifacts caused by registers changing too often or
inconsistently between 20 ms frames. Sources and cures:

### Pitch jitter
- **Cause:** analysis pitch wobble crossing `TP` quantisation boundaries; octave flips.
- **Cure:**
  - **Pitch debounce / hysteresis:** require a new target period to persist ≥N frames and to
    differ by a meaningful margin before committing; otherwise hold the current `TP`.
  - **Octave lock:** once a note's octave is chosen, resist flipping unless salience strongly
    favours it (prevents the classic "octave jumping" of naïve trackers).
  - **Note continuity:** keep a sustained note on the same channel so its period only moves when
    the *music* moves.

### Amplitude jitter (zipper noise)
- **Cause:** large per-frame volume jumps; flicker between adjacent DAC levels.
- **Cure:** **slew limiting** — cap |Δlevel| per frame except on detected onsets; small dead-band
  so adjacent-level dithering doesn't oscillate.

### Channel-thrash jitter
- **Cause:** voice allocation reassigning notes between channels frame-to-frame.
- **Cure:** **stealing penalty + lock-in** in the assignment cost
  (§[06](06-conversion-pipeline.md#41-voice-allocation-the-central-problem)); a note, once
  placed, stays put unless displaced by a clearly more important note.

### Update-rate discipline
- **Cause:** "excessive rapid register updates that cause audible artifacts" (brief).
- **Cure:** the encoder writes a register **only when its value changes**; a frame with no
  musical change re-emits the previous values (cheap, stable). At 50 Hz there is exactly one
  update opportunity per 20 ms — we never try to fake sub-frame motion.

> Net effect: registers move *with the music*, not with the analysis noise floor.

---

## 7.2 Percussion

Percussion must be "solid and consistent" despite only **one** shared noise generator and a
50 Hz update budget. Digidrums (PCM via fast volume writes) are **out of scope** because they
require register writes far faster than 50 Hz; we therefore synthesise drums from the chip's
native noise + envelope + tone.

### Drum models (register recipes)

| Drum | Noise period `NP` | Envelope | Tone assist | Notes |
|------|-------------------|----------|-------------|-------|
| **Kick** | mid/high `NP` (short body) | fast decay `0x09 \___`, short `EP` | optional low tone pitch-dropping for "thump" | brief; 2–4 frames |
| **Snare** | low `NP` (bright) | fast decay | optional mid tone layer for "crack" | broadband |
| **Hat (closed)** | very low `NP` (brightest) | very fast decay | none | 1–2 frames |
| **Hat (open)/cymbal** | very low `NP` | slower decay / sustained then cut | none | longer tail |
| **Tom** | higher `NP` (pitched-ish) | medium decay | tone layer sets the tom pitch | melodic toms via tone |

Implementation notes:

- **Envelope re-trigger per hit:** writing R13 on the onset frame restarts the decay shape —
  this is how a drum "strikes" cleanly each time without sub-frame writes.
- **Hosting channel:** noise is routed through one tone channel via the mixer (commonly C);
  amplitude uses that channel's envelope bit so the hardware envelope shapes the hit.
- **Consistency:** identical drum kinds use identical recipes frame-to-frame, so a hi-hat
  pattern sounds *even*, not random — directly serving "solid, consistent percussion."

### Collision policy (single noise generator)

When multiple hits land on the same frame:

1. **Priority:** kick > snare > tom > hat. The winner owns the noise generator that frame.
2. **Assist:** a losing hit may be approximated on a **free tone channel** (e.g., a short
   high tone "tick" for a hat under a kick).
3. **Borrowing budget:** percussion may transiently borrow a tone channel only if doing so
   doesn't drop a high-salience sustained note; otherwise the lower-priority hit is dropped.

### Percussion vs. melody balance
- A configurable policy decides how aggressively drums may borrow tone channels. Dense drum
  tracks (e.g. `samples/short/03_drum_loop.wav`) lean percussion; melodic tracks protect tones.

---

## 7.3 Full chip utilisation

"Fully utilise AY features: three tone channels, noise generator, envelope control."

- **Three tone channels:** always offered to the arrangement; idle channels are filled with
  the next-most-salient harmony or used to assist percussion rather than left silent (subject to
  not introducing clutter — silence is preferred over wrong notes).
- **Noise generator:** primary percussion engine; also usable for breath/air textures on
  sustained pads when energy in high bands is present and no drums need it.
- **Envelope generator:** three roles —
  1. **Percussion decay** (above).
  2. **"Buzzer" timbres:** repeating envelope shapes (`0x08/0x0A/0x0C/0x0E`) tuned so the
     envelope frequency relates to a tone create richer, reedier sustained sounds than a bare
     square — a classic AY trick for fuller bass/lead.
  3. **Swells/accents** on long notes.
- **Constraint honesty:** the envelope is a **single shared** resource. The arrangement tracks a
  global envelope state and only assigns it where it adds the most value that frame (typically a
  drum hit or one feature voice), avoiding contention artifacts.

---

## 7.4 Pitch fidelity & the high-frequency problem

From §[02](02-ay-3-8910-reference.md#pitch-resolution--quantisation-error): high notes quantise
badly (e.g. ~+27 cents at A7). Strategy:

- **Sweet-spot bias:** prefer placing leads in the chip's accurate mid register.
- **Octave folding:** very high fundamentals are folded down an octave (or two) when their exact
  pitch can't be represented, preserving melodic contour over absolute register.
- **Cents-aware quantisation:** when two periods bracket a target, choose the one minimising
  *cents* error, not Hz error (perceptually correct).
- **No micro-tuning hacks** that would require sub-frame writes — we accept the grid and choose
  wisely on it.

---

## 7.5 Loudness & timbre within a logarithmic 4-bit DAC

- Map **perceptual loudness** (dB/LUFS) to DAC levels via the **measured table**, not linearly
  (§[02](02-ay-3-8910-reference.md#28-dac-volume--amplitude-table)).
- **Compress dynamics upstream** into the usable band (mostly levels 8–15) so quiet passages
  remain audible and loud passages don't all pin to 15. Legal — it shapes registers.
- **Timbre is fixed** (square + noise); we do not chase spectral matching. Perceived richness
  comes from *arrangement* (which 3 notes, how voiced) and *envelope* use, not from filtering.

---

## 7.6 What we explicitly will NOT do

- ❌ No EQ, reverb, chorus, compression, or de-noise on the rendered audio.
- ❌ No digidrums / sample playback (needs >50 Hz writes; out of hardware-faithful 50 Hz scope).
- ❌ No "fake" sub-frame register motion.
- ❌ No spectral-matching synthesis that produces register states a real chip can't hold.

Every quality gain is a *better register stream*, verifiable by playing the `.ym` on real
hardware or an independent emulator.

---

## 7.7 Tuning methodology

Quality is achieved by **measure → adjust → re-measure**, never by ear alone:

1. Convert the curated `samples/` set with a given `RunConfig`.
2. Render previews via the trusted emulator (Milestone 1).
3. Inspect debug artefacts (piano-roll vs chosen voices; jitter metrics; onset alignment).
4. Adjust smoothing/allocation/percussion parameters; re-run.
5. Lock good presets as named profiles (`--profile quality`).

Objective metrics (jitter rate, onset alignment error, melody-recall against a reference
transcription where available) are tracked in CI to prevent regressions
(§[10](10-testing-validation.md)).
