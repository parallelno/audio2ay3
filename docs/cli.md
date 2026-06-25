# audio2ay3 CLI reference

How to use the `audio2ay3` command-line tooling: the three subcommands, every option, exit
codes, and worked examples.

For installation (including the Python 3.12 and MT3/WSL notes) see the project
[README](../README.md#installation). This document assumes the package is already installed in
your active environment.

## Invoking the tool

Two equivalent entry points:

```powershell
audio2ay3 <command> [options]        # console script (installed by pip)
python -m audio2ay3 <command> [options]   # module form (works even when the script isn't on PATH)
```

Get help at any level:

```powershell
audio2ay3 --help
audio2ay3 convert --help
```

There are three commands:

| Command | Purpose | Needs neural deps? |
|---------|---------|--------------------|
| [`validate`](#validate) | Render an existing `.ym` register dump to WAV/MP3 through the emulator. | No |
| [`convert`](#convert) | Convert instrumental audio into a hardware-legal `.ym` register stream. | Yes |
| [`preview`](#preview) | `convert`, then render the result back to audio in one step. | Yes |

`validate` is the ground-truth path and depends only on the core install. `convert` and `preview`
drive the neural analysis front-end (separation + transcription) and need the `neural` (and
`audio`) extras.

## Output paths

If you omit `-o`/`--output`, the tool writes to `build/<input-stem><ext>`, creating the `build/`
directory if needed. For example, `convert samples\long\Goblins_Lair.mp3` writes
`build\Goblins_Lair.ym`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `2` | Argument parsing error (bad/missing options — argparse). |
| `3` | Runtime error: input file not found, unreadable YM, a missing neural backend, or an unsupported option. The message is printed to stderr. |

---

## `validate`

Render a `.ym` file to audio through the emulator. This is the trusted reproduction path and the
reference everything else is measured against. No neural dependencies.

```
audio2ay3 validate <input.ym> [-o OUT] [--wav] [--sr N] [--oversample N]
                              [--bitrate N] [--duration SEC] [--clock HZ] [--frame-rate HZ]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `input` | — | Input `.ym` file. YM2/3/3b/5/6 are supported; LHA-packed (`-lh5-`) files are depacked transparently. |
| `-o`, `--output` | `build/<name>.mp3` | Output audio file. |
| `--wav` | off | Write WAV instead of MP3. (MP3 output needs the `[mp3]` extra.) |
| `--sr N` | `44100` | Render sample rate (Hz). |
| `--oversample N` | `2` | Anti-alias oversample factor before decimation. |
| `--bitrate N` | `192` | MP3 bitrate (kbps); ignored for WAV. |
| `--duration SEC` | full song | Limit the rendered length in seconds. |
| `--clock HZ` | from file | Override the master clock. |
| `--frame-rate HZ` | from file | Override the replay frame rate. |

On success it prints the YM version, clock, frame rate, frame count, duration, and output path.

**Examples**

```powershell
# MP3 (needs the [mp3] extra)
audio2ay3 validate samples\ym\song01.ym -o build\song01.mp3

# WAV, no MP3 encoder required
audio2ay3 validate samples\ym\song01.ym --wav

# First 10 seconds only, at 48 kHz
audio2ay3 validate samples\ym\song01.ym --wav --sr 48000 --duration 10
```

---

## `convert`

Run the neural pipeline (source separation → transcription → arrangement) and write a
hardware-legal `.ym` register stream. Needs the `neural` and `audio` extras.

```
audio2ay3 convert <input-audio> [-o OUT]
                  [--separation {demucs,spleeter,none}] [--transcription {basic-pitch,mt3,yourmt3,onsets-frames}]
                  [--yourmt3-model NAME] [--stems-dir DIR]
                  [--clock HZ] [--frame-rate HZ] [--chips N] [--no-gpu] [--no-progress] [--seed N]
                  [--no-amp-envelope] [--explain]
```

### Analysis options

| Option | Default | Meaning |
|--------|---------|---------|
| `input` | — | Input audio (WAV/FLAC/OGG; MP3 if your libsndfile build supports it; M4A/AAC via `ffmpeg` on PATH). |
| `-o`, `--output` | `build/<name>.ym` | Output `.ym` path. |
| `--separation` | `demucs` | Source-separation backend. `demucs` is `htdemucs` (4-stem). `demucs-ft` is the fine-tuned `htdemucs_ft` — better separation, ~4× slower. `demucs6` is the 6-stem `htdemucs_6s` (adds guitar/piano; **experimental**). `none` skips separation — use it for already-instrumental input. `spleeter` is recognised but not implemented (raises a clear error). |
| `--transcription` | `basic-pitch` | Transcription backend. `mt3` is the heavyweight multi-instrument path (Linux/WSL only). `yourmt3` is an optional, opt-in pure-PyTorch multi-instrument backend that installs on native Windows (GPL-3.0 model installed separately; see the README). It defaults to the `YMT3+` variant, which did best in testing; the heavier MoE variants transcribed more sparsely. `onsets-frames` is reserved and raises a clear error. |
| `--yourmt3-model NAME` | `YMT3+` | Only with `--transcription yourmt3`: pick the YourMT3 variant. Choices: `YMT3+` (the default; did best in testing — recovered notes the MoE variants missed, and was faster), `YPTF.MoE+Multi (noPS)`, `YPTF.MoE+Multi (PS)`, `YPTF+Multi (PS)`. When omitted, the value falls back to the `AUDIO2AY3_YOURMT3_MODEL` env var and then the `YMT3+` default. |
| `--clock HZ` | `1773400` | AY master clock (default ≈ 1.7734 MHz, ZX Spectrum). |
| `--frame-rate HZ` | `50` | Replay frame rate. |
| `--chips N` | `1` | Number of AY chips: `1` (3 tone channels) or `2` (dual-AY, 6 tone channels). With two chips the melody spreads across four channels instead of one or two, bass keeps its own channel, and percussion is isolated on the second chip. `convert` writes chip 0 to the named `.ym` and chip 1 to `<name>.ay2.ym` (the YM format has no standard dual-PSG container); `preview` renders both chips into one audio file. |
| `--no-gpu` | off | Force CPU for the neural models (they otherwise auto-detect). |
| `--no-progress` | off | Disable the per-stage progress bar (also auto-suppressed when stderr is not a terminal, e.g. when piped or redirected). |
| `--seed N` | `0` | Deterministic seed for the neural stages. |
| `--stems-dir DIR` | — | Load pre-separated stems from `<DIR>/<song-name>/` instead of running Demucs. The folder must contain `<song-name> (Synth).<ext>` (mandatory), and optionally `(Bass)`, `(Drums)`, and `(FX)` files. When found, Demucs is skipped entirely and `--separation` is ignored. When `<DIR>/<song-name>/` does not exist, the pipeline falls back to Demucs normally. The folder name must match the audio filename stem exactly. Accepts MP3, WAV, FLAC, OGG, M4A. |

### Arrangement options

| Option | Default | Meaning |
|--------|---------|---------|
| `--no-amp-envelope` | off | Disable per-note amplitude shaping; notes become flat, constant-volume blocks. By default each note follows its source loudness with a struck attack/decay. |
| `--vibrato` | off | Add a small pitch-LFO vibrato to expressive voices (organ, strings, reed, pipe, synth lead). |
| `--breath` | off | Add a short breathy noise chiff at the attack of wind voices (reeds/pipes). |
| `--arpeggio` | off | When more notes sound than there are free channels, cycle the squeezed chord tones on one channel instead of dropping them (the classic chiptune arpeggio). |
| `--noise-volume SCALE` | `1.0` | Noise channel volume as a linear scale. `1.0` = full (default), `0.5` = half as loud, `0.0` = muted. Applied uniformly to every amplitude frame the percussion renderer writes, so it scales the noise channel without touching the melodic tone channels. |
| `--explain` | off | After writing the `.ym`, print register-level diagnostics for the arranged song plus a voice-contention report (how many notes were dropped for lack of channels, and an estimate of what a second AY would recover). See [Reading `--explain`](#reading---explain). |

> `--vibrato`, `--breath`, and `--arpeggio` are opt-in timbre features (off by default). They were
> found to hurt some material, so enable them per run when the source benefits.

**Examples**

```powershell
# Already-instrumental input: skip Demucs for speed
audio2ay3 convert samples\short\trumpet.ogg -o build\trumpet.ym --separation none

# Full song: Demucs strips vocals first (default)
audio2ay3 convert samples\long\Goblins_Lair.mp3 -o build\goblins.ym

# Use perfect pre-separated stems — no Demucs pass at all
# samples\stems\Goblins_Lair\ must contain "Goblins_Lair (Synth).mp3" etc.
audio2ay3 convert samples\long\Goblins_Lair.mp3 --stems-dir samples\stems

# Flat, constant-volume notes plus the diagnostics dump
audio2ay3 convert samples\long\Goblins_Lair.mp3 --no-amp-envelope --explain

# Opt in to the expressive timbre features
audio2ay3 convert samples\long\Goblins_Lair.mp3 --vibrato --breath --arpeggio

# MT3 multi-instrument backend (Linux/WSL, AUDIO2AY3_MT3_CHECKPOINT must be set)
python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-mt3.ym --transcription mt3

# YourMT3+ multi-instrument backend (native Windows OK). Run `audio2ay3 setup-yourmt3` once to
# fetch the GPL-3.0 model into a per-user cache; then no env vars are needed. Defaults to the
# YMT3+ variant, which did best in testing:
python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-ymt3.ym --transcription yourmt3
```

---

## `setup-yourmt3`

Fetch the optional **YourMT3+** transcription backend (GPL-3.0) into a per-user cache directory so
`--transcription yourmt3` works without manual cloning or env vars. Needs `git` (and `git-lfs` for
the checkpoints); it does **not** import torch. The GPL model code is fetched onto your machine at
runtime and is never bundled into this MIT project.

> **Optional, opt-in.** The `yourmt3` backend defaults to the `YMT3+` variant, which did best in
> testing (the heavier MoE variants transcribed more sparsely). `setup-yourmt3` verifies `YMT3+`'s
> checkpoint by default; pick another with `--model` and select it at convert time with
> `--yourmt3-model`.

```
audio2ay3 setup-yourmt3 [--dir PATH] [--repo-url URL] [--model NAME] [--force]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--dir PATH` | per-user cache | Checkout location. `AUDIO2AY3_YOURMT3_DIR` overrides the default at convert time. |
| `--repo-url URL` | YourMT3 HuggingFace Space | Clone URL. The Space colocates `model_helper.py` + `amt/` and carries the LFS checkpoints. Use this to point at a mirror/fork. |
| `--model NAME` | `YMT3+` | Model variant whose checkpoint presence is verified after cloning. Choices: `YMT3+` (default; did best in testing), `YPTF.MoE+Multi (noPS)`, `YPTF.MoE+Multi (PS)`, `YPTF+Multi (PS)`. |
| `--force` | off | Update an existing checkout (`git pull`) instead of skipping. |

The command clones the repo, verifies that `model_helper.py` + `amt/src` are present, and reports
whether the chosen variant's checkpoint was found (downloading large checkpoints may require
git-lfs or a manual step it points you to). Exit code `3` on failure (e.g. `git` missing).

```powershell
audio2ay3 setup-yourmt3                      # default cache dir + recommended variant
audio2ay3 setup-yourmt3 --dir D:\models\ymt3 # custom location
audio2ay3 setup-yourmt3 --force              # update an existing checkout
```

---

## `preview`

Convert audio and immediately render the result back to audio through the emulator — handy for
quick A/B listening without a separate `validate` step. Needs the same extras as `convert`.

```
audio2ay3 preview <input-audio> [-o OUT] [--wav] [--sr N] [--oversample N] [--bitrate N]
                  [--duration SEC]
                  [--separation ...] [--transcription ...] [--yourmt3-model NAME] [--stems-dir DIR]
                  [--clock HZ] [--frame-rate HZ]
                  [--chips N] [--no-gpu] [--no-progress] [--seed N] [--no-amp-envelope] [--explain]
```

`preview` accepts every `convert` analysis/arrangement option above, plus the audio-rendering
options from `validate`:

| Option | Default | Meaning |
|--------|---------|---------|
| `-o`, `--output` | `build/<name>.mp3` | Output audio file. |
| `--wav` | off | Write WAV instead of MP3. |
| `--sr N` | `44100` | Render sample rate (Hz). |
| `--oversample N` | `2` | Anti-alias oversample factor. |
| `--bitrate N` | `192` | MP3 bitrate (kbps). |
| `--duration SEC` | full | Limit the rendered length in seconds. |

**Examples**

```powershell
audio2ay3 preview samples\short\trumpet.ogg -o build\trumpet.mp3 --separation none
audio2ay3 preview samples\long\Goblins_Lair.mp3 --wav --duration 20 --explain
```

---

## Batch conversion with pre-separated stems

When you have perfectly isolated per-instrument stems (from your DAW or any other source),
passing them via `--stems-dir` gives the transcriber a cleaner signal than Demucs can produce
and skips the slowest step in the pipeline entirely.

### Stem folder layout

For a song whose audio file is `Goblins_Lair.mp3`, create:

```
samples/stems/
  Goblins_Lair/
    Goblins_Lair (Synth).mp3    ← melody / harmony  (required)
    Goblins_Lair (Bass).mp3     ← bass               (optional)
    Goblins_Lair (Drums).mp3    ← drums              (optional)
    Goblins_Lair (FX).mp3       ← effects            (optional, mixed into Synth)
```

The **folder name must match the audio filename stem exactly** (the match is case-sensitive on
Linux/macOS). Any audio format that soundfile understands is accepted (MP3, WAV, FLAC, OGG); M4A
is decoded via `ffmpeg` (must be on PATH).

| Stem | Role when present |
|------|------------------|
| `(Synth)` | Fed to the note transcriber as the melodic/harmonic content. **Required.** |
| `(Bass)` | Transcribed separately; placed on its own dedicated AY tone channel. |
| `(Drums)` | Fed to the onset-detection stage; placed on the AY noise channel. |
| `(FX)` | Mixed into `(Synth)` before transcription, adding tonal effects as extra notes. |

### Single-song conversion

```powershell
audio2ay3 convert samples\long\Goblins_Lair.mp3 --stems-dir samples\stems --chips 2
```

### Batch dual-AY conversion

`scripts/convert_long_dual.py` accepts `--stems-dir` and `--stems-only`. With `--stems-only`
the input list is derived from the stems directory itself — `--in-dir` is not needed:

```powershell
# Convert every song that has a stems folder; ignore --in-dir entirely
python scripts/convert_long_dual.py --stems-dir samples\stems --stems-only --out-dir results\stems_dual

# Half-volume noise channel
python scripts/convert_long_dual.py --stems-dir samples\stems --stems-only --noise-volume 0.5 --out-dir results\stems_dual

# Or use the convenience batch script (has --noise-volume 0.5 set by default):
.\convert_all_stems_dual.bat
```

Without `--stems-only`, the script processes every file in `--in-dir` (default `samples/long`).
Songs that have a matching stems folder skip Demucs; songs that do not fall back to Demucs.

---

## Reading `--explain`

With `--explain`, `convert`/`preview` print two blocks after the `ok:` line:

1. **Song stats** — register-level facts about the arranged `.ym`: a polyphony histogram (how many
   of the audible tone voices sound per frame), tone-on percentage per channel A/B/C (A..F on
   dual-AY), noise frame count, the number of distinct bass tone periods on channel A,
   amplitude-change activity per channel, and the count of distinct amplitude levels used.
2. **Voice contention** — how the deterministic allocator coped with the transcription: how many
   melodic notes were silenced, note-frames demanded vs sounded, how many were dropped for lack of
   a free channel vs. lost to a drum hit on the shared channel, plus an estimate of what a second
   AY chip would recover.

These are diagnostics, not warnings — they help explain why a dense passage loses notes (a 3-voice
chip can only sound three tones at once) and quantify how much a dual-AY setup would help — which
you can then turn on with `--chips 2`.

---

## Config-only options (no CLI flag yet)

The finer parameters of the timbre features — the vibrato rate/depth, the breath noise period, and
the amplitude-envelope attack/decay/sustain — live on `RunConfig`/`Vibrato`/`AmpEnvelope` in
[`config.py`](../src/audio2ay3/config.py) and are only reachable when driving the pipeline
programmatically. The on/off switches themselves are exposed as the `--vibrato`, `--breath`,
`--arpeggio`, and `--no-amp-envelope` flags above.

Example (programmatic, customising the vibrato shape):

```python
from audio2ay3.config import RunConfig, Vibrato
from audio2ay3.pipeline import convert

cfg = RunConfig(vibrato=Vibrato(enabled=True, rate_hz=5.0, depth_cents=15.0), breath=True)
song = convert("samples/long/Goblins_Lair.mp3", cfg)
```

---

## Tips

- Use `--separation none` whenever the input is already instrumental (loops, stems, chiptune
  sources); it skips the slow Demucs pass entirely.
- Use `--stems-dir` when you have original project stems. The transcriber gets a perfectly
  isolated signal per instrument — no Demucs artifacts — which improves note detection accuracy
  for melody, bass, and drums simultaneously. The Demucs pass is skipped entirely.
- The first Demucs run downloads model weights to a local cache; later runs are faster.
- For a fast turnaround while tuning, combine `preview --wav --duration 15` to render a short clip
  without MP3 encoding.
- `--explain` is the quickest way to see whether missing notes are a transcription problem or a
  channel-contention problem.
