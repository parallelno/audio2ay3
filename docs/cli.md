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
                  [--clock HZ] [--frame-rate HZ] [--chips N] [--no-gpu] [--seed N]
                  [--no-amp-envelope] [--explain]
```

### Analysis options

| Option | Default | Meaning |
|--------|---------|---------|
| `input` | — | Input audio (WAV/FLAC/OGG; MP3 if your libsndfile build supports it). |
| `-o`, `--output` | `build/<name>.ym` | Output `.ym` path. |
| `--separation` | `demucs` | Source-separation backend. `none` skips separation — use it for already-instrumental input. `spleeter` is recognised but not implemented (raises a clear error). |
| `--transcription` | `basic-pitch` | Transcription backend. `mt3` is the heavyweight multi-instrument path (Linux/WSL only). `yourmt3` is an **experimental** pure-PyTorch multi-instrument backend that installs on native Windows, but transcribed more sparsely than `basic-pitch` in testing and is slow — not recommended (GPL-3.0 model installed separately; see the README). `onsets-frames` is reserved and raises a clear error. |
| `--clock HZ` | `1773400` | AY master clock (default ≈ 1.7734 MHz, ZX Spectrum). |
| `--frame-rate HZ` | `50` | Replay frame rate. |
| `--chips N` | `1` | Number of AY chips. **Experimental:** accepted and stored, but dual-AY arrangement is not yet implemented, so only the first chip is rendered today. |
| `--no-gpu` | off | Force CPU for the neural models (they otherwise auto-detect). |
| `--seed N` | `0` | Deterministic seed for the neural stages. |

### Arrangement options

| Option | Default | Meaning |
|--------|---------|---------|
| `--no-amp-envelope` | off | Disable per-note amplitude shaping; notes become flat, constant-volume blocks. By default each note follows its source loudness with a struck attack/decay. |
| `--vibrato` | off | Add a small pitch-LFO vibrato to expressive voices (organ, strings, reed, pipe, synth lead). |
| `--breath` | off | Add a short breathy noise chiff at the attack of wind voices (reeds/pipes). |
| `--arpeggio` | off | When more notes sound than there are free channels, cycle the squeezed chord tones on one channel instead of dropping them (the classic chiptune arpeggio). |
| `--explain` | off | After writing the `.ym`, print register-level diagnostics for the arranged song plus a voice-contention report (how many notes were dropped for lack of channels, and an estimate of what a second AY would recover). See [Reading `--explain`](#reading---explain). |

> `--vibrato`, `--breath`, and `--arpeggio` are opt-in timbre features (off by default). They were
> found to hurt some material, so enable them per run when the source benefits.

**Examples**

```powershell
# Already-instrumental input: skip Demucs for speed
audio2ay3 convert samples\short\trumpet.ogg -o build\trumpet.ym --separation none

# Full song: Demucs strips vocals first (default)
audio2ay3 convert samples\long\Goblins_Lair.mp3 -o build\goblins.ym

# Flat, constant-volume notes plus the diagnostics dump
audio2ay3 convert samples\long\Goblins_Lair.mp3 --no-amp-envelope --explain

# Opt in to the expressive timbre features
audio2ay3 convert samples\long\Goblins_Lair.mp3 --vibrato --breath --arpeggio

# MT3 multi-instrument backend (Linux/WSL, AUDIO2AY3_MT3_CHECKPOINT must be set)
python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-mt3.ym --transcription mt3

# YourMT3+ multi-instrument backend (native Windows OK). Run `audio2ay3 setup-yourmt3` once to
# fetch the GPL-3.0 model into a per-user cache; then no env vars are needed.
python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-ymt3.ym --transcription yourmt3
```

---

## `setup-yourmt3`

Fetch the optional **YourMT3+** transcription backend (GPL-3.0) into a per-user cache directory so
`--transcription yourmt3` works without manual cloning or env vars. Needs `git` (and `git-lfs` for
the checkpoints); it does **not** import torch. The GPL model code is fetched onto your machine at
runtime and is never bundled into this MIT project.

> **Experimental, not recommended.** The `yourmt3` backend transcribed more sparsely than
> `basic-pitch` in testing and is slow; this helper is kept for experimentation only.

```
audio2ay3 setup-yourmt3 [--dir PATH] [--repo-url URL] [--model NAME] [--force]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--dir PATH` | per-user cache | Checkout location. `AUDIO2AY3_YOURMT3_DIR` overrides the default at convert time. |
| `--repo-url URL` | YourMT3 HuggingFace Space | Clone URL. The Space colocates `model_helper.py` + `amt/` and carries the LFS checkpoints. Use this to point at a mirror/fork. |
| `--model NAME` | `YPTF.MoE+Multi (noPS)` | Model variant whose checkpoint presence is verified after cloning. |
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
                  [--separation ...] [--transcription ...] [--clock HZ] [--frame-rate HZ]
                  [--chips N] [--no-gpu] [--seed N] [--no-amp-envelope] [--explain]
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

## Reading `--explain`

With `--explain`, `convert`/`preview` print two blocks after the `ok:` line:

1. **Song stats** — register-level facts about the arranged `.ym`: a polyphony histogram (how many
   of the audible tone voices sound per frame), tone-on percentage per channel A/B/C, noise frame
   count, the number of distinct bass tone periods on channel A, amplitude-change activity per
   channel, and the count of distinct amplitude levels used.
2. **Voice contention** — how the deterministic allocator coped with the transcription: how many
   melodic notes were silenced, note-frames demanded vs sounded, how many were dropped for lack of
   a free channel vs. lost to a drum hit on the shared channel, plus an estimate of what a second
   AY chip would recover.

These are diagnostics, not warnings — they help explain why a dense passage loses notes (a 3-voice
chip can only sound three tones at once) and quantify how much a dual-AY setup would help.

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
- The first Demucs run downloads model weights to a local cache; later runs are faster.
- For a fast turnaround while tuning, combine `preview --wav --duration 15` to render a short clip
  without MP3 encoding.
- `--explain` is the quickest way to see whether missing notes are a transcription problem or a
  channel-contention problem.
