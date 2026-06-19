# 08 — CLI Design

A single entry point `audio2ay3` exposes three subcommands. The CLI is a thin layer over
`audio2ay3.pipeline`: it parses arguments into a `RunConfig`, runs the requested flow, and maps
results to exit codes. No business logic lives here.

## 8.1 Commands at a glance

| Command | Input | Output | Flow |
|---------|-------|--------|------|
| `validate` | `.ym` (any version, LHA-ok) | `.mp3` (or `.wav`) | YM → emulate → encode |
| `convert` | audio (mp3/wav/flac/ogg/…) | `.ym` (AY register stream) | audio → analyse → map → encode → YM |
| `preview` | audio | `.mp3` (emulated result) | audio → … → YM (in-memory) → emulate → encode |

These map directly to the brief's three required tools (AY Validator, Converter, Preview).

## 8.2 `validate` — YM playback (Milestone 1 deliverable)

Render an existing YM register dump through **our** emulator to prove accuracy / let users
listen.

```
audio2ay3 validate INPUT.ym [-o OUTPUT.mp3]
                   [--wav] [--duration SECONDS] [--loops N]
                   [--sr 44100] [--bitrate 192]
                   [--clock HZ] [--frame-rate HZ]
```

- `--clock` / `--frame-rate` override the YM header (for headerless YM3 or experimentation);
  by default the header's `masterClock`/`frameRate` are honoured.
- `--loops` / `--duration` control playback length using the YM `loopFrame`.
- Auto-detects YM2/3/3b/5/6 and LHA packing.

## 8.3 `convert` — audio → YM

```
audio2ay3 convert INPUT.(mp3|wav|flac|ogg) [-o OUTPUT.ym]
                  [--profile {quality,balanced,fast}]
                  [--clock HZ] [--frame-rate {50,100}]
                  [--chips {1,2}]
                  [--separation {hpss,demucs,spleeter,none}]
                  [--multipitch {cqt-salience,nmf,klapuri,basic-pitch,mt3,onsets-frames}]
                  [--no-gpu] [--threads N] [--seed N]
                  [--loop-frame N] [--name STR] [--author STR]
                  [--explain] [--lha]
```

- `--profile` selects a tuned `RunConfig` preset (overridable by explicit flags).
- `--explain` writes debug artefacts (piano-roll, chosen voices, register tracks) next to the
  output for fidelity inspection.
- `--lha` packs the YM (default off → plain, maximally-compatible YM6).
- `--chips 2` / `--frame-rate 100` engage scalability paths (§[11](11-scalability.md)).

## 8.4 `preview` — audio → emulated MP3

```
audio2ay3 preview INPUT.(mp3|wav|flac|ogg) [-o OUTPUT.mp3]
                  [--keep-ym PATH]            # also save the intermediate .ym
                  [--wav] [--sr 44100] [--bitrate 192]
                  # ...all convert tuning flags also accepted...
```

`preview` is `convert` + `validate` fused in memory: it runs the full conversion, emulates the
resulting register stream, and encodes audio — the fastest way to hear "how will this sound on
an AY?" without writing a file. `--keep-ym` optionally also persists the `.ym`.

## 8.5 Common conventions

- **Output defaulting:** if `-o` is omitted, derive from input stem into `build/`
  (e.g. `convert samples/long/Dungeon_Ore.mp3` → `build/Dungeon_Ore.ym`).
- **Formats:** `--wav` switches lossless output where MP3 is the default; both go through the
  same peak-safe gain (no tonal EQ — §[05](05-emulator-design.md#57-render--mp3-path)).
- **Progress & logging:** `-v/-vv` verbosity; a progress bar for long tracks; `--quiet` for
  scripts.
- **Determinism:** `--seed` fixes all stochastic steps; same input+config+seed ⇒ same output.
- **Exit codes:** `0` success; `2` bad arguments; `3` unsupported/corrupt input; `4` internal
  pipeline error. Never writes a partial artefact on failure.

## 8.6 Implementation notes

- Built with `argparse` (stdlib) or `typer`/`click` for nicer help; choice finalised in
  [12-tech-stack-dependencies.md](12-tech-stack-dependencies.md). Stdlib `argparse` keeps deps
  minimal and is the default recommendation.
- The CLI imports lazily so `--help` is instant and GPU/ML libs load only when a command needs
  them.
- Each subcommand is a function `cmd_validate/convert/preview(args) -> int` returning an exit
  code, trivially unit-testable.

## 8.7 Example session

```console
$ audio2ay3 validate tunes/classic.ym -o build/classic.mp3
✓ YM6, 1.7734 MHz, 50 Hz, 8123 frames → build/classic.mp3 (2:42)

$ audio2ay3 convert samples/short/02_bass_and_lead.wav --profile quality --explain
✓ analysed 6.0 s · 3 voices · 12 onsets
✓ wrote build/02_bass_and_lead.ym  (YM6, 300 frames, 50 Hz)
ℹ debug artefacts → build/02_bass_and_lead.explain/

$ audio2ay3 preview samples/long/Dungeon_Ore.mp3 --keep-ym build/Dungeon_Ore.ym
✓ converted + emulated → build/Dungeon_Ore.mp3
```
