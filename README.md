# audio2ay3

Convert **instrumental audio** into a **50 Hz AY-3-8910 register stream** (`.ym`), and emulate
YM register streams back to audio.

See [design/](design/) for the full design and implementation plan.

## Status

| Capability | Command | State |
|------------|---------|-------|
| AY-3-8910 / YM2149 emulator | — | implemented |
| YM read/write (YM2/3/3b/5/6, **transparent LHA depack**) | — | implemented |
| Render a `.ym` to WAV/MP3 | `validate` | implemented |
| Audio → `.ym` (neural analysis → arrangement → legal registers) | `convert` | end-to-end skeleton |
| Audio → emulated WAV/MP3 | `preview` | end-to-end skeleton |

The `convert`/`preview` analysis stack is **neural** (Demucs separation + Basic Pitch
transcription). Deeper polyphony/percussion analysis and arrangement quality tuning are the next
phases — see [What's not done yet](#whats-not-done-yet).

## Requirements

- **Core (emulator, `validate`, tests):** Python 3.11 or 3.12.
- **Neural converter (`convert`/`preview`, the `neural` extra):**
  - **Python 3.11 — recommended.** `pip install -e ".[neural]"` resolves cleanly.
  - **Python 3.12 — works, with one extra step.** Basic Pitch 0.4.0 declares a TensorFlow
    dependency on Python ≥ 3.11 (`tensorflow>=2.4.1,<2.15.1`), but TF only ships 3.12 wheels from
    2.16+, so that pin is unsatisfiable on 3.12 and the plain install fails. We never run
    TensorFlow (the transcriber uses the **ONNX** model), so the fix is to install Basic Pitch
    **without its dependencies** — see [Python 3.12: installing the neural extra](#python-312-installing-the-neural-extra).
- Git.
- ~2–6 GB free disk **if installing the `neural` extra** (pulls PyTorch + a transcription model
  runtime). The first `convert` with Demucs downloads model weights to a local cache.
- GPU is optional. Neural models fall back to CPU automatically (or force it with `--no-gpu`).

## Installation

### 1. Clone and create a virtual environment

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv      # 3.11 = cleanest for the neural converter (3.12 also works — see below)
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

> If activation is blocked, run once per session:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

**Linux / macOS (bash):**

```bash
git clone <your-repo-url> audio2ay3
cd audio2ay3
python3.11 -m venv .venv     # 3.11 = cleanest for the neural converter (3.12 also works — see below)
source .venv/bin/activate
python -m pip install -U pip
```

### 2. Pick what to install

Editable install with extras. Combine extras in one bracket, e.g. `".[dev,mp3]"`.

| Goal | Command |
|------|---------|
| Emulator + `validate` + tests (no heavy deps) | `pip install -e ".[dev]"` |
| …plus MP3 output | `pip install -e ".[dev,mp3]"` |
| …plus reading non-WAV audio (FLAC/OGG; needed by `convert`) | `pip install -e ".[dev,mp3,audio]"` |
| **Full converter** (neural `convert`/`preview`) | `pip install -e ".[dev,mp3,neural]"` |

The `neural` extra installs Demucs (PyTorch) for separation and Basic Pitch (plus the
**ONNX** runtime) for transcription, plus `soundfile` for decoding input audio. On **Python 3.11**
the command above resolves as-is; on **Python 3.12** use the workaround just below.

> **Note (TensorFlow):** Basic Pitch pulls in TensorFlow on Python ≥ 3.11, and installing it
> downgrades `numpy` to the 1.26.x line — that's expected and numba/the emulator are fine with it.
> This project prefers the **ONNX** model for transcription and never runs TF inference, so a slow
> or fragile TF install does not affect conversion.

### Python 3.12: installing the neural extra

Basic Pitch's TensorFlow pin can't be satisfied on 3.12, but it isn't needed at runtime (we use
ONNX). Install Basic Pitch with `--no-deps` and supply its real runtime deps yourself:

```powershell
pip install -e ".[dev,mp3,audio]"          # core + MP3 + audio decoding
pip install demucs onnxruntime             # separation + the ONNX runtime we actually use
pip install --no-deps basic-pitch          # skip its unsatisfiable tensorflow marker
# Basic Pitch's genuine runtime deps (everything except tensorflow):
pip install "librosa>=0.8.0" "mir_eval>=0.6.0" "pretty_midi>=0.2.9" `
            "resampy>=0.2.2,<0.4.3" scikit-learn "scipy>=1.4.1" typing_extensions
```

Then verify the transcriber imports and selects the ONNX model:

```powershell
python -c "from audio2ay3.analysis.transcribe import _basic_pitch_model_path as p; print(p().name)"
# -> nmp.onnx
```

### 3. Verify the install

```powershell
pytest                 # full unit suite should be green
audio2ay3 --help       # CLI is wired
```

## Usage

### Validate — render a YM register dump to audio (no neural deps)

```powershell
audio2ay3 validate samples\ym\song01.ym -o build\song01.mp3   # MP3 needs the [mp3] extra
audio2ay3 validate samples\ym\song01.ym -o build\song01.wav --wav
```

LHA-packed `.ym` files (the common `-lh5-` form) are depacked transparently — no external tool
needed.

### Convert — audio → `.ym` register stream (needs the `neural` + `audio` extras)

```powershell
# Already-instrumental input: skip Demucs separation for speed
audio2ay3 convert samples\short\trumpet.ogg -o build\trumpet.ym --separation none

# Full song: let Demucs strip vocals first (default --separation demucs)
audio2ay3 convert samples\long\Goblins_Lair.mp3 -o build\goblins.ym
```

### Preview — audio → emulated audio (convert, then render through the emulator)

```powershell
audio2ay3 preview samples\short\trumpet.ogg -o build\trumpet.mp3 --separation none
```

### Useful options (`convert` / `preview`)

| Option | Meaning |
|--------|---------|
| `--separation {demucs,none}` | Neural vocal/stem separation (default `demucs`; `none` for instrumental input). |
| `--transcription {basic-pitch,mt3,onsets-frames}` | Transcription backend (default `basic-pitch`). |
| `--clock HZ` | Master clock (default 1.7734 MHz, ZX Spectrum). |
| `--frame-rate HZ` | Replay frame rate (default 50). |
| `--no-gpu` | Force CPU for the neural models. |
| `--duration SEC` | Limit rendered seconds (`preview`/`validate`). |

## Tests

```powershell
pytest            # unit suite (emulator, YM I/O, encode, mapping, pipeline arrange)
ruff check src tests
```

The deterministic core (`arrange` and everything below it) is fully unit-tested without any
neural dependency; only the live `convert`/`preview` front-end needs the `neural` extra.

## What's not done yet

Phases 0–3 of the [roadmap](design/13-implementation-roadmap.md) are implemented: the trusted
emulator, complete YM I/O, and a **thin end-to-end** `convert`/`preview` that produces
hardware-legal `.ym` files following the input's melody and hits. Still to come:

- **Phase 4 — analysis depth:** real multi-F0 polyphony and drum kind classification
  (kick/snare/hat). Today `convert` uses Basic Pitch notes; percussion events are not yet
  produced by the transcriber, so drum mapping is wired but unfed.
- **Phase 5 — arrangement & sound quality:** smarter 3-voice allocation (continuity and
  bass/salience bias are basic today), smoothing/anti-jitter, envelope/buzzer timbres,
  octave-folding sweet spots, and named quality profiles. This is where fidelity is won.
- **Phase 6 — performance:** GPU spectral path, parallel/segmented analysis, chunked emulation.
- **Phase 7 — scalability:** `--frame-rate 100` and `--chips 2` (dual-AY, 6 voices).
- **Phase 8 — hardening:** edge-case inputs, richer CLI UX, perceptual regression metrics in CI.
- **Alternate backends:** `--transcription mt3 / onsets-frames` and `--separation spleeter` are
  recognised but not wired (they raise a clear "not available" error).

The emulator and `validate` path (Milestone 1) are stable and are the ground truth everything
else is measured against.
