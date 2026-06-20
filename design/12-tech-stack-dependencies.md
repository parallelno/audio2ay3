# 12 — Tech Stack & Dependencies

## 12.1 Language & environment

| Item | Choice | Notes |
|------|--------|-------|
| Language | **Python 3.12** | per brief |
| Environment | **`.venv`** (project-local virtualenv) | per brief; never install globally |
| Packaging | `pyproject.toml` (PEP 621) + `src/` layout | clean imports, modern tooling |
| Build backend | `hatchling` or `setuptools` | either; hatchling preferred for simplicity |

Environment bootstrap (documented in README during implementation):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"        # editable install with dev extras
```

## 12.2 Dependency groups

Dependencies are grouped so a **CPU-only / minimal** install works, with heavier ML extras
opt-in. GPU is an accelerator, never a hard requirement
(§[09](09-performance-acceleration.md#92-gpu--cuda-acceleration)).

### Core (always)

| Package | Role |
|---------|------|
| `numpy` | arrays, vectorised DSP, register buffers |
| `scipy` | filters (decimation FIR), signal utilities |
| `soundfile` (libsndfile) | WAV/FLAC/OGG read/write |
| `ffmpeg` (system) or `imageio-ffmpeg` | decode MP3/varied containers; MP3 encode |
| `lameenc` *or* ffmpeg | MP3 encode (pick one; lameenc = pure binding, no subprocess) |
| `numba` | JIT the emulator's sequential inner loop |

### Analysis (default conversion path)

| Package | Role |
|---------|------|
| `librosa` | HPSS, CQT, onsets, spectral features (reference implementations) |
| `torch`, `torchaudio` | GPU STFT/CQT/salience; device-agnostic tensors |

### Optional / heavy (opt-in extras)

| Package | Role | Extra |
|---------|------|-------|
| `demucs` | HT-Demucs NN stem separation (**Meta**) | `[demucs]` |
| `basic-pitch` | polyphonic note transcription (**Spotify** research); recommended neural default — light (ONNX/TFLite) | `[transcribe]` |
| `t5x` + `jax` + `note-seq` | **MT3** multi-instrument transcription with instrument labels (**Google Magenta**); heaviest | `[mt3]` |
| `magenta` | **Onsets & Frames** note-onset transcription (**Google Magenta**) | `[magenta]` |
| `crepe` | monophonic pitch | `[neural]` |
| `spleeter` | alternative stems (TF-based; isolate) | `[spleeter]` |

### YM / chip support

| Need | Plan |
|------|------|
| YM read/write | **first-party** code (`ymformat/`) — full control, no obscure dep |
| LHA depack | small dependency (e.g. an `lhafile`-style lib) **or** first-party `-lh5-` decoder; v1 writes uncompressed YM so packing is read-only at first |
| Reference oracle (tests) | external **ST-Sound** (`ym2mp3`) and/or **MAME** `ay8910` for A/B validation only — not runtime deps |

### Dev / tooling

| Package | Role |
|---------|------|
| `pytest`, `pytest-xdist` | tests, parallel |
| `ruff` | lint + format |
| `mypy` | static types on `src/` |
| `rich` *(optional)* | nicer CLI progress/logging |
| `typer` *(optional)* | ergonomic CLI; else stdlib `argparse` (default, zero-dep) |

## 12.3 Why these choices

- **librosa + torch/torchaudio** cover the DSP/ML front-end with both a trusted CPU reference
  (librosa) and a GPU-capable path (torch). Algorithms can be cross-checked between them.
- **numba for the emulator**: the chip's counter/LFSR loop is inherently sequential; numba gives
  C-like speed without leaving Python or adding a build step. (A future Cython/Rust core is
  possible but not justified initially.)
- **First-party YM I/O**: the format is small and central; owning it avoids fragile third-party
  deps and guarantees we emit clean, legal registers.
- **ffmpeg/soundfile/lameenc**: robust, ubiquitous audio I/O across the required formats
  (mp3/wav/flac/ogg).
- **Neural transcription for fidelity**: Spotify **Basic Pitch**, Google Magenta **MT3** and
  **Onsets & Frames** extract musical information (notes, onsets, and — for MT3 — instrument
  identity) far better than classical DSP on dense polyphony. They are opt-in because of weight
  and weaker determinism; the DSP path stays the reproducible default and the neural path is the
  recommended `--profile quality`.
- **Minimal default footprint**: the base + analysis extras run on CPU; demucs / transcribe /
  mt3 / magenta / spleeter are isolated behind extras so the common case stays lean and
  installable.

## 12.4 `pyproject.toml` sketch

```toml
[project]
name = "audio2ay3"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "numpy", "scipy", "soundfile", "numba", "lameenc",
  "librosa", "torch", "torchaudio",
]

[project.optional-dependencies]
demucs     = ["demucs"]                  # HT-Demucs stem separation (Meta)
transcribe = ["basic-pitch"]             # Spotify polyphonic transcription (light, recommended)
mt3        = ["t5x", "jax", "note-seq"]  # Google Magenta MT3 multi-instrument (heavy)
magenta    = ["magenta"]                 # Google Magenta Onsets & Frames
neural     = ["crepe"]                   # monophonic pitch
spleeter   = ["spleeter"]                # alternative stems (TensorFlow)
dev = ["pytest", "pytest-xdist", "ruff", "mypy"]

[project.scripts]
audio2ay3 = "audio2ay3.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## 12.5 Platform & GPU notes

- **CUDA:** install the CUDA-enabled `torch` build matching the local driver; the pipeline
  auto-detects and falls back to CPU (`--no-gpu` to force).
- **ffmpeg:** required on PATH for broad MP3/container support; `imageio-ffmpeg` can vendor a
  binary if a system install isn't available.
- **Windows-first** (the dev environment) but no OS-specific code; paths via `pathlib`, processes
  via `subprocess` with explicit args.
- **MT3 is Linux/WSL-only.** Not a JAX limitation — JAX ships native Windows x86_64 CPU wheels.
  The blocker is `tensorflow-text` (pulled by `t5`/`seqio`), which has **no `win_amd64` wheel and
  no sdist** (Linux + macOS-ARM only), so the `[mt3]` stack cannot `pip install` on native Windows.
  Run `--transcription mt3` under WSL2/Linux (CPU is fine). The default Basic Pitch path is
  cross-platform (ONNX) and unaffected.
- **Determinism:** set seeds for `numpy`/`torch`; pin algorithm choices via `RunConfig` so runs
  reproduce across machines (modulo GPU non-determinism, which is confined to optional NN stages).

## 12.6 Dependency risk register

| Risk | Mitigation |
|------|-----------|
| `torch` install size/CUDA mismatch | core path works without torch on CPU via librosa; document CUDA install |
| `spleeter` drags TensorFlow | isolate behind `[spleeter]` extra; not default |
| `mt3` is heavy (JAX/T5X + model weights) | isolate behind `[mt3]`; offer **Basic Pitch** as the light neural default; both opt-in |
| `mt3` cannot install on native Windows (`tensorflow-text` has no Windows wheel/sdist) | document WSL2/Linux requirement; keep cross-platform Basic Pitch (ONNX) as the default; install MT3 via `pip install "mt3 @ git+https://github.com/magenta/mt3"` |
| NN transcription weakens determinism | keep DSP note-source as the reproducible CI default; pin model weights/seeds; confine neural to non-CI quality profiles |
| LHA packing edge cases | write uncompressed YM by default; packing optional/read-mostly |
| ffmpeg absence | fall back to `imageio-ffmpeg`; clear error if a codec is missing |
| Reference oracle availability (tests) | tests that need ST-Sound/MAME are marked and skipped if absent in CI, with at least synthetic checks always running |
