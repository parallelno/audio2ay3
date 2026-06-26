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

### MT3 transcription (`--transcription mt3`) — experimental, Linux/WSL only

MT3 (Google Magenta) is a heavyweight multi-instrument transcriber: one pass yields pitched
notes, bass, and drums together with General-MIDI instrument identity, so it self-routes
percussion and the bass line (no Demucs separation needed). It is fully opt-in and never touched
by the default `convert` or the test suite.

> **Windows is not supported for MT3.** Contrary to a common assumption, **JAX is not the
> blocker** — JAX ships native Windows x86_64 CPU wheels now. The blocker is **`tensorflow-text`**
> (pulled transitively by `t5`/`seqio`), which publishes **only Linux and macOS-ARM wheels — no
> `win_amd64` wheel and no sdist**, so the stack cannot `pip install` on native Windows. Run MT3
> under **WSL2 (Ubuntu)** or Linux. CPU-only is fine.

```bash
# In WSL2/Linux, Python 3.10–3.12. MT3's own setup.py is the authoritative installer: it pulls
# t5x + flax + seqio + t5 + note-seq + tensorflow + tensorflow-datasets (and jax via flax).
pip install "mt3 @ git+https://github.com/magenta/mt3"
pip install -e .                           # audio2ay3 itself, same env

# Download a checkpoint (large) from the public GCS bucket, then point at its directory:
gsutil -m cp -r gs://mt3/checkpoints/mt3 ./mt3-checkpoint      # or the https://storage.googleapis.com/mt3/... mirror
export AUDIO2AY3_MT3_CHECKPOINT="$PWD/mt3-checkpoint"

python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-mt3.ym --transcription mt3
```

> **Version drift:** `t5x`/`jax`/`flax` break across releases. If the install or a run fails on
> version errors, match the pins in MT3's [upstream colab](https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb).

> **Missing gin configs:** the `mt3` wheel has a packaging bug — its `setup.py` omits the `gin/`
> subdirectory, so the first run fails with a gin error. audio2ay3 detects this and prints the
> exact fix; in short, drop the 7 config files into the installed package's `gin/` folder:
>
> ```bash
> MT3_GIN="$(python -c 'import mt3, pathlib; print(pathlib.Path(mt3.__file__).parent / "gin")')"
> mkdir -p "$MT3_GIN"
> for f in model.gin mt3.gin ismir2021.gin local_tiny.gin train.gin eval.gin infer.gin; do
>   curl -sSL -o "$MT3_GIN/$f" "https://raw.githubusercontent.com/magenta/mt3/main/mt3/gin/$f"
> done
> ```

### YourMT3+ transcription (`--transcription yourmt3`) — experimental, opt-in

> **Status: experimental — opt-in.** It now defaults to the lighter **`YMT3+`** variant, which did
> best in testing — it recovered melodic notes the heavier MoE variants (and basic-pitch) missed,
> and ran faster. The MoE variants (`YPTF.MoE+Multi (noPS)`, `YPTF.MoE+Multi (PS)`) transcribed
> *more sparsely* on real material (e.g. `Goblins_Lair`: ~55% pitched duty cycle, several
> multi-second melodic gaps) and were slow. Override the variant with `--yourmt3-model` if you want
> to compare. For everyday use `basic-pitch` (the native-Windows default) is still the safe choice,
> with **MT3** (Linux/WSL) for multi-instrument quality, but `YMT3+` is worth a try when notes are
> missing.

[YourMT3+](https://github.com/mimbres/YourMT3) (mimbres, MLSP&nbsp;2024) is MT3-class
multi-instrument transcription rebuilt on a **pure PyTorch** stack — `torch`, `torchaudio`,
`lightning`, `transformers`, `einops`, `mido`, `librosa` — with **no JAX / t5x / tensorflow-text**.
That single difference means it `pip install`s on **native Windows** (the wheel that blocks MT3 is
gone). One pass yields pitched notes, bass, and drums together with General-MIDI identity, so it
self-routes exactly like the MT3 path. Fully opt-in; never touched by the default `convert` or the
test suite.

> **License — read before using.** YourMT3 is **GPL-3.0**, while audio2ay3 is **MIT**. To keep
> GPL-licensed code out of this MIT tree, the model is treated as an *optional, user-installed*
> backend: **we do not vendor or bundle any YourMT3 code.** The setup helper (and the manual route)
> fetch the GPL repo onto *your* machine at runtime; our thin adapter
> (`audio2ay3/analysis/_yourmt3_infer.py`) imports it at runtime. Distributing your own combined
> build is your responsibility under the GPL.

```powershell
# Native Windows (or Linux/WSL), Python 3.11/3.12. The pip stack is pure PyTorch:
pip install -e ".[yourmt3]"                       # torch/torchaudio/lightning/transformers/...

# Easy path: clone + wire up the GPL backend into a per-user cache (needs git + git-lfs):
python -m audio2ay3 setup-yourmt3
# ...then just run it (no env vars needed — the cache dir is auto-detected). Defaults to YMT3+:
python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-ymt3.ym --transcription yourmt3
```

`setup-yourmt3` clones the YourMT3 **HuggingFace Space** (it colocates `model_helper.py` + `amt/`
and carries the LFS checkpoints) into a per-user cache, verifies the layout, and reports if a
checkpoint still needs downloading. Flags: `--dir` (checkout location), `--repo-url` (mirror/fork),
`--model` (variant to verify, default `YMT3+`), `--force` (`git pull` an existing
checkout). It only needs `git`/`git-lfs` — no torch. To select a variant at convert time use
`--yourmt3-model` (e.g. `--yourmt3-model "YPTF.MoE+Multi (noPS)"`) or the `AUDIO2AY3_YOURMT3_MODEL`
env var.

<details><summary>Manual setup (alternative to <code>setup-yourmt3</code>)</summary>

```powershell
# Clone the GPL-3.0 model code separately (NOT installed by the extra above):
git clone https://huggingface.co/spaces/mimbres/YourMT3
# Download a checkpoint into the checkout per its README / colab, then point us at both:
setx AUDIO2AY3_YOURMT3_DIR        "C:\path\to\YourMT3"
# Optional: the preset supplies a checkpoint per variant; override it only if yours differs:
setx AUDIO2AY3_YOURMT3_CHECKPOINT "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
# Optional: pick a variant (default is "YMT3+", which did best in testing):
setx AUDIO2AY3_YOURMT3_MODEL      "YMT3+"

python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-ymt3.ym --transcription yourmt3
```

</details>

> `AUDIO2AY3_YOURMT3_DIR` always overrides the cache dir when set. CPU inference works (fp32, slow);
> a CUDA GPU uses fp16 automatically. The checkpoint files are large — `setup-yourmt3` pulls them via
> git-lfs when available, otherwise follow the upstream repo's download instructions. If
> `model_helper.py`/`amt/src` aren't importable from the checkout, the run fails with a clear message
> naming the fix.

### 3. Verify the install

```powershell
pytest                 # full unit suite should be green
audio2ay3 --help       # CLI is wired
```

## Best conversion setups

Two independent dimensions determine quality: how many AY channels you target,
and how cleanly the audio is separated before transcription.

### 1. Always use dual-AY (`--chips 2`)

A single AY-3-8910 has three tone channels. With `--chips 2` the converter
targets six: bass gets a dedicated channel on chip 1, drums are isolated on
chip 1's noise channel, and the melody spreads across four channels instead of
 one or two. The result is denser polyphony with far fewer dropped notes.
 `convert` writes two files — `<name>.ym` (chip 0) and `<name>.ay2.ym` (chip 1);
`preview` renders both chips into one mixed audio file.

```powershell
audio2ay3 convert song.mp3 --chips 2
```

Use `--explain` after conversion to see the voice-contention report: it
quantifies exactly how many notes were dropped and confirms that the second
chip was worth adding.

### 2a. Best quality — pre-separated stems (`--stems-dir`)

If you have the original project stems (exported from your DAW or any other
source), pass them via `--stems-dir` instead of letting Demucs do the
separation. Each stem is a perfectly isolated signal with zero bleed, which
means:

- Basic Pitch transcribes the **Synth** stem with no bass or drum
contamination — cleaner notes,
  fewer phantom pitches, better onset timing.
- The **Bass** stem is perfectly isolated — the dedicated bass channel tracks
the original more
  accurately.
- The **Drums** stem has no bleed — percussion detection fires only on real
hits.
- **FX** stems (when present) are mixed into the instrumental, adding tonal
effects as extra notes.
- Demucs is **skipped entirely** — the pipeline runs significantly faster.

**Stem folder layout** — name the folder exactly after the audio file stem:

```
samples/stems/
  Goblins_Lair/
    Goblins_Lair (Synth).mp3    ← melody / harmony  (required)
    Goblins_Lair (Bass).mp3     ← bass               (optional)
    Goblins_Lair (Drums).mp3    ← drums              (optional)
    Goblins_Lair (FX).mp3       ← effects            (optional)
```

```powershell
# Single song — stems + dual-AY
audio2ay3 convert samples\long\Goblins_Lair.mp3 --stems-dir samples\stems --chips 2

# Batch: convert every stems folder, output dual-AY .ym + mixed .mp3
.\convert_all_stems_dual.bat
# or directly:
python scripts\convert_long_dual.py --stems-dir samples\stems --stems-only --out-dir results\stems_dual
```

### 2b. Good quality — fine-tuned Demucs (`--separation demucs-ft`)

When you only have the mixed audio (no project stems), use the fine-tuned Demucs model. It
produces better stem separation than the default `htdemucs` at the cost of roughly 4× the
separation time, and the improved stems propagate directly into transcription accuracy.

```powershell
# Single song
audio2ay3 convert samples\long\Goblins_Lair.mp3 --separation demucs-ft --chips 2

# Batch — dual-AY, fine-tuned Demucs
.\convert_all_demucs_tf_dual.bat
# or directly:
python scripts\convert_long_dual.py --separation demucs-ft --out-dir results\demucs-ft_dual
```

The default `--separation demucs` (un-fine-tuned) is the fastest neural option and a reasonable
starting point, but `demucs-ft` is worth the extra time for final-quality renders.

> **Tip — keep the separator's stems.** Add `--save-stems` to any Demucs run to dump the raw
> separator output (every source, stereo, native sample rate) next to the `-o` file as
> `<name> (Vocals).wav`, `<name> (Drums).wav`, `<name> (Bass).wav`, `<name> (Other).wav` (plus
> `(Guitar)`/`(Piano)` for `demucs6`). The files round-trip straight back through `--stems-dir`
> (the `Other` stem is accepted as the melodic stem), so you can separate once and re-render
> many times without paying for Demucs again. It is a no-op when no separator runs
> (`--separation none`, the `mt3`/`yourmt3` backends, or `--stems-dir`). Add
> `--save-stems-format mp3` to write compact MP3s (~1/10th the size, encoded at `--bitrate`)
> instead of lossless WAV when disk space matters on a big batch.

---

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
| `--chips {1,2}` | Number of AY chips: `1` (3 channels) or `2` (dual-AY, 6 channels — recommended). |
| `--stems-dir DIR` | Load pre-separated stems from `<DIR>/<song>/` instead of running Demucs (see [Best conversion setups](#best-conversion-setups)). |
| `--separation {demucs,demucs-ft,demucs6,none}` | Neural separation backend (default `demucs`; `demucs-ft` = better/slower; `none` for instrumental input). |
| `--transcription {basic-pitch,mt3,yourmt3}` | Transcription backend (default `basic-pitch`). |
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
- **Phase 7 — scalability:** `--chips 2` (dual-AY, 6 tone channels) is implemented — bass keeps
  its own channel, percussion is isolated on the second chip, and the melody spreads across four
  channels. `convert` writes chip 1 to `<name>.ay2.ym`; `preview` mixes both chips. `--frame-rate
  100` is also available.
- **Phase 8 — hardening:** edge-case inputs, richer CLI UX, perceptual regression metrics in CI.
- **Alternate backends:** `--transcription mt3 / onsets-frames` and `--separation spleeter` are
  recognised but not wired (they raise a clear "not available" error).

The emulator and `validate` path (Milestone 1) are stable and are the ground truth everything
else is measured against.
