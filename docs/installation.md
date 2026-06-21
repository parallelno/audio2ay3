# audio2ay3 installation guide

How to install audio2ay3 and the optional backends. For day-to-day command usage once installed,
see the [CLI reference](cli.md).

## At a glance

| You want to… | Install | Python |
|--------------|---------|--------|
| Run the emulator, `validate`, and the test suite | `pip install -e ".[dev]"` | 3.11 or 3.12 |
| …also write MP3 output | `pip install -e ".[dev,mp3]"` | 3.11 or 3.12 |
| …also read FLAC/OGG input (needed by `convert`) | `pip install -e ".[dev,mp3,audio]"` | 3.11 or 3.12 |
| The full neural converter (`convert`/`preview`) | `pip install -e ".[dev,mp3,neural]"` | **3.11 recommended** (3.12 needs an extra step) |
| MT3 multi-instrument transcription | `pip install "mt3 @ git+https://github.com/magenta/mt3"` | **Linux/WSL only** |
| YourMT3+ multi-instrument transcription (**experimental, not recommended**) | `pip install -e ".[yourmt3]"` + `audio2ay3 setup-yourmt3` (clones the GPL model repo) | **native Windows OK** |

Extras combine inside one bracket, e.g. `".[dev,mp3,audio]"`.

## Requirements

- **Core (emulator, `validate`, tests):** Python 3.11 or 3.12, and Git.
- **Neural converter (`convert`/`preview`, the `neural` extra):**
  - **Python 3.11 — recommended.** `pip install -e ".[neural]"` resolves cleanly.
  - **Python 3.12 — works with one extra step.** Basic Pitch 0.4.0 declares
    `tensorflow>=2.4.1,<2.15.1` on Python ≥ 3.11, but TensorFlow only ships 3.12 wheels from 2.16+,
    so that pin is unsatisfiable on 3.12 and the plain install fails. We never run TensorFlow (the
    transcriber uses the **ONNX** model), so the fix is to install Basic Pitch without its
    dependencies — see [Python 3.12: the neural extra](#python-312-the-neural-extra).
- **Disk:** ~2–6 GB free **if installing the `neural` extra** (pulls PyTorch plus a transcription
  runtime). The first `convert` with Demucs downloads model weights to a local cache.
- **GPU:** optional. Neural models fall back to CPU automatically, or force it with `--no-gpu`.

## 1. Clone and create a virtual environment

**Windows (PowerShell):**

```powershell
git clone <your-repo-url> audio2ay3
cd audio2ay3
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

## 2. Pick what to install

Editable install with extras (see the table at the top for the common combinations):

```powershell
pip install -e ".[dev]"              # emulator + validate + tests, no heavy deps
pip install -e ".[dev,mp3]"          # + MP3 output
pip install -e ".[dev,mp3,audio]"    # + FLAC/OGG decoding (convert needs this)
pip install -e ".[dev,mp3,neural]"   # full converter (Python 3.11)
```

The extras:

| Extra | Pulls in | Used for |
|-------|----------|----------|
| `dev` | pytest, pytest-xdist, ruff, mypy | running tests and linting |
| `mp3` | lameenc | MP3 output from `validate`/`preview` |
| `audio` | soundfile | decoding non-WAV input audio |
| `neural` | demucs (PyTorch), basic-pitch, onnxruntime, soundfile | the default `convert`/`preview` pipeline |
| `mt3` | jax, note-seq, seqio, t5, gin-config, tensorflow, librosa (convenience subset only) | the experimental MT3 backend — **see the dedicated section, this extra alone is not enough** |
| `yourmt3` | torch, torchaudio, lightning, transformers, einops, mido, librosa, pretty_midi, wandb | the **experimental** YourMT3+ backend (runs on native Windows, but transcribed sparsely vs basic-pitch in testing — not recommended) — **the GPL-3.0 model code is cloned separately; see the dedicated section** |

On **Python 3.11** the `neural` command above resolves as-is. On **Python 3.12** use the
workaround below.

> **Note (TensorFlow):** Basic Pitch pulls in TensorFlow on Python ≥ 3.11, and installing it
> downgrades `numpy` to the 1.26.x line — that's expected; numba and the emulator are fine with it.
> This project prefers the **ONNX** model and never runs TF inference, so a slow or fragile TF
> install does not affect conversion.

## Python 3.12: the neural extra

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

## MT3 transcription (`--transcription mt3`) — experimental, Linux/WSL only

MT3 (Google Magenta) is a heavyweight multi-instrument transcriber: one pass yields pitched notes,
bass, and drums together with General-MIDI instrument identity, so it self-routes percussion and
the bass line (no Demucs separation needed). It is fully opt-in and never touched by the default
`convert` or the test suite.

> **Windows is not supported for MT3.** Contrary to a common assumption, **JAX is not the
> blocker** — JAX ships native Windows x86_64 CPU wheels now. The blocker is **`tensorflow-text`**
> (pulled transitively by `t5`/`seqio`), which publishes **only Linux and macOS-ARM wheels — no
> `win_amd64` wheel and no sdist** — so the stack cannot `pip install` on native Windows. Run MT3
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
> version errors, match the pins in MT3's
> [upstream colab](https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb).

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

## YourMT3+ transcription (`--transcription yourmt3`) — experimental, not recommended

> **Status: experimental — not recommended.** In testing it transcribed *more sparsely* than
> `basic-pitch` on real material (e.g. `Goblins_Lair`: ~55% pitched duty cycle, multi-second melodic
> gaps, very staccato notes) and was far slower (tens of minutes on a GPU vs seconds for
> basic-pitch). The shortfall is the raw transcription, not the AY arrangement. Kept as an optional
> backend for experimentation only; prefer **`basic-pitch`** (native-Windows default) or **MT3**
> (Linux/WSL) for multi-instrument quality.

[YourMT3+](https://github.com/mimbres/YourMT3) is MT3-class multi-instrument transcription on a
**pure PyTorch** stack (`torch`, `torchaudio`, `lightning`, `transformers`, `einops`, `mido`,
`librosa`) with **no JAX / t5x / tensorflow-text** — so, unlike MT3, it `pip install`s on **native
Windows**. One pass yields pitched notes, bass, and drums together with General-MIDI identity, so it
self-routes like the MT3 path (no Demucs separation). Fully opt-in; never touched by the default
`convert` or the test suite.

> **License — read before using.** YourMT3 is **GPL-3.0**; audio2ay3 is **MIT**. To keep GPL code
> out of this MIT tree it is an *optional, user-installed* backend: **we do not vendor or bundle any
> YourMT3 code.** Both the helper and the manual route fetch the GPL repo onto *your* machine at
> runtime; the adapter in `audio2ay3/analysis/_yourmt3_infer.py` imports it at runtime. Distributing
> your own combined build is your responsibility under the GPL.

```powershell
# Native Windows (PowerShell) or Linux/WSL, Python 3.11/3.12. The pip stack is pure PyTorch:
pip install -e ".[yourmt3]"

# Easy path: clone + wire up the GPL backend into a per-user cache (needs git + git-lfs):
python -m audio2ay3 setup-yourmt3
python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-ymt3.ym --transcription yourmt3
```

`setup-yourmt3` clones the YourMT3 **HuggingFace Space** (which colocates `model_helper.py` + `amt/`
and carries the LFS checkpoints) into a per-user cache dir, verifies the layout, and tells you if a
checkpoint still needs downloading. Flags: `--dir`, `--repo-url`, `--model` (default
`YPTF.MoE+Multi (noPS)`), `--force`. It only needs `git`/`git-lfs` — no torch.

<details><summary>Manual setup (alternative to <code>setup-yourmt3</code>)</summary>

```powershell
# Clone the GPL-3.0 model code separately (NOT installed by the extra above):
git clone https://huggingface.co/spaces/mimbres/YourMT3
# Download a checkpoint into the checkout per its README/colab, then point us at both:
setx AUDIO2AY3_YOURMT3_DIR        "C:\path\to\YourMT3"
setx AUDIO2AY3_YOURMT3_CHECKPOINT "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
setx AUDIO2AY3_YOURMT3_MODEL      "YPTF.MoE+Multi (noPS)"   # optional; this is the default variant

python -m audio2ay3 convert samples/long/Goblins_Lair.mp3 -o build/goblins-ymt3.ym --transcription yourmt3
```

</details>

> `AUDIO2AY3_YOURMT3_DIR` overrides the cache dir when set. CPU inference works (fp32, slow); a CUDA
> GPU uses fp16 automatically. Checkpoints are large — `setup-yourmt3` pulls them via git-lfs when
> available, otherwise follow the upstream repo's download instructions. If `model_helper.py` /
> `amt/src` are not importable from the checkout, or a checkpoint is missing, the run fails with a
> clear message naming the fix rather than a deep ImportError.

## 3. Verify the install

```powershell
pytest                 # full unit suite should be green
ruff check src tests   # lint clean
audio2ay3 --help       # CLI is wired
```

If the `audio2ay3` console script isn't on your PATH (e.g. installed into a global interpreter
rather than a venv), use the module form instead:

```powershell
python -m audio2ay3 --help
```

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `pip install -e ".[neural]"` fails on Python 3.12 with a TensorFlow resolution error | Expected — use [Python 3.12: the neural extra](#python-312-the-neural-extra). |
| `convert` errors that a backend is missing | The `neural` (and `audio`) extras aren't installed in the active environment. |
| `validate` fails to write MP3 | Install the `mp3` extra (`lameenc`), or use `--wav`. |
| `convert` can't read a FLAC/OGG/MP3 input | Install the `audio` extra (`soundfile`); MP3 decoding also depends on your libsndfile build. |
| `audio2ay3: command not found` | The console script isn't on PATH; run `python -m audio2ay3 …`. |
| MT3 install fails on Windows | Not supported on native Windows (`tensorflow-text` has no Windows wheel); use WSL2/Linux. |
| MT3 first run fails with a gin error | The `mt3` wheel omits its `gin/` configs; apply the fix in the MT3 section above. |

## Next steps

- [CLI reference](cli.md) — the `validate`, `convert`, and `preview` commands and their options.
- [design/](../design/) — the full design and implementation plan.
