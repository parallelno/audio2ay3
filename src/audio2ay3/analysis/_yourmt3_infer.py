"""YourMT3+ (mimbres) multi-instrument transcription driver — isolated heavy backend.

Like :mod:`._mt3_infer`, this module is quarantined from :mod:`.transcribe`: importing it is cheap
because every heavy dependency (torch/torchaudio/lightning/transformers and the YourMT3 model code
itself) is imported lazily, inside the functions that need it. The deterministic core and its tests
never drag the neural stack in.

Why YourMT3+ instead of MT3: it is MT3-class multi-instrument transcription rebuilt on a *pure
PyTorch* stack (torch, torchaudio, lightning, transformers, einops, mido, librosa) with **no
JAX / t5x / tensorflow-text**, so unlike MT3 it pip-installs on native Windows. A single pass emits
a General-MIDI multi-instrument MIDI file (pitched notes + bass + drums together), which we route
through the same :func:`audio2ay3.analysis.transcribe.note_sequence_to_transcription` glue as MT3.

License caveat (important): YourMT3 is **GPL-3.0**, while audio2ay3 is MIT. This project therefore
treats it exactly like the MT3 backend — an *optional, user-installed* component. We do **not**
vendor or bundle any YourMT3 code; the user clones the GPL repo themselves and points an env var at
it, and this thin adapter imports it dynamically at runtime. That keeps GPL-licensed code out of the
MIT tree.

Setup (see the ``[yourmt3]`` extra for the pip-installable runtime deps):

    pip install -e ".[yourmt3]"          # torch/torchaudio/lightning/transformers/.../pretty_midi
    git clone https://github.com/mimbres/YourMT3   # the GPL-3.0 model code (kept out of this repo)
    # download a checkpoint into the repo per its README / colab, then point us at both:
    setx AUDIO2AY3_YOURMT3_DIR        C:\\path\\to\\YourMT3
    setx AUDIO2AY3_YOURMT3_CHECKPOINT  some_checkpoint@last.ckpt   # optional; preset has a default
    setx AUDIO2AY3_YOURMT3_MODEL      "YMT3+"                      # optional; this is the default

The output is parsed with ``pretty_midi`` into a duck-typed ``NoteSequence`` shim;
:func:`audio2ay3.analysis.transcribe.note_sequence_to_transcription` turns it into the neutral IR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_DIR_ENV = "AUDIO2AY3_YOURMT3_DIR"
_CHECKPOINT_ENV = "AUDIO2AY3_YOURMT3_CHECKPOINT"
_MODEL_ENV = "AUDIO2AY3_YOURMT3_MODEL"

# YourMT3 organises checkpoints under a project id; "2024" matches the released MLSP'24 models.
_PROJECT = "2024"
# Default variant: the MT3-lineage model with a GM-extended vocabulary (``gm_ext``) and no
# pitch-shift augmentation (``nops``). It is lighter/faster than the Perceiver-TF MoE decoders and
# transcribed notably better on our test material (the heavy MoE variants came out sparse).
_DEFAULT_MODEL = "YMT3+"

# Architecture flags + default checkpoint per model variant, transcribed from the upstream demo's
# ``app.py``. These are invocation parameters (CLI-style flags) for the user's own YourMT3 install,
# not vendored model code. ``-pr`` (precision) and the leading checkpoint name are appended later.
_MOE_FLAGS = [
    "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
    "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu",
    "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1",
]
_MULTI_FLAGS = [
    "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
    "-ac", "spec", "-hop", "300", "-atc", "1",
]
_MODEL_PRESETS: dict[str, tuple[str, list[str]]] = {
    "YPTF.MoE+Multi (noPS)": (
        "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt", _MOE_FLAGS
    ),
    "YPTF.MoE+Multi (PS)": (
        "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt", _MOE_FLAGS
    ),
    "YPTF+Multi (PS)": (
        "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k"
        "@model.ckpt",
        _MULTI_FLAGS,
    ),
    "YMT3+": ("notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt", []),
}

# Cache the (expensive) model load across calls in one process, keyed by (dir, model, checkpoint).
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def default_yourmt3_dir() -> Path:
    """The local cache directory ``audio2ay3 setup-yourmt3`` clones the GPL backend into.

    Picked from the platform cache root (``%LOCALAPPDATA%`` on Windows, ``$XDG_CACHE_HOME`` or
    ``~/.cache`` elsewhere) so a one-time ``setup-yourmt3`` makes ``--transcription yourmt3`` work
    without any env var. ``AUDIO2AY3_YOURMT3_DIR`` still overrides this when set.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "audio2ay3" / "yourmt3"


def _resolve_model_name(explicit: str | None = None) -> str:
    """Resolve the YourMT3 variant to use, validating it against :data:`_MODEL_PRESETS`.

    Precedence: an *explicit* name (e.g. from ``--yourmt3-model``) wins, then the
    ``AUDIO2AY3_YOURMT3_MODEL`` env var, then :data:`_DEFAULT_MODEL`. Raises ``RuntimeError`` naming
    the valid variants when the resolved name is unknown.
    """
    name = (explicit or "").strip() or os.environ.get(_MODEL_ENV, "").strip() or _DEFAULT_MODEL
    if name not in _MODEL_PRESETS:
        raise RuntimeError(
            f"{name!r} is not a known YourMT3 variant; choose one of: "
            + ", ".join(sorted(_MODEL_PRESETS))
        )
    return name


def transcribe_yourmt3(audio: np.ndarray, sr: int, model_name: str | None = None):
    """Transcribe mono ``audio`` (sample rate ``sr``) with YourMT3+, returning a NoteSequence shim.

    ``model_name`` selects the variant (e.g. ``"YMT3+"``); when ``None`` it falls back to the
    ``AUDIO2AY3_YOURMT3_MODEL`` env var and then :data:`_DEFAULT_MODEL`.

    Raises ``RuntimeError`` with actionable guidance when the repo checkout, the optional pip stack,
    or a checkpoint is unavailable — rather than surfacing a raw ImportError from deep in the model.
    """
    repo_dir = _repo_dir()
    model_name = _resolve_model_name(model_name)
    checkpoint = os.environ.get(_CHECKPOINT_ENV, "").strip()
    model = _get_model(repo_dir, model_name, checkpoint)
    midi_path = _run_transcribe(model, repo_dir, audio, sr)
    return _pretty_midi_to_note_sequence(_load_pretty_midi(midi_path))


def _repo_dir() -> str:
    path = os.environ.get(_DIR_ENV, "").strip()
    if path:
        if not Path(path).exists():
            raise RuntimeError(f"{_DIR_ENV} points at a missing path: {path!r}")
        return path
    default = default_yourmt3_dir()
    if default.exists():
        return str(default)
    raise RuntimeError(
        "YourMT3 transcription needs the model repo checkout. Easiest: run "
        f"`audio2ay3 setup-yourmt3` to fetch it (GPL-3.0) into the local cache ({default}). "
        f"Or clone https://github.com/mimbres/YourMT3 yourself and set {_DIR_ENV}. The GPL model "
        "code is kept separate from this MIT project (see the 'yourmt3' backend setup)."
    )


def _ensure_on_path(repo_dir: str) -> None:
    """Make YourMT3's ``model_helper`` and its ``amt/src`` package importable from *repo_dir*."""
    candidates = [Path(repo_dir) / "amt" / "src", Path(repo_dir)]
    for cand in candidates:
        s = str(cand)
        if cand.exists() and s not in sys.path:
            sys.path.insert(0, s)


def _require_dependencies(repo_dir: str):
    """Import YourMT3's inference glue, raising a clean RuntimeError when the stack is absent."""
    _ensure_on_path(repo_dir)
    try:
        import torch  # noqa: F401

        from model_helper import load_model_checkpoint, transcribe  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the optional stack + GPL checkout
        raise RuntimeError(
            "YourMT3 transcription needs the 'yourmt3' extra (PyTorch stack) plus the model repo on "
            f"the path. Install with: pip install -e \".[yourmt3]\"  and ensure {_DIR_ENV} points "
            "at a YourMT3 checkout whose 'model_helper.py' and 'amt/src' are importable (see "
            "audio2ay3/analysis/_yourmt3_infer.py)."
        ) from exc
    return load_model_checkpoint, transcribe


def _get_model(repo_dir: str, model_name: str, checkpoint: str):
    key = (repo_dir, model_name, checkpoint)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    import torch

    load_model_checkpoint, _ = _require_dependencies(repo_dir)
    default_ckpt, flags = _MODEL_PRESETS[model_name]
    ckpt = checkpoint or default_ckpt
    # Half precision is GPU-only; CPU inference must run in fp32 (slower but functional).
    precision = "16" if torch.cuda.is_available() else "32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = [ckpt, "-p", _PROJECT, *flags, "-pr", precision]
    # YourMT3 resolves the checkpoint path (amt/logs/<project>/<exp>/checkpoints/*.ckpt) relative
    # to the *current working directory*, so the model must be loaded from inside the checkout —
    # otherwise it fails with "No checkpoint found in amt." regardless of where the CLI was run.
    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        model = load_model_checkpoint(args=args, device=device)
    finally:
        os.chdir(prev_cwd)
    model.to(device)
    _MODEL_CACHE[key] = model
    return model


def _run_transcribe(model, repo_dir: str, audio: np.ndarray, sr: int) -> str:
    """Write *audio* to a temp WAV, run YourMT3's ``transcribe``, return the output MIDI path.

    YourMT3's ``transcribe`` writes ``./model_output/<track>.mid`` relative to the working
    directory, so we run it inside a temp dir to avoid polluting the caller's cwd.
    """
    import tempfile

    import torchaudio

    _, transcribe = _require_dependencies(repo_dir)
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=tuple(range(1, audio.ndim)))
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    prev_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "input.wav")
        _write_wav_torchaudio(torchaudio, wav_path, mono, sr)
        info = torchaudio.info(wav_path)
        audio_info = {
            "filepath": wav_path,
            "track_name": "input",
            "sample_rate": int(info.sample_rate),
            "bits_per_sample": int(getattr(info, "bits_per_sample", 16)),
            "num_channels": int(getattr(info, "num_channels", 1)),
            "num_frames": int(info.num_frames),
            "duration": float(info.num_frames) / float(info.sample_rate or sr),
            "encoding": getattr(info, "encoding", "PCM_S"),
        }
        os.chdir(tmp)
        try:
            midi_path = transcribe(model, audio_info)
        finally:
            os.chdir(prev_cwd)
        # ``transcribe`` returns a path relative to its cwd (tmp); resolve it before tmp is removed.
        resolved = Path(midi_path)
        if not resolved.is_absolute():
            resolved = Path(tmp) / midi_path
        stable = Path(prev_cwd) / "_yourmt3_last_output.mid"
        stable.write_bytes(resolved.read_bytes())
    return str(stable)


def _write_wav_torchaudio(torchaudio, path: str, mono: np.ndarray, sr: int) -> None:
    import torch

    wav = torch.from_numpy(mono).unsqueeze(0)  # (1, n) mono
    torchaudio.save(path, wav, sr)


def _load_pretty_midi(midi_path: str):
    try:
        import pretty_midi
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            "Parsing YourMT3 output needs pretty_midi (part of the 'yourmt3' extra): "
            'pip install -e ".[yourmt3]"'
        ) from exc
    return pretty_midi.PrettyMIDI(midi_path)


def _pretty_midi_to_note_sequence(pm):
    """Adapt a ``pretty_midi.PrettyMIDI`` into the duck-typed NoteSequence the IR converter wants.

    ``pm`` is duck-typed: it needs an iterable ``instruments`` whose items expose ``program`` (GM),
    ``is_drum`` and ``notes`` (each with ``pitch``/``start``/``end``/``velocity``). The returned
    object exposes ``total_time`` and ``notes`` with ``pitch``/``start_time``/``end_time``/
    ``velocity``/``program``/``is_drum``, matching what
    :func:`audio2ay3.analysis.transcribe.note_sequence_to_transcription` reads.
    """
    notes: list[SimpleNamespace] = []
    total_time = 0.0
    for inst in getattr(pm, "instruments", ()):
        program = int(getattr(inst, "program", 0))
        is_drum = bool(getattr(inst, "is_drum", False))
        for n in getattr(inst, "notes", ()):
            end = float(n.end)
            notes.append(
                SimpleNamespace(
                    pitch=int(n.pitch),
                    start_time=float(n.start),
                    end_time=end,
                    velocity=int(n.velocity),
                    program=program,
                    is_drum=is_drum,
                )
            )
            total_time = max(total_time, end)
    return SimpleNamespace(notes=notes, total_time=total_time)
