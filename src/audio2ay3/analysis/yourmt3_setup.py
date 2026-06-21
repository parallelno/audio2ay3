"""Optional installer for the YourMT3+ GPL backend — keeps GPL code out of this MIT tree.

``audio2ay3 setup-yourmt3`` clones the YourMT3 backend into a local cache directory **at runtime**:
the GPL-3.0 code lands on the user's own machine and is never bundled into this repository (which
stays MIT). This is the same stance as the MT3 backend, just automated so ``--transcription
yourmt3`` works without manual ``git clone`` + env-var juggling.

We clone the **HuggingFace Space** (not the bare GitHub repo) because it colocates the inference
glue (`model_helper.py`) with the model package (`amt/src`) — exactly what
:mod:`audio2ay3.analysis._yourmt3_infer` imports — and carries the checkpoints via git-LFS, so a
single clone is self-contained when git-lfs is installed.

Nothing here imports torch; setup only needs ``git`` (and ``git-lfs`` for the checkpoints).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from ._yourmt3_infer import _DEFAULT_MODEL, _MODEL_PRESETS, default_yourmt3_dir

# The HF Space bundles model_helper.py + amt/ + (LFS) checkpoints in one place.
_DEFAULT_REPO_URL = "https://huggingface.co/spaces/mimbres/YourMT3"


def _run(cmd: list[str]) -> int:
    """Run *cmd*, streaming its output; return the exit code (never raises on non-zero)."""
    return subprocess.run(cmd, check=False).returncode  # noqa: S603 - fixed argv, no shell


def setup_yourmt3(
    target_dir: str | Path | None = None,
    *,
    repo_url: str = _DEFAULT_REPO_URL,
    model_name: str = _DEFAULT_MODEL,
    force: bool = False,
    runner: Callable[[list[str]], int] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Clone the YourMT3 backend into *target_dir* (default: the local cache) and verify it.

    ``runner`` (a ``list[str] -> int`` command executor) is injectable so the flow is unit-testable
    without touching the network. Returns the resolved checkout directory.
    """
    run = runner or _run
    if model_name not in _MODEL_PRESETS:
        raise ValueError(
            f"unknown model variant {model_name!r}; choose from: " + ", ".join(sorted(_MODEL_PRESETS))
        )
    target = Path(target_dir) if target_dir else default_yourmt3_dir()

    if run(["git", "--version"]) != 0:
        raise RuntimeError("git is required for `setup-yourmt3` but was not found on PATH.")
    if run(["git", "lfs", "version"]) != 0:
        log(
            "WARNING: git-lfs not found — checkpoints stored via LFS will download as tiny pointer "
            "files, not the real weights. Install git-lfs (https://git-lfs.com) and re-run with "
            "--force, or fetch a checkpoint manually."
        )

    already = (target / ".git").exists()
    if already and not force:
        log(f"YourMT3 already present at {target} (pass --force to update). Skipping clone.")
    elif already and force:
        log(f"Updating existing checkout at {target} ...")
        if run(["git", "-C", str(target), "pull"]) != 0:
            raise RuntimeError(f"failed to update the checkout at {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"Cloning YourMT3 (GPL-3.0) into {target} ...")
        if run(["git", "clone", repo_url, str(target)]) != 0:
            raise RuntimeError(
                f"git clone of {repo_url} failed. Check connectivity, or pass --repo-url to use a "
                "mirror/fork."
            )

    _verify_layout(target, model_name, log)
    log(f"\nDone. The 'yourmt3' backend will use: {target}")
    log("Install the runtime stack if you have not yet:  pip install -e \".[yourmt3]\"")
    log("Then convert with:  audio2ay3 convert <audio> --transcription yourmt3")
    return target


def _verify_layout(target: Path, model_name: str, log: Callable[[str], None]) -> None:
    """Warn (don't fail) if the inference glue or the chosen checkpoint isn't in the checkout."""
    if not (target / "model_helper.py").exists() or not (target / "amt" / "src").exists():
        log(
            "WARNING: 'model_helper.py' and/or 'amt/src' were not found in the checkout. If you "
            "cloned the GitHub repo rather than the HF Space, the inference glue lives elsewhere; "
            "re-run with --repo-url https://huggingface.co/spaces/mimbres/YourMT3"
        )
    checkpoint = _MODEL_PRESETS[model_name][0]
    exp_id = checkpoint.split("@", 1)[0]  # the directory/experiment name before '@<file>.ckpt'
    if not _has_real_checkpoint(target, exp_id):
        log(
            f"\nNOTE: no checkpoint for '{model_name}' (exp '{exp_id}') was found in {target}.\n"
            "Checkpoints are large and need git-lfs (install it, then re-run with --force) or a "
            "manual download from the Colab demo linked at https://github.com/mimbres/YourMT3 .\n"
            f"Place the weights so YourMT3 resolves '{checkpoint}' before converting."
        )


def _has_real_checkpoint(target: Path, exp_id: str) -> bool:
    """True if a non-pointer ``.ckpt`` for *exp_id* exists (LFS pointer files are a few hundred B)."""
    if not target.exists():
        return False
    try:
        for p in target.rglob("*.ckpt"):
            try:
                if exp_id in str(p) and p.stat().st_size > 1_000_000:
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False
