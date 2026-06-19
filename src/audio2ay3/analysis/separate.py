"""Source separation: drop vocals and (optionally) isolate stems before transcription.

Per the project's neural-only analysis decision, separation is a neural step (Demucs by
default). The model is imported lazily so the package — and the whole emulator/``validate``
path — stays usable without the heavy ``[neural]`` extra installed.

``mode="none"`` is a passthrough for already-instrumental input.
"""

from __future__ import annotations

import numpy as np


def separate(audio: np.ndarray, sr: int, mode: str = "demucs") -> np.ndarray:
    """Return an instrumental mono mix for transcription.

    - ``none``   -> input unchanged.
    - ``demucs`` -> Demucs stems summed minus vocals (lazy import).
    - ``spleeter`` -> not wired yet.
    """
    if mode == "none":
        return audio
    if mode == "demucs":
        return _separate_demucs(audio, sr)
    if mode == "spleeter":
        raise NotImplementedError(
            "Spleeter backend is not wired yet; use --separation demucs or none."
        )
    raise ValueError(f"unknown separation mode: {mode!r}")


def _separate_demucs(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Demucs separation needs the 'neural' extra: pip install audio2ay3[neural]"
        ) from exc

    model = get_model("htdemucs")
    model.eval()

    # Demucs wants (batch, channels, samples) at the model's sample rate.
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if sr != model.samplerate:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    wav = wav.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # mono -> fake stereo, add batch

    with torch.no_grad():
        stems = apply_model(model, wav, split=True, overlap=0.1)[0]

    # Sum every stem except 'vocals' back into one instrumental track.
    names = model.sources
    keep = [stems[i] for i, name in enumerate(names) if name != "vocals"]
    inst = torch.stack(keep).sum(dim=0).mean(dim=0)  # -> mono
    out = inst.cpu().numpy().astype(np.float32)

    if sr != model.samplerate:
        from .load_audio import _resample_linear

        out = _resample_linear(out, model.samplerate, sr)
    return out
