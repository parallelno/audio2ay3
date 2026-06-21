"""Source separation: split vocals / drums / pitched content before transcription.

Per the project's neural-only analysis decision, separation is a neural step (Demucs by
default). The model is imported lazily so the package — and the whole emulator/``validate``
path — stays usable without the heavy ``[neural]`` extra installed.

Three products come out of one Demucs pass:

* **instrumental** — the pitched harmony/lead content (every stem except *vocals*, *drums*,
  and *bass*), fed to the note transcriber. Excluding drums keeps transients from being
  mis-heard as pitches; excluding bass frees it for its own dedicated channel.
* **bass** — the isolated bass stem, transcribed on its own and given a dedicated AY tone
  channel so the low end never competes with the lead. ``None`` when unavailable.
* **drums** — the isolated drum stem, fed to :mod:`.percussion_detect` so hits can be placed on
  the AY noise channel. ``None`` when no real separation happened.

``mode="none"`` is a passthrough for already-instrumental input (no bass/drum stems).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeparationResult:
    """Stems handed to the transcription stage."""

    instrumental: np.ndarray  # mono pitched content (no vocals, drums, or bass)
    drums: np.ndarray | None  # mono isolated drum stem, or None when unavailable
    bass: np.ndarray | None  # mono isolated bass stem, or None when unavailable
    sr: int


# Friendly separation mode -> Demucs pretrained model name. ``htdemucs`` is the 4-source default;
# ``htdemucs_ft`` is its fine-tuned bag-of-4 (better SDR, ~4x slower); ``htdemucs_6s`` adds
# separate guitar + piano stems (experimental — those two are noisier than the core four).
_DEMUCS_MODELS = {
    "demucs": "htdemucs",
    "demucs-ft": "htdemucs_ft",
    "demucs6": "htdemucs_6s",
}


def separate_stems(audio: np.ndarray, sr: int, mode: str = "demucs") -> SeparationResult:
    """Split *audio* into a pitched instrumental plus isolated bass and drum stems.

    - ``none``      -> input is the instrumental, no bass/drum stems.
    - ``demucs``    -> Demucs ``htdemucs`` (4-source): instrumental = (all − vocals − drums −
      bass), bass & drums kept separate.
    - ``demucs-ft`` -> Demucs ``htdemucs_ft`` (fine-tuned bag-of-4): same stems, better SDR,
      ~4× slower — a quality option for offline ``convert``.
    - ``demucs6``   -> Demucs ``htdemucs_6s`` (6-source, **experimental**): the model also
      separates guitar + piano internally; we still fold every non-vocal/drum/bass stem
      (other + guitar + piano) into the instrumental. Those two extra stems are noisier than
      the core four, so treat it as experimental.
    - ``spleeter``  -> not wired yet.
    """
    if mode == "none":
        return SeparationResult(instrumental=audio, drums=None, bass=None, sr=sr)
    if mode in _DEMUCS_MODELS:
        return _separate_demucs(audio, sr, _DEMUCS_MODELS[mode])
    if mode == "spleeter":
        raise NotImplementedError(
            "Spleeter backend is not wired yet; use --separation demucs or none."
        )
    raise ValueError(f"unknown separation mode: {mode!r}")


def separate(audio: np.ndarray, sr: int, mode: str = "demucs") -> np.ndarray:
    """Back-compat helper: return only the pitched instrumental mono mix."""
    return separate_stems(audio, sr, mode).instrumental


def _separate_demucs(audio: np.ndarray, sr: int, model_name: str = "htdemucs") -> SeparationResult:
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Demucs separation needs the 'neural' extra: pip install audio2ay3[neural]"
        ) from exc

    model = get_model(model_name)
    model.eval()
    model_sr = model.samplerate

    # Demucs wants (batch, channels, samples) at the model's sample rate.
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if sr != model_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, model_sr)
    wav = wav.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # mono -> fake stereo, add batch

    with torch.no_grad():
        stems = apply_model(model, wav, split=True, overlap=0.1)[0]

    names = list(model.sources)

    def stem_mono(name: str):
        return stems[names.index(name)].mean(dim=0)  # channels -> mono

    # Pitched content = everything except vocals, drums, and bass (bass gets its own voice).
    pitched = [n for n in names if n not in ("vocals", "drums", "bass")]
    inst = torch.stack([stem_mono(n) for n in pitched]).sum(dim=0)
    inst_np = inst.cpu().numpy().astype(np.float32)

    drums_np = stem_mono("drums").cpu().numpy().astype(np.float32) if "drums" in names else None
    bass_np = stem_mono("bass").cpu().numpy().astype(np.float32) if "bass" in names else None

    if sr != model_sr:
        from .load_audio import _resample_linear

        inst_np = _resample_linear(inst_np, model_sr, sr)
        if drums_np is not None:
            drums_np = _resample_linear(drums_np, model_sr, sr)
        if bass_np is not None:
            bass_np = _resample_linear(bass_np, model_sr, sr)

    return SeparationResult(instrumental=inst_np, drums=drums_np, bass=bass_np, sr=sr)
