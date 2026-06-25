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
from pathlib import Path

import numpy as np


# Audio extensions tried when searching for a stem file, in preference order.
_AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def find_stems_folder(name: str, stems_dir: Path | str) -> Path | None:
    """Return ``stems_dir / name`` if that directory exists, otherwise ``None``."""
    d = Path(stems_dir) / name
    return d if d.is_dir() else None


@dataclass
class SeparationResult:
    """Stems handed to the transcription stage."""

    instrumental: np.ndarray  # mono pitched content (no vocals, drums, or bass)
    drums: np.ndarray | None  # mono isolated drum stem, or None when unavailable
    bass: np.ndarray | None  # mono isolated bass stem, or None when unavailable
    sr: int
    # Mono isolated vocal stem, kept only when ``keep_vocals=True`` so the sung melody can be
    # transcribed into a lead voice; ``None`` otherwise (the historical "drop vocals" path).
    vocals: np.ndarray | None = None


# Friendly separation mode -> Demucs pretrained model name. ``htdemucs`` is the 4-source default;
# ``htdemucs_ft`` is its fine-tuned bag-of-4 (better SDR, ~4x slower); ``htdemucs_6s`` adds
# separate guitar + piano stems (experimental — those two are noisier than the core four).
_DEMUCS_MODELS = {
    "demucs": "htdemucs",
    "demucs-ft": "htdemucs_ft",
    "demucs6": "htdemucs_6s",
}


def separate_stems(
    audio: np.ndarray, sr: int, mode: str = "demucs", *, keep_vocals: bool = False
) -> SeparationResult:
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

    When *keep_vocals* is true, the isolated vocal stem is returned in
    :attr:`SeparationResult.vocals` (instead of being discarded) so the caller can transcribe
    the sung melody into a lead voice. It stays ``None`` for ``mode="none"`` (no separation).
    """
    if mode == "none":
        return SeparationResult(instrumental=audio, drums=None, bass=None, sr=sr)
    if mode in _DEMUCS_MODELS:
        return _separate_demucs(audio, sr, _DEMUCS_MODELS[mode], keep_vocals=keep_vocals)
    if mode == "spleeter":
        raise NotImplementedError(
            "Spleeter backend is not wired yet; use --separation demucs or none."
        )
    raise ValueError(f"unknown separation mode: {mode!r}")


def separate(audio: np.ndarray, sr: int, mode: str = "demucs") -> np.ndarray:
    """Back-compat helper: return only the pitched instrumental mono mix."""
    return separate_stems(audio, sr, mode).instrumental


def _separate_demucs(
    audio: np.ndarray, sr: int, model_name: str = "htdemucs", *, keep_vocals: bool = False
) -> SeparationResult:
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
    vocals_np = (
        stem_mono("vocals").cpu().numpy().astype(np.float32)
        if (keep_vocals and "vocals" in names)
        else None
    )

    if sr != model_sr:
        from .load_audio import _resample_linear

        inst_np = _resample_linear(inst_np, model_sr, sr)
        if drums_np is not None:
            drums_np = _resample_linear(drums_np, model_sr, sr)
        if bass_np is not None:
            bass_np = _resample_linear(bass_np, model_sr, sr)
        if vocals_np is not None:
            vocals_np = _resample_linear(vocals_np, model_sr, sr)

    return SeparationResult(
        instrumental=inst_np, drums=drums_np, bass=bass_np, sr=sr, vocals=vocals_np
    )


def load_from_stems_dir(
    name: str,
    stems_dir: Path | str,
    target_sr: int,
    *,
    keep_vocals: bool = False,
) -> "SeparationResult | None":
    """Load pre-separated stems from *stems_dir*/*name*/ instead of running Demucs.

    Uses :func:`find_stems_folder` to locate the song directory with fuzzy name matching, so
    ``Goblins_Lair`` finds ``Goblin's Lair/``, ``PixelQuest(1)`` finds ``Pixel Quest/``, etc.

    Returns ``None`` when no matching folder is found (caller should fall back to Demucs).

    Looks for ``<song_dir>/<name> (<StemType>).<ext>`` (falling back to ``(<StemType>).<ext>``
    without the name prefix) where *StemType* is one of ``Synth``, ``Bass``, ``Drums``, ``FX``,
    ``Vocals``.

    * **Synth** → ``instrumental`` (mandatory — raises :exc:`FileNotFoundError` if folder found
      but the Synth stem file is absent).
    * **Bass**  → ``bass`` (``None`` when file missing).
    * **Drums** → ``drums`` (``None`` when file missing).
    * **FX**    → mixed into ``instrumental`` when present; ignored otherwise.
    * **Vocals** → ``vocals`` (only when *keep_vocals*; ``None`` when file missing).
    """
    from .load_audio import load_audio

    stems_dir = Path(stems_dir)
    song_dir = find_stems_folder(name, stems_dir)
    if song_dir is None:
        return None

    # Use both "<name> (StemType)" and plain "(StemType)" candidate names.
    folder_name = song_dir.name

    _AUDIO_EXTS_SET = set(_AUDIO_EXTS)

    def _find(stem_type: str) -> Path | None:
        # Priority 1: exact patterns (folder-name prefix or bare bracket prefix)
        for ext in _AUDIO_EXTS:
            for candidate in (
                song_dir / f"{folder_name} ({stem_type}){ext}",
                song_dir / f"({stem_type}){ext}",
            ):
                if candidate.is_file():
                    return candidate
        # Priority 2: any audio file in the folder whose stem contains (<stem_type>),
        # case-insensitive — handles files whose name prefix differs from the folder name.
        marker = f"({stem_type.lower()})"
        for f in sorted(song_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTS_SET \
                    and marker in f.stem.lower():
                return f
        return None

    # --- Synth (instrumental melodic content) ---
    synth_path = _find("Synth")
    if synth_path is None:
        raise FileNotFoundError(
            f"No Synth stem found in {song_dir}. "
            f"Expected e.g. '{folder_name} (Synth).mp3'."
        )
    instrumental, sr = load_audio(str(synth_path), target_sr)

    # --- FX: mix into instrumental when present (tonal effects add melodic colour) ---
    fx_path = _find("FX")
    if fx_path is not None:
        fx, _ = load_audio(str(fx_path), sr)
        n = min(instrumental.size, fx.size)
        blend = instrumental.copy()
        blend[:n] += fx[:n]
        if fx.size > instrumental.size:
            blend = np.concatenate([blend, fx[n:]])
        instrumental = blend

    # --- Drums ---
    drums: np.ndarray | None = None
    drums_path = _find("Drums")
    if drums_path is not None:
        drums, _ = load_audio(str(drums_path), sr)

    # --- Bass ---
    bass: np.ndarray | None = None
    bass_path = _find("Bass")
    if bass_path is not None:
        bass, _ = load_audio(str(bass_path), sr)

    # --- Vocals (kept only on request, for transcribing the sung melody as a lead) ---
    vocals: np.ndarray | None = None
    if keep_vocals:
        vocals_path = _find("Vocals")
        if vocals_path is not None:
            vocals, _ = load_audio(str(vocals_path), sr)

    return SeparationResult(
        instrumental=instrumental, drums=drums, bass=bass, sr=sr, vocals=vocals
    )
