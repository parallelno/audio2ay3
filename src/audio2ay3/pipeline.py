"""End-to-end conversion pipeline: audio -> Transcription -> arrangement -> YmSong.

The neural front-end (``load_audio`` -> ``separate`` -> ``transcribe``) is the only part that
needs the heavy optional extras; :func:`arrange` and everything below it are deterministic and
unit-tested directly with synthetic transcriptions. ``preview`` simply renders the arranged song
through the same emulator that ``validate`` uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import detect_percussion, load_audio, separate_stems, transcribe
from .analysis.model import Transcription
from .config import RunConfig
from .encode import RegisterStreamBuilder, quantize_tone, velocity_to_amplitude
from .encode.quantize import frames_for_duration
from .mapping import allocate_voices, apply_percussion, place_bass
from .ymformat.model import YmSong


def arrange(tr: Transcription, cfg: RunConfig, name: str = "") -> YmSong:
    """Turn a transcription into a hardware-legal :class:`YmSong` (no neural deps)."""
    clock = cfg.chip.master_clock_hz
    frame_rate = cfg.chip.frame_rate_hz

    end_s = max(
        tr.duration_s,
        max((n.offset_s for n in tr.notes), default=0.0),
        max((n.offset_s for n in tr.bass_notes), default=0.0),
        # Pad a short decay tail only when percussion actually exists.
        (max(p.onset_s for p in tr.percussion) + 0.1) if tr.percussion else 0.0,
    )
    n_frames = frames_for_duration(end_s, frame_rate)

    builder = RegisterStreamBuilder(n_frames)
    # Bass owns a dedicated channel; the melodic allocator fills the channels left free.
    bass_voices, reserved = place_bass(tr.bass_notes, frame_rate, n_frames)
    assignment = allocate_voices(tr.notes, frame_rate, n_frames, reserved=reserved)
    for f in range(n_frames):
        ch = reserved[f]
        if ch is not None:
            assignment[f][ch] = bass_voices[f]

    for f in range(n_frames):
        for ch in range(3):
            voice = assignment[f][ch]
            if voice is None:
                continue
            tone_period = quantize_tone(voice.pitch_hz, clock)
            if tone_period <= 0:
                continue
            # The allocator already decided this voice should sound; never let velocity
            # rounding silence it — floor a placed note to the quietest audible amplitude.
            amplitude = max(1, velocity_to_amplitude(voice.velocity))
            builder.set_tone(f, ch, tone_period, amplitude)

    apply_percussion(builder, tr.percussion, frame_rate, n_frames)

    return YmSong(
        frames=builder.finish(),
        master_clock=clock,
        frame_rate=frame_rate,
        version="YM6",
        name=name,
        author="audio2ay3",
        comment="Converted by audio2ay3",
    )


def convert(path: str, cfg: RunConfig) -> YmSong:
    """Full neural conversion: audio file -> arranged :class:`YmSong`."""
    audio, sr = load_audio(path, cfg.render_sr)
    stems = separate_stems(audio, sr, cfg.separation)
    tr = transcribe(stems.instrumental, stems.sr, cfg.transcription, cfg.chip.frame_rate_hz)
    if stems.drums is not None:
        # Reference the whole-track RMS so a drum-less stem's residual bleed can't fire phantom
        # hits (its velocities are otherwise normalised against its own near-silent dynamics).
        ref_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        tr.percussion = detect_percussion(stems.drums, stems.sr, reference_rms=ref_rms)
    if stems.bass is not None:
        # Transcribe the isolated bass stem on its own; place_bass monophonises it later.
        bass_tr = transcribe(stems.bass, stems.sr, cfg.transcription, cfg.chip.frame_rate_hz)
        tr.bass_notes = bass_tr.notes
    return arrange(tr, cfg, name=Path(path).stem)


def preview(path: str, out_path: str, cfg: RunConfig, *, max_seconds: float | None = None) -> YmSong:
    """Convert *path* and render the result to audio at *out_path*; returns the song."""
    from .render import Renderer

    song = convert(path, cfg)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Renderer(render_sr=cfg.render_sr, oversample=cfg.oversample).render_to_file(
        song, out_path, bitrate_kbps=cfg.mp3_bitrate_kbps, max_seconds=max_seconds
    )
    return song
