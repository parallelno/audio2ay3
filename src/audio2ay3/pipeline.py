"""End-to-end conversion pipeline: audio -> Transcription -> arrangement -> YmSong.

The neural front-end (``load_audio`` -> ``separate`` -> ``transcribe``) is the only part that
needs the heavy optional extras; :func:`arrange` and everything below it are deterministic and
unit-tested directly with synthetic transcriptions. ``preview`` simply renders the arranged song
through the same emulator that ``validate`` uses.
"""

from __future__ import annotations

from pathlib import Path

from .analysis import load_audio, separate, transcribe
from .analysis.model import Transcription
from .config import RunConfig
from .encode import RegisterStreamBuilder, quantize_tone, velocity_to_amplitude
from .encode.quantize import frames_for_duration
from .mapping import allocate_voices, apply_percussion
from .ymformat.model import YmSong


def arrange(tr: Transcription, cfg: RunConfig, name: str = "") -> YmSong:
    """Turn a transcription into a hardware-legal :class:`YmSong` (no neural deps)."""
    clock = cfg.chip.master_clock_hz
    frame_rate = cfg.chip.frame_rate_hz

    end_s = max(
        tr.duration_s,
        max((n.offset_s for n in tr.notes), default=0.0),
        # Pad a short decay tail only when percussion actually exists.
        (max(p.onset_s for p in tr.percussion) + 0.1) if tr.percussion else 0.0,
    )
    n_frames = frames_for_duration(end_s, frame_rate)

    builder = RegisterStreamBuilder(n_frames)
    assignment = allocate_voices(tr.notes, frame_rate, n_frames)
    for f in range(n_frames):
        for ch in range(3):
            voice = assignment[f][ch]
            if voice is None:
                continue
            tone_period = quantize_tone(voice.pitch_hz, clock)
            amplitude = velocity_to_amplitude(voice.velocity)
            if tone_period > 0 and amplitude > 0:
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
    instrumental = separate(audio, sr, cfg.separation)
    tr = transcribe(instrumental, sr, cfg.transcription, cfg.chip.frame_rate_hz)
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
