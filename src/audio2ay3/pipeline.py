"""End-to-end conversion pipeline: audio -> Transcription -> arrangement -> YmSong.

The neural front-end (``load_audio`` -> ``separate`` -> ``transcribe``) is the only part that
needs the heavy optional extras; :func:`arrange` and everything below it are deterministic and
unit-tested directly with synthetic transcriptions. ``preview`` simply renders the arranged song
through the same emulator that ``validate`` uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import (
    attach_amp_contours,
    detect_percussion,
    load_audio,
    separate_stems,
    transcribe,
)
from .analysis.model import Transcription
from .config import RunConfig
from .encode import (
    RegisterStreamBuilder,
    quantize_tone,
    scale_amplitude,
    velocity_to_amplitude,
)
from .encode.quantize import TONE_PERIOD_MIN, frames_for_duration
from .mapping import (
    allocate_voices,
    apply_percussion,
    is_breath_program,
    is_sustained_program,
    is_vibrato_program,
    percussion_busy_frames,
    place_bass,
)
from .progress import NullProgress, Progress
from .ymformat.model import YmSong

# A breathy wind voice gets a short noise "chiff" at each note's attack: bright air for the first
# couple of frames, then a clean tone. It yields to drums, which own the shared noise generator.
_BREATH_NOISE_PERIOD = 4  # small period -> high-frequency air hiss
_BREATH_ATTACK_FRAMES = 2


def arrange(tr: Transcription, cfg: RunConfig, name: str = "") -> YmSong:
    """Turn a transcription into a hardware-legal :class:`YmSong` (no neural deps)."""
    clock = cfg.chip.master_clock_hz
    frame_rate = cfg.chip.frame_rate_hz
    n_chips = cfg.chip.n_chips
    tpc = cfg.chip.tone_channels  # tone channels per chip (3 on a real AY)
    n_channels = cfg.chip.total_tone_channels  # 3 for a single AY, 6 for dual-AY

    end_s = max(
        tr.duration_s,
        max((n.offset_s for n in tr.notes), default=0.0),
        max((n.offset_s for n in tr.bass_notes), default=0.0),
        # Pad a short decay tail only when percussion actually exists.
        (max(p.onset_s for p in tr.percussion) + 0.1) if tr.percussion else 0.0,
    )
    n_frames = frames_for_duration(end_s, frame_rate)

    # One independent 16-register stream per chip; a global channel routes to (chip, local ch).
    builders = [RegisterStreamBuilder(n_frames) for _ in range(n_chips)]

    def route(global_ch: int) -> tuple[RegisterStreamBuilder, int]:
        return builders[global_ch // tpc], global_ch % tpc

    # Bass owns a dedicated channel; the melodic allocator fills the channels left free. With a
    # second chip the melody spreads over four channels instead of one or two, which is the win.
    bass_voices, reserved = place_bass(tr.bass_notes, frame_rate, n_frames)
    assignment = allocate_voices(
        tr.notes, frame_rate, n_frames, reserved=reserved,
        n_channels=n_channels, arpeggiate=cfg.arpeggio,
    )
    for f in range(n_frames):
        ch = reserved[f]
        if ch is not None:
            assignment[f][ch] = bass_voices[f]

    # Per-note amplitude shaping. When a voice carries a source-derived loudness for this frame
    # (amp_scale), follow it so the note keeps the original's character; otherwise fall back to
    # the synthetic envelope. Both are gated by amp_envelope.enabled (off -> flat amplitude).
    env = cfg.amp_envelope
    vib = cfg.vibrato
    # Frames a drum decay owns the shared noise generator, so a breath chiff defers to it.
    drum_busy = percussion_busy_frames(tr.percussion, frame_rate, n_frames)
    cur_note: list[int | None] = [None] * n_channels
    age = [0] * n_channels
    for f in range(n_frames):
        for ch in range(n_channels):
            voice = assignment[f][ch]
            if voice is None:
                cur_note[ch] = None
                continue
            builder, local_ch = route(ch)
            if voice.note_id != cur_note[ch]:
                cur_note[ch] = voice.note_id
                age[ch] = 0
            else:
                age[ch] += 1
            tone_period = quantize_tone(voice.pitch_hz, clock)
            if tone_period <= 0:
                continue
            # Vibrato: a small pitch LFO on idiomatically-expressive instruments (flute/strings/
            # reed/organ/synth lead) makes a bare square read as a living tone. Period is the
            # inverse of frequency, so a sharp-by-c-cents frame divides the period accordingly.
            if vib.enabled and is_vibrato_program(voice.program):
                cents = vib.cents(age[ch], frame_rate)
                if cents:
                    tone_period = max(
                        TONE_PERIOD_MIN, round(tone_period / (2.0 ** (cents / 1200.0)))
                    )
            # The allocator already decided this voice should sound; never let velocity
            # rounding silence it — floor a placed note to the quietest audible amplitude.
            peak = max(1, velocity_to_amplitude(voice.velocity))
            sustained = is_sustained_program(voice.program)
            if env.enabled and voice.amp_scale is not None:
                if age[ch] == 0:
                    # Strike each note's first frame at its full (velocity-scaled) peak so every
                    # onset has a sharp attack. Without this, the smoothed source contour blurs
                    # fast repeated notes into one sustained tone (few strong onsets).
                    level = peak
                elif sustained:
                    # Held instrument (strings/brass/reed/pipe/organ/synth lead+pad): follow the
                    # source loudness but never impose the synthetic struck decay, so a legato
                    # line stays connected instead of fragmenting into short, isolated notes.
                    level = max(1, scale_amplitude(peak, voice.amp_scale))
                else:
                    # Shape the source loudness by the synthetic attack/decay so every note keeps
                    # a struck character even where the (whole-stem) contour is flat — otherwise
                    # dense passages collapse to a lifeless sustain. Applied in the DAC's
                    # logarithmic domain so below-peak frames aren't crushed into near-silence.
                    shaped = voice.amp_scale * env.factor(age[ch])
                    level = max(1, scale_amplitude(peak, shaped))
            elif env.enabled and sustained:
                # Held instrument with no source contour: hold flat at the note's peak so it
                # sustains for its whole length rather than decaying like a plucked note.
                level = peak
            else:
                level = env.level(age[ch], peak)
            builder.set_tone(f, local_ch, tone_period, level)
            # Breath: a short noise chiff at the attack of flute/pipe/reed voices imitates the
            # instrument's air, but only when no drum is using the shared noise generator.
            if (
                cfg.breath
                and is_breath_program(voice.program)
                and age[ch] < _BREATH_ATTACK_FRAMES
                and not drum_busy[f]
            ):
                builder.enable_noise(f, local_ch, _BREATH_NOISE_PERIOD)

    # Percussion steals the last tone channel: channel C on a single chip (the historical
    # placement), or the second chip's channel C on dual-AY, isolating drums from the melody.
    perc_builder, perc_local = route(n_channels - 1)
    apply_percussion(
        perc_builder, tr.percussion, frame_rate, n_frames, channel=perc_local
    )

    frames = (
        builders[0].finish()
        if n_chips == 1
        else np.concatenate([b.finish() for b in builders], axis=1)
    )
    return YmSong(
        frames=frames,
        master_clock=clock,
        frame_rate=frame_rate,
        n_chips=n_chips,
        version="YM6",
        name=name,
        author="audio2ay3",
        comment="Converted by audio2ay3",
    )


def convert(
    path: str,
    cfg: RunConfig,
    *,
    trace: list[Transcription] | None = None,
    progress: Progress | None = None,
) -> YmSong:
    """Full neural conversion: audio file -> arranged :class:`YmSong`.

    When *trace* is given, the pre-arrange :class:`Transcription` is appended to it, so callers
    (e.g. ``--explain``) can inspect the musical demand without re-running the neural stack.

    *progress* (optional) is advanced once per pipeline stage so the CLI can draw a bar; pass
    ``None`` (the default) for silent library use.
    """
    p = progress or NullProgress()
    tr = _build_transcription(path, cfg, p)
    if trace is not None:
        trace.append(tr)
    p.step("arranging")
    return arrange(tr, cfg, name=Path(path).stem)


def _build_transcription(path: str, cfg: RunConfig, progress: Progress) -> Transcription:
    """Run the neural front-end into a :class:`Transcription` (everything before ``arrange``)."""
    progress.step("loading audio")
    audio, sr = load_audio(path, cfg.render_sr)
    if cfg.transcription in ("mt3", "yourmt3"):
        progress.step("transcribing (multitrack)")
        return _build_transcription_multitrack(audio, sr, cfg)
    # Mirrors _stage_labels: the "separating stems" / "detecting percussion" stages only exist
    # when a real separation runs (mode != "none" yields bass + drum stems).
    if cfg.separation != "none":
        progress.step("separating stems")
    stems = separate_stems(audio, sr, cfg.separation)
    progress.step("transcribing")
    tr = transcribe(stems.instrumental, stems.sr, cfg.transcription, cfg.chip.frame_rate_hz)
    # Follow each note's real loudness shape from its own stem so held notes sustain and plucks
    # decay like the original; skipped under --no-amp-envelope, which wants flat amplitude.
    if cfg.amp_envelope.enabled:
        tr.notes = attach_amp_contours(
            tr.notes, stems.instrumental, stems.sr, cfg.chip.frame_rate_hz
        )
    if stems.drums is not None:
        progress.step("detecting percussion")
        # Reference the whole-track RMS so a drum-less stem's residual bleed can't fire phantom
        # hits (its velocities are otherwise normalised against its own near-silent dynamics).
        ref_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        tr.percussion = detect_percussion(stems.drums, stems.sr, reference_rms=ref_rms)
    if stems.bass is not None:
        # Transcribe the isolated bass stem on its own; place_bass monophonises it later.
        bass_tr = transcribe(stems.bass, stems.sr, cfg.transcription, cfg.chip.frame_rate_hz)
        bass_notes = bass_tr.notes
        if cfg.amp_envelope.enabled:
            bass_notes = attach_amp_contours(
                bass_notes, stems.bass, stems.sr, cfg.chip.frame_rate_hz
            )
        tr.bass_notes = bass_notes
    return tr


def _build_transcription_multitrack(audio: np.ndarray, sr: int, cfg: RunConfig) -> Transcription:
    """MT3 / YourMT3+ path: one multitrack pass yields notes, bass, and drums together.

    Both backends emit General-MIDI note events for every instrument at once, so unlike the Basic
    Pitch path there are no Demucs stems to isolate: :func:`transcribe` already routes drums to
    percussion and the GM bass family to ``bass_notes``. Loudness contours therefore follow the
    full mix (the only signal available), still gated by ``--no-amp-envelope``.
    """
    tr = transcribe(
        audio, sr, cfg.transcription, cfg.chip.frame_rate_hz, yourmt3_model=cfg.yourmt3_model
    )
    if cfg.amp_envelope.enabled:
        tr.notes = attach_amp_contours(tr.notes, audio, sr, cfg.chip.frame_rate_hz)
        tr.bass_notes = attach_amp_contours(tr.bass_notes, audio, sr, cfg.chip.frame_rate_hz)
    return tr


def preview(
    path: str,
    out_path: str,
    cfg: RunConfig,
    *,
    max_seconds: float | None = None,
    trace: list[Transcription] | None = None,
    progress: Progress | None = None,
) -> YmSong:
    """Convert *path* and render the result to audio at *out_path*; returns the song."""
    from .render import Renderer

    p = progress or NullProgress()
    song = convert(path, cfg, trace=trace, progress=p)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    p.step("rendering audio")
    Renderer(render_sr=cfg.render_sr, oversample=cfg.oversample).render_to_file(
        song, out_path, bitrate_kbps=cfg.mp3_bitrate_kbps, max_seconds=max_seconds
    )
    return song


def stage_labels(cfg: RunConfig, *, render: bool) -> list[str]:
    """The ordered stage labels :func:`convert`/:func:`preview` will emit for *cfg*.

    The single source of truth for the progress bar's length: it mirrors the branching in
    :func:`_build_transcription` (separation/percussion stages exist only when a real separation
    runs) so the emitted ``step`` count always matches ``len(stage_labels(...))``.
    """
    labels = ["loading audio"]
    if cfg.transcription in ("mt3", "yourmt3"):
        labels.append("transcribing (multitrack)")
    else:
        if cfg.separation != "none":
            labels.append("separating stems")
        labels.append("transcribing")
        if cfg.separation != "none":
            labels.append("detecting percussion")
    labels.append("arranging")
    if render:
        labels.append("rendering audio")
    return labels


def progress_total(cfg: RunConfig, *, render: bool) -> int:
    """Number of progress steps :func:`convert` (``render=False``)/:func:`preview` will report."""
    return len(stage_labels(cfg, render=render))
