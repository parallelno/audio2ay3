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
from .ymformat.model import YmSong

# A breathy wind voice gets a short noise "chiff" at each note's attack: bright air for the first
# couple of frames, then a clean tone. It yields to drums, which own the shared noise generator.
_BREATH_NOISE_PERIOD = 4  # small period -> high-frequency air hiss
_BREATH_ATTACK_FRAMES = 2


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
    assignment = allocate_voices(
        tr.notes, frame_rate, n_frames, reserved=reserved, arpeggiate=cfg.arpeggio
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
    cur_note: list[int | None] = [None, None, None]
    age = [0, 0, 0]
    for f in range(n_frames):
        for ch in range(3):
            voice = assignment[f][ch]
            if voice is None:
                cur_note[ch] = None
                continue
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
            builder.set_tone(f, ch, tone_period, level)
            # Breath: a short noise chiff at the attack of flute/pipe/reed voices imitates the
            # instrument's air, but only when no drum is using the shared noise generator.
            if (
                cfg.breath
                and is_breath_program(voice.program)
                and age[ch] < _BREATH_ATTACK_FRAMES
                and not drum_busy[f]
            ):
                builder.enable_noise(f, ch, _BREATH_NOISE_PERIOD)

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


def convert(
    path: str, cfg: RunConfig, *, trace: list[Transcription] | None = None
) -> YmSong:
    """Full neural conversion: audio file -> arranged :class:`YmSong`.

    When *trace* is given, the pre-arrange :class:`Transcription` is appended to it, so callers
    (e.g. ``--explain``) can inspect the musical demand without re-running the neural stack.
    """
    tr = _build_transcription(path, cfg)
    if trace is not None:
        trace.append(tr)
    return arrange(tr, cfg, name=Path(path).stem)


def _build_transcription(path: str, cfg: RunConfig) -> Transcription:
    """Run the neural front-end into a :class:`Transcription` (everything before ``arrange``)."""
    audio, sr = load_audio(path, cfg.render_sr)
    if cfg.transcription == "mt3":
        return _build_transcription_mt3(audio, sr, cfg)
    stems = separate_stems(audio, sr, cfg.separation)
    tr = transcribe(stems.instrumental, stems.sr, cfg.transcription, cfg.chip.frame_rate_hz)
    # Follow each note's real loudness shape from its own stem so held notes sustain and plucks
    # decay like the original; skipped under --no-amp-envelope, which wants flat amplitude.
    if cfg.amp_envelope.enabled:
        tr.notes = attach_amp_contours(
            tr.notes, stems.instrumental, stems.sr, cfg.chip.frame_rate_hz
        )
    if stems.drums is not None:
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


def _build_transcription_mt3(audio: np.ndarray, sr: int, cfg: RunConfig) -> Transcription:
    """MT3 path: one multitrack pass yields notes, bass, and drums together (no separation).

    MT3 emits General-MIDI note events for every instrument at once, so unlike the Basic Pitch
    path there are no Demucs stems to isolate: :func:`transcribe` already routes drums to
    percussion and the GM bass family to ``bass_notes``. Loudness contours therefore follow the
    full mix (the only signal available), still gated by ``--no-amp-envelope``.
    """
    tr = transcribe(audio, sr, "mt3", cfg.chip.frame_rate_hz)
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
) -> YmSong:
    """Convert *path* and render the result to audio at *out_path*; returns the song."""
    from .render import Renderer

    song = convert(path, cfg, trace=trace)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Renderer(render_sr=cfg.render_sr, oversample=cfg.oversample).render_to_file(
        song, out_path, bitrate_kbps=cfg.mp3_bitrate_kbps, max_seconds=max_seconds
    )
    return song
