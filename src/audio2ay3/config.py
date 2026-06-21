"""Typed configuration for the chip target and a conversion/render run.

The analysis defaults are neural (Demucs separation + Basic Pitch transcription); there is no
classical-DSP note-detection path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ChipConfig:
    """Describes the target PSG configuration."""

    master_clock_hz: int = 1_773_400  # ZX Spectrum default
    frame_rate_hz: int = 50
    n_chips: int = 1  # 1 or 2 (dual-AY)
    tone_channels: int = 3  # per chip

    @property
    def total_tone_channels(self) -> int:
        return self.n_chips * self.tone_channels


@dataclass(frozen=True)
class AmpEnvelope:
    """Per-note **software** amplitude shape (attack / decay / sustain).

    The arranger writes this straight to the amplitude registers (R8-R10) every frame, so each
    note gets an independent attack and decay that the chip reproduces verbatim. This is the
    primary timbre/decay tool and is deliberately *not* the AY's shared hardware envelope
    generator (which is a single contended resource that's hard to tame); that path stays opt-in.

    Times are in 50 Hz frames (one frame = 20 ms). ``sustain`` is a fraction of the note's peak
    amplitude: ``1.0`` means a flat note (no decay), lower values give a plucky fall-off.
    """

    enabled: bool = True
    attack_frames: int = 0  # frames to ramp up to peak at the onset (0 = instant strike)
    decay_frames: int = 10  # frames to fall from peak to the sustain level (~200 ms)
    sustain: float = 0.6  # sustain level as a fraction of peak (1.0 = no decay)

    def level(self, age_frames: int, peak: int) -> int:
        """Amplitude (0..*peak*) for a note *age_frames* into its life at the given *peak*."""
        if peak <= 0:
            return 0
        if not self.enabled:
            return peak
        if self.attack_frames > 0 and age_frames < self.attack_frames:
            # Linear attack ramp; never below 1 so the very first frame is audible.
            return max(1, round(peak * (age_frames + 1) / (self.attack_frames + 1)))
        sustain_level = max(1, round(peak * self.sustain))
        decay_age = age_frames - self.attack_frames
        if self.decay_frames <= 0 or decay_age >= self.decay_frames:
            return sustain_level
        # Linear decay from peak down to the sustain level over decay_frames.
        return max(
            sustain_level,
            round(peak - (peak - sustain_level) * decay_age / self.decay_frames),
        )

    def factor(self, age_frames: int) -> float:
        """Attack/decay *shape* in ``[sustain, 1.0]`` for a note *age_frames* old (1.0 = full).

        The amplitude-domain counterpart of :meth:`level`: the arranger multiplies a note's
        source loudness contour by this so every note keeps a struck attack-and-decay shape even
        when the (whole-stem) contour is flat — which is what otherwise made dense passages sound
        like a sustained organ rather than individual piano notes.
        """
        if not self.enabled:
            return 1.0
        if self.attack_frames > 0 and age_frames < self.attack_frames:
            return (age_frames + 1) / (self.attack_frames + 1)
        decay_age = age_frames - self.attack_frames
        if self.decay_frames <= 0 or decay_age >= self.decay_frames:
            return self.sustain
        return 1.0 - (1.0 - self.sustain) * decay_age / self.decay_frames


@dataclass(frozen=True)
class Vibrato:
    """Idiomatic pitch vibrato for sustained, expressive instruments.

    MT3 gives no expression data, so we *synthesise* a small pitch LFO from the instrument
    identity: flutes, strings, reeds, organs and synth leads idiomatically vibrato, and a few
    cents of wobble is what makes a bare AY square read as a living tone instead of a dead
    oscillator. The wobble ramps in after the attack so onsets stay clean and in tune.

    Opt-in: ``enabled`` defaults to ``False`` (it can colour timbres in ways that hurt some
    sources); turn it on per run when the material benefits.
    """

    enabled: bool = False
    rate_hz: float = 6.0  # LFO frequency (musical vibrato sits ~5-7 Hz)
    depth_cents: float = 22.0  # peak pitch deviation either side of the note
    onset_delay_frames: int = 3  # hold a clean, in-tune attack before the wobble starts
    ramp_frames: int = 5  # frames to grow the depth from 0 to full after the delay

    def cents(self, age_frames: int, frame_rate_hz: int) -> float:
        """Signed pitch offset (in cents) for a note *age_frames* into its life."""
        if not self.enabled or age_frames < self.onset_delay_frames or frame_rate_hz <= 0:
            return 0.0
        grown = age_frames - self.onset_delay_frames
        ramp = 1.0 if self.ramp_frames <= 0 else min(1.0, (grown + 1) / self.ramp_frames)
        phase = 2.0 * math.pi * self.rate_hz * age_frames / frame_rate_hz
        return self.depth_cents * ramp * math.sin(phase)


@dataclass(frozen=True)
class RunConfig:
    """End-to-end run configuration."""

    chip: ChipConfig = field(default_factory=ChipConfig)
    use_gpu: bool = True  # auto-falls back to CPU
    threads: int = 0  # 0 = auto
    # Neural analysis stack (no DSP option by design):
    separation: Literal["demucs", "spleeter", "none"] = "demucs"
    transcription: Literal["basic-pitch", "mt3", "onsets-frames"] = "basic-pitch"
    render_sr: int = 44_100
    oversample: int = 2
    mp3_bitrate_kbps: int = 192
    seed: int = 0
    amp_envelope: AmpEnvelope = field(default_factory=AmpEnvelope)
    # Idiomatic pitch vibrato for expressive instruments (opt-in: see Vibrato.enabled).
    vibrato: Vibrato = field(default_factory=Vibrato)
    # Breathy noise chiff at the attack of wind voices (opt-in: off by default).
    breath: bool = False
    # Cycle squeezed chord tones on one channel instead of dropping them when more notes sound
    # at once than there are free tone channels (the classic chiptune "arpeggio chord").
    # Opt-in: off by default.
    arpeggio: bool = False
