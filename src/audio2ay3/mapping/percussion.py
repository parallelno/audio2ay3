"""Render drum hits onto channel C (skeleton recipes).

Percussion is coarse by nature. Snares and hats are short amplitude decays on the shared noise
generator (bright hiss for hats, mid for snares). The kick is instead a fast downward low-tone
sweep, because the noise generator has almost no sub-bass energy — a noise "kick" is a thin
click with no thump. We route hits to channel C and briefly steal it, which is the "trivial
mapping" the roadmap calls for at this phase — collision refinement comes later.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.model import Percussion
from ..encode.quantize import scale_amplitude, seconds_to_frame
from ..encode.register_stream import RegisterStreamBuilder

PERCUSSION_CHANNEL = 2  # channel C

# Detected drum velocities are erratic and frequently low (onset-strength normalisation pushes
# most hits well below the loudest). Floor them so every hit lands strong and consistent —
# matching the prominent, steady drums of typical sources — rather than flickering in and out.
_MIN_DRUM_SCALE = 0.7


@dataclass(frozen=True)
class _Recipe:
    decay: tuple[int, ...]  # amplitude per frame from the onset
    noise_period: int | None = None  # shared-noise centre; None for a tonal (noiseless) hit
    tone_sweep: tuple[int, ...] = ()  # per-frame tone periods (low -> lower) for a tonal body


# Snare/hat are pure noise (larger noise period -> lower centre frequency, f = clock / (16*NP)).
# The kick is instead a fast downward tone sweep (~158 -> 54 Hz): the noise generator has almost
# no sub-bass energy, so a noise "kick" reads as a thin click with no low-end thump.
_RECIPES: dict[str, _Recipe] = {
    "kick": _Recipe(decay=(15, 13, 8, 3), tone_sweep=(700, 1150, 1650, 2050)),
    "snare": _Recipe(decay=(15, 9, 4), noise_period=14),
    "hat": _Recipe(decay=(12, 4), noise_period=2),
}


def percussion_busy_frames(
    percussion: list[Percussion], frame_rate_hz: int, n_frames: int
) -> list[bool]:
    """Frames on which a drum hit's decay occupies (and overwrites) the percussion channel.

    Each hit steals channel C for the length of its recipe decay, clobbering any melodic note
    the arranger placed there. This exposes that footprint so the contention diagnostic can count
    melody lost to drums without re-deriving the (private) recipe decay lengths.
    """
    busy = [False] * n_frames
    for hit in percussion:
        recipe = _RECIPES.get(hit.kind, _RECIPES["snare"])
        onset = seconds_to_frame(hit.onset_s, frame_rate_hz)
        for i in range(len(recipe.decay)):
            f = onset + i
            if 0 <= f < n_frames:
                busy[f] = True
    return busy


def apply_percussion(
    builder: RegisterStreamBuilder,
    percussion: list[Percussion],
    frame_rate_hz: int,
    n_frames: int,
    *,
    channel: int = PERCUSSION_CHANNEL,
) -> None:
    """Overlay drum hits onto *builder*, stealing *channel* for each hit's decay."""
    for hit in percussion:
        recipe = _RECIPES.get(hit.kind, _RECIPES["snare"])
        onset = seconds_to_frame(hit.onset_s, frame_rate_hz)
        scale = max(_MIN_DRUM_SCALE, min(1.0, hit.velocity))
        for i, level in enumerate(recipe.decay):
            f = onset + i
            if f < 0 or f >= n_frames:
                continue
            # Scale in the DAC's logarithmic domain. A linear ``level * scale`` crushes soft
            # hits into near-silence (the 16 levels are log-spaced), which made most detected
            # drums inaudible; scale_amplitude keeps them prominent at their true loudness.
            amp = scale_amplitude(level, scale)
            if recipe.tone_sweep:
                # Tonal kick: a fast downward low-frequency sweep gives a real thump with
                # genuine sub-bass energy the noise generator cannot produce.
                tp = recipe.tone_sweep[min(i, len(recipe.tone_sweep) - 1)]
                builder.set_tone(f, channel, tp, amp)  # sets amplitude + enables tone
            else:
                builder.disable_tone(f, channel)  # let the noise read cleanly
                if recipe.noise_period is not None:
                    builder.enable_noise(f, channel, recipe.noise_period)
                builder.set_amplitude(f, channel, amp)
