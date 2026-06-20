"""Render drum hits onto the shared noise generator (skeleton recipes).

The AY has a single noise source, so percussion is coarse by nature. Each hit is a short
amplitude decay on a chosen channel with a per-kind noise period: a low rumble for kicks, a
bright hiss for hats, mid for snares. We route hits to channel C and briefly steal its tone,
which is the "trivial mapping" the roadmap calls for at this phase — collision refinement and
envelope-driven drums come later.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.model import Percussion
from ..encode.quantize import scale_amplitude, seconds_to_frame
from ..encode.register_stream import RegisterStreamBuilder

PERCUSSION_CHANNEL = 2  # channel C


@dataclass(frozen=True)
class _Recipe:
    noise_period: int
    decay: tuple[int, ...]  # amplitude per frame from the onset


# Larger noise period -> lower centre frequency (f = clock / (16 * NP)).
_RECIPES: dict[str, _Recipe] = {
    "kick": _Recipe(noise_period=28, decay=(15, 11, 6, 2)),
    "snare": _Recipe(noise_period=14, decay=(15, 9, 4)),
    "hat": _Recipe(noise_period=2, decay=(12, 4)),
}


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
        scale = max(0.0, min(1.0, hit.velocity))
        for i, level in enumerate(recipe.decay):
            f = onset + i
            if f < 0 or f >= n_frames:
                continue
            builder.disable_tone(f, channel)  # let the noise read cleanly
            builder.enable_noise(f, channel, recipe.noise_period)
            # Scale in the DAC's logarithmic domain. A linear ``level * scale`` crushes soft
            # hits into near-silence (the 16 levels are log-spaced), which made most detected
            # drums inaudible; scale_amplitude keeps them prominent at their true loudness.
            builder.set_amplitude(f, channel, scale_amplitude(level, scale))
