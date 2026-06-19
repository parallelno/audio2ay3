"""Accurate-enough AY-3-8910 / YM2149 emulator.

A two-clock, fractional-counter model: the chip's tone/noise/envelope generators are advanced
by the number of chip cycles elapsed per (oversampled) output sample, toggling/stepping on
counter underflow. The inner loop is JIT-compiled with numba when available.

Pitch conventions (match the datasheet formulae used throughout the design):
    f_tone  = master_clock / (16 * TP)
    f_noise = master_clock / (16 * NP)
    f_env   = master_clock / (256 * EP)   (per envelope step)
"""

from __future__ import annotations

import numpy as np

from ..config import ChipConfig
from .volume_tables import AY_DAC

try:  # numba is a core dependency, but degrade gracefully if missing.
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba installed
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco


@njit(cache=True)
def _render_core(frames, master_clock, frame_rate, internal_sr, dac):
    """Render an (n_frames, 16) uint8 register array to mono PCM at ``internal_sr``."""
    n_frames = frames.shape[0]
    spf = internal_sr / frame_rate
    total = int(n_frames * spf + 0.5)
    out = np.empty(total, dtype=np.float32)

    # --- generator state (persists across frames) ---
    tone_cnt = np.zeros(3, dtype=np.float64)
    tone_ph = np.zeros(3, dtype=np.int64)
    period = np.ones(3, dtype=np.float64)
    tone_on = np.zeros(3, dtype=np.int64)
    noise_on = np.zeros(3, dtype=np.int64)
    use_env = np.zeros(3, dtype=np.int64)
    lev_fix = np.zeros(3, dtype=np.int64)

    noise_cnt = 0.0
    noise_period = 1.0
    lfsr = 1

    env_cnt = 0.0
    env_period = 1.0
    env_pos = 0
    env_attack = 0
    env_holding = 0
    env_zero = 0
    env_cont = 0
    env_att = 0
    env_alt = 0
    env_hold = 0

    # Per-sample increments (in chip cycles, accounting for each generator's prescaler).
    tone_tick = (master_clock / 8.0) / internal_sr  # /8 + toggle => f = clock/(16*TP)
    noise_tick = (master_clock / 16.0) / internal_sr
    env_tick = (master_clock / 256.0) / internal_sr

    oi = 0
    prev_b = 0
    for f in range(n_frames):
        tpa = ((frames[f, 1] & 0x0F) << 8) | frames[f, 0]
        tpb = ((frames[f, 3] & 0x0F) << 8) | frames[f, 2]
        tpc = ((frames[f, 5] & 0x0F) << 8) | frames[f, 4]
        if tpa == 0:
            tpa = 1
        if tpb == 0:
            tpb = 1
        if tpc == 0:
            tpc = 1
        period[0] = tpa
        period[1] = tpb
        period[2] = tpc

        npv = frames[f, 6] & 0x1F
        if npv == 0:
            npv = 1
        noise_period = float(npv)

        mixer = frames[f, 7]
        tone_on[0] = 1 if ((mixer >> 0) & 1) == 0 else 0
        tone_on[1] = 1 if ((mixer >> 1) & 1) == 0 else 0
        tone_on[2] = 1 if ((mixer >> 2) & 1) == 0 else 0
        noise_on[0] = 1 if ((mixer >> 3) & 1) == 0 else 0
        noise_on[1] = 1 if ((mixer >> 4) & 1) == 0 else 0
        noise_on[2] = 1 if ((mixer >> 5) & 1) == 0 else 0

        for c in range(3):
            amp = frames[f, 8 + c]
            use_env[c] = (amp >> 4) & 1
            lev_fix[c] = amp & 0x0F

        epv = (frames[f, 12] << 8) | frames[f, 11]
        if epv == 0:
            epv = 1
        env_period = float(epv)

        # R13 == 0xFF is the ST-Sound sentinel for "no write / do not retrigger".
        r13 = frames[f, 13]
        if r13 != 0xFF:
            shape = r13 & 0x0F
            env_cont = (shape >> 3) & 1
            env_att = (shape >> 2) & 1
            env_alt = (shape >> 1) & 1
            env_hold = shape & 1
            env_pos = 0
            env_attack = env_att
            env_holding = 0
            env_zero = 0

        next_b = int((f + 1) * spf + 0.5)
        n_samp = next_b - prev_b
        prev_b = next_b

        for _k in range(n_samp):
            # Tone generators
            for c in range(3):
                tone_cnt[c] += tone_tick
                while tone_cnt[c] >= period[c]:
                    tone_cnt[c] -= period[c]
                    tone_ph[c] ^= 1

            # Noise generator (17-bit LFSR, taps at bits 0 and 3)
            noise_cnt += noise_tick
            while noise_cnt >= noise_period:
                noise_cnt -= noise_period
                bit = (lfsr ^ (lfsr >> 3)) & 1
                lfsr = ((lfsr >> 1) | (bit << 16)) & 0x1FFFF
            noise_bit = lfsr & 1

            # Envelope generator (16-step ramp; shape from R13)
            env_cnt += env_tick
            while env_cnt >= env_period:
                env_cnt -= env_period
                if env_holding == 0:
                    env_pos += 1
                    if env_pos > 15:
                        if env_cont == 0:
                            env_holding = 1
                            env_pos = 15
                            env_zero = 1
                        else:
                            if env_hold == 1:
                                env_holding = 1
                                if env_alt == 1:
                                    env_attack ^= 1
                                env_pos = 15
                            else:
                                if env_alt == 1:
                                    env_attack ^= 1
                                env_pos = 0

            if env_zero == 1:
                env_level = 0
            elif env_attack == 1:
                env_level = env_pos
            else:
                env_level = 15 - env_pos

            # Mix the three channels (active-low tone/noise gating; disabled source -> 1)
            s = 0.0
            for c in range(3):
                t = tone_ph[c] if tone_on[c] == 1 else 1
                nz = noise_bit if noise_on[c] == 1 else 1
                g = t * nz
                if use_env[c] == 1:
                    lvl = env_level
                else:
                    lvl = lev_fix[c]
                s += dac[lvl] * g
            out[oi] = s / 3.0
            oi += 1

    return out


class Ay3Emulator:
    """Render YM register frames to mono PCM."""

    def __init__(self, chip: ChipConfig | None = None, render_sr: int = 44_100,
                 oversample: int = 2, dac: np.ndarray | None = None) -> None:
        self.chip = chip if chip is not None else ChipConfig()
        self.render_sr = int(render_sr)
        self.oversample = max(1, int(oversample))
        self.dac = AY_DAC if dac is None else np.asarray(dac, dtype=np.float64)

    def render_frames(self, frames: np.ndarray, master_clock: int,
                      frame_rate: int) -> np.ndarray:
        """Render an ``(n, 16)`` (or ``(n, <16)``) uint8 register array to PCM."""
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        if frames.ndim != 2:
            raise ValueError("frames must be a 2-D array of shape (n_frames, <=16)")
        if frames.shape[1] < 16:
            buf = np.zeros((frames.shape[0], 16), dtype=np.uint8)
            buf[:, : frames.shape[1]] = frames
            frames = buf
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)

        internal_sr = float(self.render_sr * self.oversample)
        raw = _render_core(frames, float(master_clock), float(frame_rate),
                           internal_sr, self.dac)
        if self.oversample > 1 and raw.size:
            n = (raw.shape[0] // self.oversample) * self.oversample
            raw = raw[:n].reshape(-1, self.oversample).mean(axis=1).astype(np.float32)
        return raw

    def render_song(self, song) -> np.ndarray:
        """Render a :class:`audio2ay3.ymformat.model.YmSong` to PCM."""
        return self.render_frames(song.frames, song.master_clock, song.frame_rate)
