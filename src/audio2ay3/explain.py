"""Human-readable diagnostics for an arranged :class:`YmSong` — the ``--explain`` artefact.

Reports register-level statistics (voice polyphony, per-channel utilisation, percussion
coverage, amplitude dynamics) so a conversion can be inspected without an external player. This
is the in-package, vectorised form of the diagnostics used throughout development: it answers
"what did the arranger actually emit?" straight from the legal register stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ymformat.model import YmSong

_AMP_ENV_BIT = 0x10  # amplitude bit 4: level comes from the envelope generator


@dataclass(frozen=True)
class SongStats:
    """Register-level summary of a :class:`YmSong`."""

    frames: int
    duration_s: float
    poly: tuple[int, ...]  # frame counts with 0..n_channels audible tone voices
    tone_on: tuple[int, ...]  # frames each channel sounds a tone (A, B, C, [D, E, F])
    noise_frames: int  # frames with the noise generator routed to any channel
    bass_distinct_periods: int  # distinct channel-A tone periods while audible
    amp_changes: tuple[int, ...]  # amplitude-change events per channel
    distinct_amp_levels: int  # how many of the 16 amplitude steps are used


def song_stats(song: YmSong) -> SongStats:
    """Compute register-level diagnostics for *song* (empty-song safe).

    Reads every chip: a dual-AY song reports across all six channels (poly up to 6, one tone-on /
    amp-change entry per channel), while a single chip keeps the historical three-channel shape.
    """
    frames = np.asarray(song.frames, dtype=np.int64)
    n = int(frames.shape[0])
    n_chips = max(1, int(getattr(song, "n_chips", 1)))
    n_ch = 3 * n_chips
    if n == 0:
        zero_ch = (0,) * n_ch
        return SongStats(0, 0.0, (0,) * (n_ch + 1), zero_ch, 0, 0, zero_ch, 0)

    width = frames.shape[1] // n_chips
    audible_cols: list[np.ndarray] = []
    amp_cols: list[np.ndarray] = []
    noise_cols: list[np.ndarray] = []
    for c in range(n_chips):
        off = c * width
        mixer = frames[:, off + 7]
        for ch in range(3):
            # Mixer bits 0-2 disable tone A/B/C, bits 3-5 disable noise A/B/C (0 = enabled).
            tone_on = ((mixer >> ch) & 1) == 0
            noise_on = ((mixer >> (3 + ch)) & 1) == 0
            amp_level = frames[:, off + 8 + ch] & 0x0F
            env_bit = (frames[:, off + 8 + ch] & _AMP_ENV_BIT) != 0
            # A channel sounds when its tone is enabled and it is not silent (level>0 or env).
            audible_cols.append(tone_on & ((amp_level > 0) | env_bit))
            amp_cols.append(amp_level)
            noise_cols.append(noise_on)
    audible = np.stack(audible_cols, axis=1)  # (n, n_ch)
    amp_level = np.stack(amp_cols, axis=1)
    noise_on = np.stack(noise_cols, axis=1)

    voices_per_frame = audible.sum(axis=1)
    poly = tuple(int(np.count_nonzero(voices_per_frame == k)) for k in range(n_ch + 1))
    tone_on_counts = tuple(int(audible[:, ch].sum()) for ch in range(n_ch))
    noise_frames = int(np.count_nonzero(np.any(noise_on, axis=1)))

    # Bass (channel A of chip 0) distinct tone periods while audible: >1 means a moving line.
    a_period = frames[:, 0] | (frames[:, 1] << 8)
    a_audible = audible[:, 0]
    bass_distinct = int(np.unique(a_period[a_audible]).size) if a_audible.any() else 0

    # Amplitude dynamics: a "change" is the first audible frame after silence or a level change
    # while continuously audible; reset across silence so re-onsets count (mirrors the ear).
    amp_changes: list[int] = []
    distinct_amps: set[int] = set()
    for ch in range(n_ch):
        aud = audible[:, ch]
        eff = np.where(aud, amp_level[:, ch], -1)
        changed = np.empty(n, dtype=bool)
        changed[0] = bool(aud[0])
        changed[1:] = aud[1:] & (eff[1:] != eff[:-1])
        amp_changes.append(int(changed.sum()))
        distinct_amps.update(int(v) for v in np.unique(amp_level[:, ch][aud]))

    return SongStats(
        frames=n,
        duration_s=float(song.duration_s),
        poly=poly,
        tone_on=tone_on_counts,
        noise_frames=noise_frames,
        bass_distinct_periods=bass_distinct,
        amp_changes=tuple(amp_changes),
        distinct_amp_levels=len(distinct_amps),
    )


def describe_song(song: YmSong) -> str:
    """Format :func:`song_stats` as a human-readable multi-line report."""
    s = song_stats(song)
    n = s.frames
    if n == 0:
        return "explain: empty song (0 frames)"

    def pct(x: int) -> float:
        return 100.0 * x / n

    labels = "/".join("ABCDEFGH"[: len(s.tone_on)])
    poly_str = " ".join(f"{k}={s.poly[k]}" for k in range(len(s.poly)))
    tone_str = " / ".join(f"{pct(t):.1f}%" for t in s.tone_on)
    return "\n".join(
        [
            f"explain: {n} frames, {s.duration_s:.1f}s",
            f"  polyphony (audible voices/frame): {poly_str}",
            f"  tone-on {labels}: {tone_str}",
            f"  noise frames: {s.noise_frames} ({pct(s.noise_frames):.1f}%)",
            f"  bass (ch A) distinct tone periods: {s.bass_distinct_periods}",
            f"  amp-change frames {labels}: {list(s.amp_changes)}  "
            f"distinct amp levels: {s.distinct_amp_levels}",
        ]
    )
