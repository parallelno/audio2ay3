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
    poly: tuple[int, int, int, int]  # frame counts with 0/1/2/3 audible tone voices
    tone_on: tuple[int, int, int]  # frames each channel A/B/C sounds a tone
    noise_frames: int  # frames with the noise generator routed to any channel
    bass_distinct_periods: int  # distinct channel-A tone periods while audible
    amp_changes: tuple[int, int, int]  # amplitude-change events per channel A/B/C
    distinct_amp_levels: int  # how many of the 16 amplitude steps are used


def song_stats(song: YmSong) -> SongStats:
    """Compute register-level diagnostics for *song* (empty-song safe)."""
    frames = np.asarray(song.frames, dtype=np.int64)
    n = int(frames.shape[0])
    if n == 0:
        return SongStats(0, 0.0, (0, 0, 0, 0), (0, 0, 0), 0, 0, (0, 0, 0), 0)

    mixer = frames[:, 7]
    # Mixer bits 0-2 disable tone A/B/C, bits 3-5 disable noise A/B/C (0 = enabled).
    tone_on = np.stack([((mixer >> ch) & 1) == 0 for ch in range(3)], axis=1)
    noise_on = np.stack([((mixer >> (3 + ch)) & 1) == 0 for ch in range(3)], axis=1)
    amp_level = frames[:, 8:11] & 0x0F
    env_bit = (frames[:, 8:11] & _AMP_ENV_BIT) != 0
    # A channel sounds when its tone is enabled and it is not silent (level > 0 or envelope).
    audible = tone_on & ((amp_level > 0) | env_bit)

    voices_per_frame = audible.sum(axis=1)
    poly = tuple(int(np.count_nonzero(voices_per_frame == k)) for k in range(4))
    tone_on_counts = tuple(int(audible[:, ch].sum()) for ch in range(3))
    noise_frames = int(np.count_nonzero(np.any(noise_on, axis=1)))

    # Bass (channel A) distinct tone periods while audible: >1 means a moving line, not one note.
    a_period = frames[:, 0] | (frames[:, 1] << 8)
    a_audible = audible[:, 0]
    bass_distinct = int(np.unique(a_period[a_audible]).size) if a_audible.any() else 0

    # Amplitude dynamics: a "change" is the first audible frame after silence or a level change
    # while continuously audible; reset across silence so re-onsets count (mirrors the ear).
    amp_changes: list[int] = []
    distinct_amps: set[int] = set()
    for ch in range(3):
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
        poly=(poly[0], poly[1], poly[2], poly[3]),
        tone_on=(tone_on_counts[0], tone_on_counts[1], tone_on_counts[2]),
        noise_frames=noise_frames,
        bass_distinct_periods=bass_distinct,
        amp_changes=(amp_changes[0], amp_changes[1], amp_changes[2]),
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

    return "\n".join(
        [
            f"explain: {n} frames, {s.duration_s:.1f}s",
            f"  polyphony (audible voices/frame): "
            f"0={s.poly[0]} 1={s.poly[1]} 2={s.poly[2]} 3={s.poly[3]}",
            f"  tone-on A/B/C: "
            f"{pct(s.tone_on[0]):.1f}% / {pct(s.tone_on[1]):.1f}% / {pct(s.tone_on[2]):.1f}%",
            f"  noise frames: {s.noise_frames} ({pct(s.noise_frames):.1f}%)",
            f"  bass (ch A) distinct tone periods: {s.bass_distinct_periods}",
            f"  amp-change frames A/B/C: {list(s.amp_changes)}  "
            f"distinct amp levels: {s.distinct_amp_levels}",
        ]
    )
