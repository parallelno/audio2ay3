"""Diagnostic for a converted .ym: poly histogram, noise coverage, bass-channel activity.

Usage:  python build/diag.py path/to/song.ym
Run with the SAME interpreter that has audio2ay3 installed (e.g. the global Python312
on the validation machine, or `.\\.venv\\Scripts\\python.exe` locally).

Reference pro tune (Ay_Emul.ym, 8688 frames): poly {0:120,1:6,2:5044,3:3518}, noise ~38%.
After the dedicated-bass change, channel A should carry a steady, *moving* bass line:
- A% high on bass-heavy songs, and
- "channel A distinct tone periods" > 1 (a real bassline, not one stuck note).
"""

from __future__ import annotations

import sys
from collections import Counter

from audio2ay3.ymformat.ym_reader import load


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python build/diag.py <song.ym>")
        return 2

    song = load(argv[1])
    frames = song.frames
    n = len(frames)
    if n == 0:
        print("empty song")
        return 1

    poly: Counter[int] = Counter()
    noise_frames = 0
    chan_tone_on = [0, 0, 0]
    chan_a_periods: set[int] = set()
    amp_changes = [0, 0, 0]  # frames where a channel's tone-amplitude differs from the prior
    distinct_amps: set[int] = set()
    prev_amp = [-1, -1, -1]

    for fr in frames:
        mixer = int(fr[7])
        audible = 0
        for ch in range(3):
            tone_on = (mixer >> ch) & 1 == 0
            amp = int(fr[8 + ch])
            loud = (amp & 0x0F) > 0 or (amp & 0x10) != 0
            if tone_on and loud:
                audible += 1
                chan_tone_on[ch] += 1
                level = amp & 0x0F
                distinct_amps.add(level)
                if level != prev_amp[ch]:
                    amp_changes[ch] += 1
                prev_amp[ch] = level
                if ch == 0:
                    chan_a_periods.add(int(fr[0]) | (int(fr[1]) << 8))
            else:
                prev_amp[ch] = -1
        poly[audible] += 1
        if any((mixer >> (3 + ch)) & 1 == 0 for ch in range(3)):
            noise_frames += 1

    print(f"file: {argv[1]}")
    print(f"frames: {n}  duration: {song.duration_s:.1f}s")
    print(f"poly_hist (audible voices/frame): {dict(sorted(poly.items()))}")
    print(f"noise_frames: {noise_frames} ({100 * noise_frames / n:.1f}%)")
    print(
        f"tone-on A/B/C: {chan_tone_on}  "
        f"(A%={100 * chan_tone_on[0] / n:.1f}  "
        f"B%={100 * chan_tone_on[1] / n:.1f}  "
        f"C%={100 * chan_tone_on[2] / n:.1f})"
    )
    print(
        f"channel A (bass) distinct tone periods: {len(chan_a_periods)} "
        f"(want >1 = a moving bass line, not one stuck note)"
    )
    print(
        f"amp-change frames A/B/C: {amp_changes}  distinct amp levels: {len(distinct_amps)} "
        f"(envelope ON -> many; flat notes -> ~1-2)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
