"""Normalised AY-3-8910 DAC amplitude table.

16 fixed-amplitude levels (0..15) mapped to a non-linear, roughly logarithmic output, using a
device-measured (MAME-style) curve. The encoder maps perceptual loudness to these levels by
nearest-amplitude lookup; the emulator uses them to render.
"""

from __future__ import annotations

import numpy as np

AY_DAC = np.array(
    [
        0.0000, 0.0076, 0.0110, 0.0158,
        0.0231, 0.0344, 0.0519, 0.0764,
        0.1170, 0.1632, 0.2392, 0.3536,
        0.5043, 0.6261, 0.8071, 1.0000,
    ],
    dtype=np.float64,
)
