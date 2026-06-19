"""Tests for percussion detection: onset finding + kick/snare/hat classification.

The pure ``_classify`` logic runs everywhere; the detector tests are gated on ``librosa``
(the neural extra) and on the drum-loop sample being present, so they skip cleanly on a
minimal install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio2ay3.analysis.percussion_detect import _classify, detect_percussion

DRUM_LOOP = Path(__file__).resolve().parents[1] / "samples" / "short" / "03_drum_loop.wav"


def test_classify_buckets_by_low_band_then_brightness():
    # Bass-dominant energy -> kick, even with a fairly bright centroid.
    assert _classify(centroid_hz=1800.0, low_ratio=0.40) == "kick"
    # Mid brightness with little low end -> snare.
    assert _classify(centroid_hz=2000.0, low_ratio=0.10) == "snare"
    # Bright with little low end -> hat.
    assert _classify(centroid_hz=5000.0, low_ratio=0.05) == "hat"


def test_detect_percussion_empty_or_silent_returns_no_hits():
    pytest.importorskip("librosa")
    assert detect_percussion(np.zeros(0, dtype=np.float32), 22050) == []
    assert detect_percussion(np.zeros(4096, dtype=np.float32), 22050) == []


@pytest.mark.skipif(not DRUM_LOOP.exists(), reason="drum-loop sample not present")
def test_detect_percussion_finds_varied_ordered_hits_on_drum_loop():
    librosa = pytest.importorskip("librosa")
    y, sr = librosa.load(str(DRUM_LOOP), sr=22050, mono=True)
    hits = detect_percussion(y, sr)

    assert len(hits) >= 6  # a 4 s drum loop has plenty of onsets
    duration = len(y) / sr
    for h in hits:
        assert h.kind in ("kick", "snare", "hat")
        assert 0.0 <= h.onset_s <= duration
        assert 0.3 <= h.velocity <= 1.0
    # Onsets come out time-ordered and cover more than one drum type.
    assert [h.onset_s for h in hits] == sorted(h.onset_s for h in hits)
    assert len({h.kind for h in hits}) >= 2
