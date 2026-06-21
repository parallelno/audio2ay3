"""Tests for source-separation dispatch (no torch/Demucs imported here).

The heavy Demucs path needs the ``[neural]`` extra, so these only exercise the pure dispatch:
the ``none`` passthrough, the friendly-mode -> Demucs-model map, and the error surfaces for the
unimplemented (``spleeter``) and unknown modes. Real separation is validated out-of-band.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from audio2ay3.analysis.separate import _DEMUCS_MODELS, separate_stems

# The package re-exports a ``separate`` function, which shadows the submodule on dotted access;
# fetch the real module object straight from sys.modules for monkeypatching.
_SEP_MOD = sys.modules[separate_stems.__module__]


def test_none_mode_is_passthrough():
    audio = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    res = separate_stems(audio, 44_100, "none")
    assert res.instrumental is audio
    assert res.drums is None
    assert res.bass is None
    assert res.sr == 44_100


def test_demucs_mode_map_covers_the_three_variants():
    assert _DEMUCS_MODELS == {
        "demucs": "htdemucs",
        "demucs-ft": "htdemucs_ft",
        "demucs6": "htdemucs_6s",
    }


def test_spleeter_mode_is_not_implemented():
    with pytest.raises(NotImplementedError):
        separate_stems(np.zeros(8, dtype=np.float32), 44_100, "spleeter")


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="unknown separation mode"):
        separate_stems(np.zeros(8, dtype=np.float32), 44_100, "bogus")


def test_demucs_modes_dispatch_to_separate_demucs(monkeypatch):
    """Each demucs* mode forwards the mapped model name to ``_separate_demucs``."""
    seen = {}

    def fake_demucs(audio, sr, model_name="htdemucs"):
        seen["model"] = model_name
        return "sentinel"

    monkeypatch.setattr(_SEP_MOD, "_separate_demucs", fake_demucs)
    for mode, model in _DEMUCS_MODELS.items():
        seen.clear()
        out = separate_stems(np.zeros(8, dtype=np.float32), 44_100, mode)
        assert out == "sentinel"
        assert seen["model"] == model
