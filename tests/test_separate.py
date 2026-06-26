"""Tests for source-separation dispatch (no torch/Demucs imported here).

The heavy Demucs path needs the ``[neural]`` extra, so these only exercise the pure dispatch:
the ``none`` passthrough, the friendly-mode -> Demucs-model map, and the error surfaces for the
unimplemented (``spleeter``) and unknown modes. Real separation is validated out-of-band.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

    def fake_demucs(audio, sr, model_name="htdemucs", *, keep_vocals=False,
                    save_dir=None, save_name=None, save_fmt="wav", save_bitrate_kbps=192):
        seen["model"] = model_name
        return "sentinel"

    monkeypatch.setattr(_SEP_MOD, "_separate_demucs", fake_demucs)
    for mode, model in _DEMUCS_MODELS.items():
        seen.clear()
        out = separate_stems(np.zeros(8, dtype=np.float32), 44_100, mode)
        assert out == "sentinel"
        assert seen["model"] == model


def test_separate_stems_forwards_save_args_to_demucs(monkeypatch):
    """``save_dir``/``save_name`` reach ``_separate_demucs`` for every demucs* mode."""
    seen = {}

    def fake_demucs(audio, sr, model_name="htdemucs", *, keep_vocals=False,
                    save_dir=None, save_name=None, save_fmt="wav", save_bitrate_kbps=192):
        seen["save_dir"] = save_dir
        seen["save_name"] = save_name
        seen["save_fmt"] = save_fmt
        seen["save_bitrate_kbps"] = save_bitrate_kbps
        return "sentinel"

    monkeypatch.setattr(_SEP_MOD, "_separate_demucs", fake_demucs)
    separate_stems(
        np.zeros(8, dtype=np.float32), 44_100, "demucs",
        save_dir="out/dir", save_name="00", save_fmt="mp3", save_bitrate_kbps=128,
    )
    assert seen["save_dir"] == "out/dir"
    assert seen["save_name"] == "00"
    assert seen["save_fmt"] == "mp3"
    assert seen["save_bitrate_kbps"] == 128


def test_save_raw_stems_writes_one_wav_per_source(tmp_path, monkeypatch):
    """``_save_raw_stems`` writes ``<name> (<Source>).wav`` per source, capitalised."""
    written = {}

    class _FakeSf:
        @staticmethod
        def write(path, data, sr):
            written[path] = (np.asarray(data).shape, sr)

    monkeypatch.setitem(sys.modules, "soundfile", _FakeSf)

    raw = {
        "vocals": np.zeros((16, 2), dtype=np.float32),
        "drums": np.zeros((16, 2), dtype=np.float32),
        "bass": np.zeros((16, 2), dtype=np.float32),
        "other": np.zeros((16, 2), dtype=np.float32),
    }
    _SEP_MOD._save_raw_stems(raw, 44_100, tmp_path, "00")

    names = {Path(p).name for p in written}
    assert names == {
        "00 (Vocals).wav",
        "00 (Drums).wav",
        "00 (Bass).wav",
        "00 (Other).wav",
    }
    for shape, sr in written.values():
        assert shape == (16, 2)  # stereo preserved
        assert sr == 44_100  # native sample rate preserved


def test_save_raw_stems_mp3_encodes_one_file_per_source(tmp_path, monkeypatch):
    """``fmt="mp3"`` writes a stereo ``.mp3`` per source via lameenc (channels preserved)."""
    encoders = []

    class _FakeEncoder:
        def __init__(self):
            self.channels = None
            self.bitrate = None
            encoders.append(self)

        def set_bit_rate(self, b):
            self.bitrate = b

        def set_in_sample_rate(self, sr):
            self.sr = sr

        def set_channels(self, c):
            self.channels = c

        def set_quality(self, q):
            pass

        def encode(self, data):
            return b"frame"

        def flush(self):
            return b"end"

    class _FakeLameenc:
        Encoder = _FakeEncoder

    monkeypatch.setitem(sys.modules, "lameenc", _FakeLameenc)

    raw = {
        "vocals": np.zeros((16, 2), dtype=np.float32),
        "other": np.zeros((16, 2), dtype=np.float32),
    }
    _SEP_MOD._save_raw_stems(raw, 44_100, tmp_path, "00", fmt="mp3", bitrate_kbps=128)

    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"00 (Vocals).mp3", "00 (Other).mp3"}
    assert all((tmp_path / n).read_bytes() == b"frameend" for n in names)
    assert [e.channels for e in encoders] == [2, 2]  # stereo preserved
    assert all(e.bitrate == 128 for e in encoders)


def test_load_from_stems_dir_accepts_other_alias(tmp_path, monkeypatch):
    """A raw-dumped ``(Other)`` stem loads as the melodic stem (round-trips --save-stems)."""
    import audio2ay3.analysis.load_audio  # noqa: F401  (ensure the submodule is imported)
    from audio2ay3.analysis.separate import load_from_stems_dir

    # The package __init__ rebinds ``load_audio`` to the function, shadowing the submodule on
    # dotted access; fetch the real module object from sys.modules to monkeypatch it.
    _load_mod = sys.modules["audio2ay3.analysis.load_audio"]

    song = tmp_path / "00"
    song.mkdir()
    (song / "00 (Other).wav").write_bytes(b"")
    (song / "00 (Bass).wav").write_bytes(b"")
    (song / "00 (Drums).wav").write_bytes(b"")

    def fake_load(path, target_sr):
        return np.zeros(8, dtype=np.float32), target_sr

    monkeypatch.setattr(_load_mod, "load_audio", fake_load)

    res = load_from_stems_dir("00", tmp_path, 44_100)
    assert res is not None
    assert res.instrumental is not None  # (Other) accepted in the Synth slot
    assert res.bass is not None
    assert res.drums is not None


