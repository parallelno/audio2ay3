"""Tests for audio decoding dispatch in ``load_audio``.

The ``.m4a`` path shells out to ``ffmpeg``; these mock the subprocess so the test needs neither
ffmpeg nor an actual AAC file. The non-m4a path keeps using soundfile (covered elsewhere).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from audio2ay3.analysis.load_audio import load_audio

_MOD = sys.modules[load_audio.__module__]


def test_m4a_decodes_via_ffmpeg(monkeypatch, tmp_path):
    path = tmp_path / "song.m4a"
    path.write_bytes(b"")  # presence check only; ffmpeg is mocked

    pcm = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)

    monkeypatch.setattr("shutil.which", lambda _: "ffmpeg")

    captured = {}

    def fake_run(cmd, stdout=None, stderr=None):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=pcm.tobytes(), stderr=b"")

    monkeypatch.setattr("subprocess.run", fake_run)

    audio, sr = load_audio(str(path), target_sr=22_050)

    assert sr == 22_050
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, pcm)
    # ffmpeg is asked to produce mono f32 PCM at the requested rate.
    cmd = captured["cmd"]
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "22050"


def test_m4a_without_ffmpeg_raises(monkeypatch, tmp_path):
    path = tmp_path / "song.m4a"
    path.write_bytes(b"")

    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        load_audio(str(path))


def test_m4a_ffmpeg_failure_surfaces(monkeypatch, tmp_path):
    path = tmp_path / "song.m4a"
    path.write_bytes(b"")

    monkeypatch.setattr("shutil.which", lambda _: "ffmpeg")

    def fake_run(cmd, stdout=None, stderr=None):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"bad data")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        load_audio(str(path))


def test_missing_m4a_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: "ffmpeg")
    with pytest.raises(FileNotFoundError):
        load_audio(str(tmp_path / "nope.m4a"))
