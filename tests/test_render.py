"""End-to-end render tests and a write -> read -> render integration check."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2ay3.render import Renderer
from audio2ay3.ymformat import ym_reader, ym_writer
from audio2ay3.ymformat.model import YmSong


def _tone_song(n: int = 25) -> YmSong:
    frames = np.zeros((n, 16), dtype=np.uint8)
    frames[:, 0] = 200  # channel A tone period (fine)
    frames[:, 7] = 0x3E  # enable tone on channel A only
    frames[:, 8] = 15  # channel A amplitude
    frames[:, 13] = 0xFF  # no envelope retrigger
    return YmSong(frames=frames, master_clock=1_773_400, frame_rate=50)


def test_render_length_and_amplitude():
    song = _tone_song(25)
    r = Renderer(render_sr=22_050, oversample=1)
    pcm = r.render(song)
    assert pcm.size == int(25 * 22_050 / 50)
    assert np.max(np.abs(pcm)) > 0.1


def test_render_normalises_to_headroom():
    song = _tone_song(25)
    r = Renderer(render_sr=22_050, oversample=1, headroom_db=-1.0)
    pcm = r.render(song)
    peak = np.max(np.abs(pcm))
    assert abs(peak - 10 ** (-1.0 / 20.0)) < 1e-3


def test_write_read_render_integration(tmp_path: Path):
    song = _tone_song(20)
    path = tmp_path / "tune.ym"
    ym_writer.write(song, str(path))
    loaded = ym_reader.load(str(path))
    pcm = Renderer(render_sr=22_050, oversample=1).render(loaded)
    assert pcm.size == int(20 * 22_050 / 50)
    assert np.max(np.abs(pcm)) > 0.1


def test_write_wav_file(tmp_path: Path):
    song = _tone_song(10)
    out = tmp_path / "out.wav"
    Renderer(render_sr=22_050, oversample=1).render_to_file(song, str(out))
    assert out.exists() and out.stat().st_size > 44  # header + samples
