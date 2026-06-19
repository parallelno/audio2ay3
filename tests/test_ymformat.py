"""Round-trip and parsing tests for the YM reader/writer."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from audio2ay3.ymformat import ym_reader, ym_writer
from audio2ay3.ymformat._lha import depack_lha, is_lha
from audio2ay3.ymformat.model import YmSong

_SAMPLE_LHA = pathlib.Path(__file__).resolve().parents[1] / "samples" / "ym" / "song01.ym"


def test_ym6_roundtrip_is_lossless():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(64, 16), dtype=np.uint8)
    song = YmSong(frames=frames, master_clock=1_773_400, frame_rate=50,
                  loop_frame=7, name="title", author="me", comment="hi")

    data = ym_writer.to_bytes(song, "YM6")
    back = ym_reader.from_bytes(data)

    assert back.version == "YM6"
    assert back.n_frames == 64
    assert back.master_clock == 1_773_400
    assert back.frame_rate == 50
    assert back.loop_frame == 7
    assert back.name == "title"
    assert back.author == "me"
    assert back.comment == "hi"
    assert np.array_equal(back.frames, frames)


def test_writer_emits_valid_header_and_end_marker():
    frames = np.zeros((4, 16), dtype=np.uint8)
    data = ym_writer.to_bytes(YmSong(frames=frames), "YM6")
    assert data[:4] == b"YM6!"
    assert data[4:12] == b"LeOnArD!"
    assert data[-4:] == b"End!"


def test_lha_packed_file_is_depacked_transparently():
    if not _SAMPLE_LHA.exists():
        pytest.skip("LHA sample samples/ym/song01.ym not present")
    song = ym_reader.load(str(_SAMPLE_LHA))
    assert song.version in ("YM5", "YM6")
    assert song.n_frames > 0
    assert song.frames.shape[1] == 16


def test_lha_depack_unwraps_to_ym_magic():
    if not _SAMPLE_LHA.exists():
        pytest.skip("LHA sample samples/ym/song01.ym not present")
    data = _SAMPLE_LHA.read_bytes()
    assert is_lha(data)
    out = depack_lha(data)
    assert out[:4] in (b"YM6!", b"YM5!", b"YM3!", b"YM2!")


def test_unsupported_lha_method_is_reported_clearly():
    # level-0 header (size byte 0x14), method "-lh1-" which we do not decode.
    fake = bytes([0x14, 0x00]) + b"-lh1-" + bytes(40)
    with pytest.raises(NotImplementedError, match="not supported"):
        depack_lha(fake)


def test_ym3_is_zero_extended_to_16_registers():
    # 14 interleaved registers, 3 frames.
    body = bytes(range(14 * 3))
    data = b"YM3!" + body
    song = ym_reader.from_bytes(data)
    assert song.frames.shape == (3, 16)
    assert np.all(song.frames[:, 14:] == 0)
    assert song.master_clock == 2_000_000
