"""Tests for the stage progress bar and the convert/preview stage plan (pure, no neural deps)."""

from __future__ import annotations

import io

from audio2ay3.config import RunConfig
from audio2ay3.pipeline import progress_total, stage_labels
from audio2ay3.progress import NullProgress, ProgressReporter


def test_null_progress_is_a_noop():
    NullProgress().step("anything")  # must not raise or print


def test_progress_reporter_advances_to_full():
    buf = io.StringIO()
    rep = ProgressReporter(3, stream=buf, width=10)
    rep.step("a")
    rep.step("b")
    rep.step("c")
    lines = buf.getvalue().splitlines()
    assert len(lines) == 3
    assert "[1/3]" in lines[0] and "a" in lines[0]
    assert "33%" in lines[0]
    assert "[3/3]" in lines[2] and "100%" in lines[2]
    assert lines[2].count("#") == 10  # the bar is full on the final step


def test_progress_reporter_never_overflows_total():
    buf = io.StringIO()
    rep = ProgressReporter(1, stream=buf)
    rep.step("x")
    rep.step("y")  # extra steps clamp at total rather than exceeding it
    last = buf.getvalue().splitlines()[-1]
    assert "[1/1]" in last and "100%" in last


def test_stage_labels_basic_pitch_with_demucs():
    cfg = RunConfig(separation="demucs", transcription="basic-pitch")
    assert stage_labels(cfg, render=False) == [
        "loading audio",
        "separating stems",
        "transcribing",
        "detecting percussion",
        "arranging",
    ]
    # preview adds the render stage.
    assert stage_labels(cfg, render=True)[-1] == "rendering audio"
    assert progress_total(cfg, render=True) == 6


def test_stage_labels_separation_none_skips_stems_and_percussion():
    cfg = RunConfig(separation="none", transcription="basic-pitch")
    assert stage_labels(cfg, render=False) == ["loading audio", "transcribing", "arranging"]
    assert progress_total(cfg, render=False) == 3


def test_stage_labels_multitrack_backend():
    cfg = RunConfig(transcription="mt3")
    assert stage_labels(cfg, render=False) == [
        "loading audio",
        "transcribing (multitrack)",
        "arranging",
    ]
    assert progress_total(cfg, render=True) == 4
