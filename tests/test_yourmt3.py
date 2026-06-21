"""Tests for the YourMT3+ backend's deterministic glue.

The heavy PyTorch/YourMT3 stack is never imported here. We exercise (1) the pure ``pretty_midi`` ->
duck-typed ``NoteSequence`` adapter, (2) that its output flows through the shared
:func:`note_sequence_to_transcription` routing (drums/bass/melody), (3) the dispatch + env-var
error surface, and (4) the ``setup-yourmt3`` clone flow with an injected command runner (no git or
network). Actual YourMT3 inference is validated out-of-band on a machine with the ``[yourmt3]``
extra and the GPL model checkout installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from audio2ay3.analysis._yourmt3_infer import (
    _DEFAULT_MODEL,
    _MODEL_PRESETS,
    _pretty_midi_to_note_sequence,
    _resolve_model_name,
)
from audio2ay3.analysis.transcribe import note_sequence_to_transcription, transcribe
from audio2ay3.analysis.yourmt3_setup import setup_yourmt3


def _pm_note(pitch, start, end, *, velocity=100):
    return SimpleNamespace(pitch=pitch, start=start, end=end, velocity=velocity)


def _pm_instrument(program, notes, *, is_drum=False):
    return SimpleNamespace(program=program, is_drum=is_drum, notes=notes)


def _pretty_midi(instruments):
    return SimpleNamespace(instruments=instruments)


def test_pretty_midi_adapter_translates_fields_and_total_time():
    pm = _pretty_midi(
        [
            _pm_instrument(0, [_pm_note(60, 0.0, 0.5, velocity=80)]),
            _pm_instrument(34, [_pm_note(40, 0.25, 1.25, velocity=110)]),
        ]
    )
    ns = _pretty_midi_to_note_sequence(pm)
    assert ns.total_time == pytest.approx(1.25)
    first = ns.notes[0]
    assert first.pitch == 60
    assert first.start_time == pytest.approx(0.0)
    assert first.end_time == pytest.approx(0.5)
    assert first.velocity == 80
    assert first.program == 0
    assert first.is_drum is False


def test_pretty_midi_adapter_marks_drum_instrument():
    pm = _pretty_midi([_pm_instrument(0, [_pm_note(36, 0.0, 0.1)], is_drum=True)])
    ns = _pretty_midi_to_note_sequence(pm)
    assert ns.notes[0].is_drum is True


def test_adapter_output_routes_through_shared_converter():
    pm = _pretty_midi(
        [
            _pm_instrument(0, [_pm_note(60, 0.0, 0.5)]),  # melody (piano)
            _pm_instrument(34, [_pm_note(40, 0.0, 1.0)]),  # GM bass family -> bass_notes
            _pm_instrument(
                0,
                [_pm_note(36, 0.25, 0.30), _pm_note(38, 0.5, 0.55)],  # kick + snare
                is_drum=True,
            ),
        ]
    )
    tr = note_sequence_to_transcription(_pretty_midi_to_note_sequence(pm))
    assert len(tr.notes) == 1
    assert len(tr.bass_notes) == 1
    assert [p.kind for p in tr.percussion] == ["kick", "snare"]
    assert tr.notes[0].program == 0
    assert tr.bass_notes[0].program == 34
    assert tr.duration_s == pytest.approx(1.0)


def test_transcribe_yourmt3_without_repo_raises_actionable_runtime_error(monkeypatch, tmp_path):
    # No repo checkout configured AND the cache dir does not exist: the call must surface a clean
    # RuntimeError naming the backend (and the setup helper), without importing torch.
    monkeypatch.delenv("AUDIO2AY3_YOURMT3_DIR", raising=False)
    monkeypatch.setattr(
        "audio2ay3.analysis._yourmt3_infer.default_yourmt3_dir",
        lambda: tmp_path / "does-not-exist",
    )
    with pytest.raises(RuntimeError, match="yourmt3"):
        transcribe(np.zeros(16, dtype=np.float32), 16000, "yourmt3")


def test_transcribe_yourmt3_unknown_model_variant_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO2AY3_YOURMT3_DIR", str(tmp_path))
    monkeypatch.setenv("AUDIO2AY3_YOURMT3_MODEL", "Not A Real Variant")
    with pytest.raises(RuntimeError, match="not a known YourMT3 variant"):
        transcribe(np.zeros(16, dtype=np.float32), 16000, "yourmt3")


class _FakeRunner:
    """Records argv lists and returns a fixed exit code (0 = success) without running anything."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        return self.returncode


def test_setup_yourmt3_clones_into_target_and_reports(tmp_path):
    runner = _FakeRunner()
    logs: list[str] = []
    target = tmp_path / "ymt3"
    out = setup_yourmt3(target_dir=target, runner=runner, log=logs.append)
    assert out == target
    # A git clone of the chosen repo into the target dir must have been issued.
    clone = [c for c in runner.calls if c[:2] == ["git", "clone"]]
    assert clone and clone[0][-1] == str(target)
    # No real checkout exists, so the verifier warns about the missing glue + checkpoint.
    joined = "\n".join(logs)
    assert "model_helper.py" in joined
    assert "checkpoint" in joined.lower()


def test_setup_yourmt3_skips_clone_when_already_present(tmp_path):
    target = tmp_path / "ymt3"
    (target / ".git").mkdir(parents=True)
    runner = _FakeRunner()
    logs: list[str] = []
    setup_yourmt3(target_dir=target, runner=runner, log=logs.append)
    assert not [c for c in runner.calls if c[:2] == ["git", "clone"]]
    assert any("Skipping clone" in line for line in logs)


def test_setup_yourmt3_unknown_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown model variant"):
        setup_yourmt3(target_dir=tmp_path, model_name="nope", runner=_FakeRunner(), log=lambda _: None)


def test_resolve_model_name_precedence(monkeypatch):
    monkeypatch.delenv("AUDIO2AY3_YOURMT3_MODEL", raising=False)
    # Nothing set -> backend default.
    assert _resolve_model_name(None) == _DEFAULT_MODEL
    # An explicit name (e.g. from --yourmt3-model) is honoured.
    assert _resolve_model_name("YPTF.MoE+Multi (noPS)") == "YPTF.MoE+Multi (noPS)"
    # The env var is used when no explicit name is given (use a non-default variant to prove it)...
    monkeypatch.setenv("AUDIO2AY3_YOURMT3_MODEL", "YPTF.MoE+Multi (noPS)")
    assert _resolve_model_name(None) == "YPTF.MoE+Multi (noPS)"
    # ...but an explicit name wins over the env var.
    assert _resolve_model_name("YPTF.MoE+Multi (PS)") == "YPTF.MoE+Multi (PS)"


def test_resolve_model_name_rejects_unknown(monkeypatch):
    monkeypatch.delenv("AUDIO2AY3_YOURMT3_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="not a known YourMT3 variant"):
        _resolve_model_name("Bogus Variant")


def test_cli_model_choices_match_presets():
    from audio2ay3.cli import _YOURMT3_MODELS

    assert set(_YOURMT3_MODELS) == set(_MODEL_PRESETS)


def test_yourmt3_model_flag_threads_into_run_config():
    from audio2ay3.cli import _build_run_config, build_parser

    args = build_parser().parse_args(
        ["convert", "in.wav", "--transcription", "yourmt3", "--yourmt3-model", "YMT3+"]
    )
    cfg = _build_run_config(args)
    assert cfg.yourmt3_model == "YMT3+"
    # Default (flag omitted) leaves the backend to resolve env var / default.
    default_cfg = _build_run_config(build_parser().parse_args(["convert", "in.wav"]))
    assert default_cfg.yourmt3_model is None


def test_yourmt3_model_flows_to_inference(monkeypatch):
    seen = {}

    def fake_transcribe_yourmt3(audio, sr, model_name=None):
        seen["model_name"] = model_name
        return _pretty_midi_to_note_sequence(
            _pretty_midi([_pm_instrument(0, [_pm_note(60, 0.0, 0.5)])])
        )

    monkeypatch.setattr(
        "audio2ay3.analysis._yourmt3_infer.transcribe_yourmt3", fake_transcribe_yourmt3
    )
    transcribe(np.zeros(16, dtype=np.float32), 16000, "yourmt3", yourmt3_model="YMT3+")
    assert seen["model_name"] == "YMT3+"


def test_setup_yourmt3_requires_git(tmp_path):
    # git --version returns non-zero -> clean RuntimeError before any clone.
    with pytest.raises(RuntimeError, match="git is required"):
        setup_yourmt3(target_dir=tmp_path, runner=_FakeRunner(returncode=1), log=lambda _: None)
