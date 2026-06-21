from audio2ay3.cli import _build_run_config, build_parser


def _convert_cfg(*argv):
    args = build_parser().parse_args(["convert", "in.wav", *argv])
    return _build_run_config(args)


def test_timbre_flags_default_off():
    cfg = _convert_cfg()
    assert cfg.vibrato.enabled is False
    assert cfg.breath is False
    assert cfg.arpeggio is False


def test_timbre_flags_opt_in():
    cfg = _convert_cfg("--vibrato", "--breath", "--arpeggio")
    assert cfg.vibrato.enabled is True
    assert cfg.breath is True
    assert cfg.arpeggio is True


def test_preview_also_wires_timbre_flags():
    # preview shares _add_arrangement_args, so the same flags must reach RunConfig.
    args = build_parser().parse_args(["preview", "in.wav", "--arpeggio"])
    cfg = _build_run_config(args)
    assert cfg.arpeggio is True
    assert cfg.vibrato.enabled is False
    assert cfg.breath is False
