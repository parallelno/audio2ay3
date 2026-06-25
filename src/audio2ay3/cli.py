"""Command-line interface: ``validate``, ``convert``, ``preview``.

- ``validate`` renders a ``.ym`` file to audio through the emulator (Milestone 1).
- ``convert`` runs the neural pipeline (separation + transcription) and writes a ``.ym``.
- ``preview`` converts, then renders the result to audio via the same emulator.

The neural backends are optional; ``convert``/``preview`` print a clear "install the extra"
message if the heavy dependencies are absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ChipConfig, RunConfig

# Selectable YourMT3 variants (kept in sync with analysis._yourmt3_infer._MODEL_PRESETS; a test
# asserts they match). Hardcoded here so building the parser never imports the heavy analysis stack.
_YOURMT3_MODELS = (
    "YPTF.MoE+Multi (noPS)",
    "YPTF.MoE+Multi (PS)",
    "YPTF+Multi (PS)",
    "YMT3+",
)


def _default_out(inp: str, ext: str) -> str:
    return str(Path("build") / (Path(inp).stem + ext))


def _write_multichip(song, out: str, ym_writer) -> list[str]:
    """Write one ``.ym`` per chip (chip 0 to *out*, chip 1 to ``<name>.ay2.ym``)."""
    base = Path(out)
    paths: list[str] = []
    for i, chip_song in enumerate(song.per_chip_songs()):
        path = base if i == 0 else base.with_name(f"{base.stem}.ay{i + 1}{base.suffix}")
        ym_writer.write(chip_song, str(path))
        paths.append(str(path))
    return paths


def _write_song(song, out: str, fmt: str) -> list[str]:
    """Write *song* in the requested format; return the list of files written.

    ``fmt='ym'``: one ``.ym`` per chip (chip 1 -> ``<stem>.ay2.ym``).
    ``fmt='vtx'``: one ``.vtx`` file; chipType=2 encodes dual-AY natively.
    """
    from .ymformat import vtx_writer, ym_writer

    if fmt == "vtx":
        vtx_writer.write(song, out)
        return [out]
    # default: ym
    if song.n_chips > 1:
        return _write_multichip(song, out, ym_writer)
    ym_writer.write(song, out)
    return [out]


def _make_progress(args: argparse.Namespace, cfg, *, render: bool):
    """A stage progress reporter for convert/preview, or ``None`` when disabled / non-interactive."""
    if getattr(args, "no_progress", False) or not sys.stderr.isatty():
        return None
    from .pipeline import progress_total
    from .progress import ProgressReporter

    return ProgressReporter(progress_total(cfg, render=render))


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    from .config import AmpEnvelope, Vibrato

    base = ChipConfig()
    chip = ChipConfig(
        master_clock_hz=getattr(args, "clock", None) or base.master_clock_hz,
        frame_rate_hz=getattr(args, "frame_rate", None) or base.frame_rate_hz,
        n_chips=getattr(args, "chips", None) or base.n_chips,
    )
    return RunConfig(
        chip=chip,
        use_gpu=not getattr(args, "no_gpu", False),
        separation=args.separation,
        transcription=args.transcription,
        yourmt3_model=getattr(args, "yourmt3_model", None),
        render_sr=getattr(args, "sr", 44_100),
        oversample=getattr(args, "oversample", 2),
        mp3_bitrate_kbps=getattr(args, "bitrate", 192),
        seed=getattr(args, "seed", 0),
        amp_envelope=AmpEnvelope(enabled=not getattr(args, "no_amp_envelope", False)),
        vibrato=Vibrato(enabled=getattr(args, "vibrato", False)),
        breath=getattr(args, "breath", False),
        arpeggio=getattr(args, "arpeggio", False),
        stems_dir=Path(args.stems_dir) if getattr(args, "stems_dir", None) else None,
        noise_volume=getattr(args, "noise_volume", 1.0),
    )


def cmd_validate(args: argparse.Namespace) -> int:
    from .ymformat import ym_reader
    from .render import Renderer

    try:
        song = ym_reader.load(args.input)
    except FileNotFoundError:
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 3
    except (NotImplementedError, ValueError) as exc:
        print(f"error: cannot read YM: {exc}", file=sys.stderr)
        return 3

    if args.clock:
        song.master_clock = args.clock
    if args.frame_rate:
        song.frame_rate = args.frame_rate

    ext = ".wav" if args.wav else ".mp3"
    out = args.output or _default_out(args.input, ext)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    renderer = Renderer(render_sr=args.sr, oversample=args.oversample)
    try:
        renderer.render_to_file(song, out, bitrate_kbps=args.bitrate,
                                max_seconds=args.duration)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(
        f"ok: {song.version}, {song.master_clock} Hz clock, {song.frame_rate} Hz, "
        f"{song.n_frames} frames ({song.duration_s:.1f}s) -> {out}"
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .pipeline import convert

    fmt = getattr(args, "format", "ym")
    default_ext = ".vtx" if fmt == "vtx" else ".ym"
    cfg = _build_run_config(args)
    out = args.output or _default_out(args.input, default_ext)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    explain = getattr(args, "explain", False)
    trace: list | None = [] if explain else None
    progress = _make_progress(args, cfg, render=False)
    try:
        song = convert(args.input, cfg, trace=trace, progress=progress)
    except FileNotFoundError:
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 3
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    out_paths = _write_song(song, out, fmt)
    print(f"ok: {song.n_frames} frames ({song.duration_s:.1f}s) -> "
          f"{', '.join(out_paths)}")
    if explain:
        from .explain import describe_song

        print(describe_song(song))
        if trace:
            from .mapping.contention import describe_contention, voice_contention

            print(describe_contention(voice_contention(trace[0], cfg)))
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from .pipeline import preview

    cfg = _build_run_config(args)
    ext = ".wav" if args.wav else ".mp3"
    out = args.output or _default_out(args.input, ext)

    explain = getattr(args, "explain", False)
    trace: list | None = [] if explain else None
    progress = _make_progress(args, cfg, render=True)
    try:
        song = preview(args.input, out, cfg, max_seconds=args.duration, trace=trace,
                       progress=progress)
    except FileNotFoundError:
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 3
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(f"ok: {song.n_frames} frames ({song.duration_s:.1f}s) -> {out}")
    if explain:
        from .explain import describe_song

        print(describe_song(song))
        if trace:
            from .mapping.contention import describe_contention, voice_contention

            print(describe_contention(voice_contention(trace[0], cfg)))
    return 0


def cmd_setup_yourmt3(args: argparse.Namespace) -> int:
    from .analysis.yourmt3_setup import _DEFAULT_REPO_URL, setup_yourmt3

    try:
        setup_yourmt3(
            target_dir=getattr(args, "dir", None),
            repo_url=getattr(args, "repo_url", None) or _DEFAULT_REPO_URL,
            model_name=args.model,
            force=getattr(args, "force", False),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


def _add_analysis_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--separation", choices=["demucs", "demucs-ft", "demucs6", "spleeter", "none"],
                    default="demucs",
                    help="neural source separation backend (demucs-ft = better/slower; "
                         "demucs6 = 6-stem, experimental)")
    sp.add_argument("--transcription", choices=["basic-pitch", "mt3", "yourmt3", "onsets-frames"],
                    default="basic-pitch",
                    help="neural transcription backend (yourmt3 is optional/opt-in; pick its "
                         "variant with --yourmt3-model)")
    sp.add_argument("--yourmt3-model", dest="yourmt3_model", choices=list(_YOURMT3_MODELS),
                    default=None,
                    help="YourMT3 variant for --transcription yourmt3 (default: 'YMT3+', "
                         "overridable via AUDIO2AY3_YOURMT3_MODEL)")
    sp.add_argument("--clock", type=int, default=None, help="master clock (Hz)")
    sp.add_argument("--frame-rate", type=int, default=None, dest="frame_rate",
                    help="replay frame rate (Hz)")
    sp.add_argument("--chips", type=int, default=None, choices=[1, 2],
                    help="number of AY chips: 1 (3 channels) or 2 (dual-AY, 6 channels; "
                         "writes chip 1 to <name>.ay2.ym)")
    sp.add_argument("--no-gpu", action="store_true", dest="no_gpu",
                    help="force CPU for neural models")
    sp.add_argument("--no-progress", action="store_true", dest="no_progress",
                    help="disable the per-stage progress bar")
    sp.add_argument("--seed", type=int, default=0, help="deterministic seed")
    sp.add_argument("--stems-dir", dest="stems_dir", default=None,
                    help="directory of pre-separated stems; when given, Demucs is skipped and "
                         "stems are loaded from <stems-dir>/<song>/<song> (Synth|Bass|Drums).mp3 "
                         "directly (--separation is ignored)")


def _add_arrangement_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--no-amp-envelope", action="store_true", dest="no_amp_envelope",
                    help="disable per-note amplitude shaping (flat, constant-volume notes)")
    sp.add_argument("--vibrato", action="store_true",
                    help="add a pitch-LFO vibrato to expressive voices "
                         "(organ/strings/reed/pipe/synth lead)")
    sp.add_argument("--breath", action="store_true",
                    help="add a breathy noise chiff at the attack of wind voices (reeds/pipes)")
    sp.add_argument("--arpeggio", action="store_true",
                    help="cycle squeezed chord tones on one channel instead of dropping them")
    sp.add_argument("--explain", action="store_true",
                    help="print register-level diagnostics for the arranged song")
    sp.add_argument("--noise-volume", type=float, default=1.0, dest="noise_volume",
                    metavar="SCALE",
                    help="noise channel volume as a linear scale (default 1.0; "
                         "0.5 = half as loud; 0.0 = muted)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio2ay3",
        description="Instrumental audio <-> AY-3-8910 register streams.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Render a .ym file to audio via the emulator")
    v.add_argument("input", help="input .ym file (YM2/3/3b/5/6, LHA-packed ok)")
    v.add_argument("-o", "--output", help="output .wav/.mp3 (default: build/<name>.mp3)")
    v.add_argument("--wav", action="store_true", help="write WAV instead of MP3")
    v.add_argument("--sr", type=int, default=44_100, help="render sample rate")
    v.add_argument("--oversample", type=int, default=2, help="anti-alias oversample factor")
    v.add_argument("--bitrate", type=int, default=192, help="MP3 bitrate (kbps)")
    v.add_argument("--duration", type=float, default=None, help="limit output seconds")
    v.add_argument("--clock", type=int, default=None, help="override master clock (Hz)")
    v.add_argument("--frame-rate", type=int, default=None, dest="frame_rate",
                   help="override replay frame rate (Hz)")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("convert", help="Convert audio to a register-dump file (.ym or .vtx)")
    c.add_argument("input", help="input audio file (WAV/FLAC/OGG; MP3 if libsndfile supports)")
    c.add_argument("-o", "--output",
                   help="output file (default: build/<name>.ym or build/<name>.vtx)")
    c.add_argument("--format", choices=["ym", "vtx"], default="ym",
                   help="output register-dump format: 'ym' (YM6, one file per chip; default) or "
                        "'vtx' (Vortex Tracker; single file, natively supports dual-AY via "
                        "turboAY chipType=2)")
    _add_analysis_args(c)
    _add_arrangement_args(c)
    c.set_defaults(func=cmd_convert)

    pr = sub.add_parser("preview", help="Convert audio then emulate it to audio")
    pr.add_argument("input", help="input audio file")
    pr.add_argument("-o", "--output", help="output .wav/.mp3 (default: build/<name>.mp3)")
    pr.add_argument("--wav", action="store_true", help="write WAV instead of MP3")
    pr.add_argument("--sr", type=int, default=44_100, help="render sample rate")
    pr.add_argument("--oversample", type=int, default=2, help="anti-alias oversample factor")
    pr.add_argument("--bitrate", type=int, default=192, help="MP3 bitrate (kbps)")
    pr.add_argument("--duration", type=float, default=None, help="limit output seconds")
    _add_analysis_args(pr)
    _add_arrangement_args(pr)
    pr.set_defaults(func=cmd_preview)

    sy = sub.add_parser(
        "setup-yourmt3",
        help="Fetch the optional (experimental) YourMT3+ transcription backend (GPL-3.0) into a "
             "local cache",
    )
    sy.add_argument("--dir", default=None,
                    help="checkout directory (default: a per-user cache dir)")
    sy.add_argument("--repo-url", dest="repo_url", default=None,
                    help="clone URL (default: the YourMT3 HuggingFace Space)")
    sy.add_argument("--model", default="YMT3+", choices=list(_YOURMT3_MODELS),
                    help="model variant to verify a checkpoint for (default 'YMT3+' did best in "
                         "testing)")
    sy.add_argument("--force", action="store_true",
                    help="update an existing checkout (git pull) instead of skipping")
    sy.set_defaults(func=cmd_setup_yourmt3)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
