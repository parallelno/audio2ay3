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


def _default_out(inp: str, ext: str) -> str:
    return str(Path("build") / (Path(inp).stem + ext))


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
        render_sr=getattr(args, "sr", 44_100),
        oversample=getattr(args, "oversample", 2),
        mp3_bitrate_kbps=getattr(args, "bitrate", 192),
        seed=getattr(args, "seed", 0),
        amp_envelope=AmpEnvelope(enabled=not getattr(args, "no_amp_envelope", False)),
        vibrato=Vibrato(enabled=getattr(args, "vibrato", False)),
        breath=getattr(args, "breath", False),
        arpeggio=getattr(args, "arpeggio", False),
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
    from .ymformat import ym_writer

    cfg = _build_run_config(args)
    out = args.output or _default_out(args.input, ".ym")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    explain = getattr(args, "explain", False)
    trace: list | None = [] if explain else None
    try:
        song = convert(args.input, cfg, trace=trace)
    except FileNotFoundError:
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 3
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    ym_writer.write(song, out)
    print(f"ok: {song.n_frames} frames ({song.duration_s:.1f}s) -> {out}")
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
    try:
        song = preview(args.input, out, cfg, max_seconds=args.duration, trace=trace)
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
    sp.add_argument("--separation", choices=["demucs", "spleeter", "none"],
                    default="demucs", help="neural source separation backend")
    sp.add_argument("--transcription", choices=["basic-pitch", "mt3", "yourmt3", "onsets-frames"],
                    default="basic-pitch", help="neural transcription backend")
    sp.add_argument("--clock", type=int, default=None, help="master clock (Hz)")
    sp.add_argument("--frame-rate", type=int, default=None, dest="frame_rate",
                    help="replay frame rate (Hz)")
    sp.add_argument("--chips", type=int, default=None, help="number of AY chips (1 or 2)")
    sp.add_argument("--no-gpu", action="store_true", dest="no_gpu",
                    help="force CPU for neural models")
    sp.add_argument("--seed", type=int, default=0, help="deterministic seed")


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

    c = sub.add_parser("convert", help="Convert audio to a .ym register stream")
    c.add_argument("input", help="input audio file (WAV/FLAC/OGG; MP3 if libsndfile supports)")
    c.add_argument("-o", "--output", help="output .ym (default: build/<name>.ym)")
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
        help="Fetch the optional YourMT3+ transcription backend (GPL-3.0) into a local cache",
    )
    sy.add_argument("--dir", default=None,
                    help="checkout directory (default: a per-user cache dir)")
    sy.add_argument("--repo-url", dest="repo_url", default=None,
                    help="clone URL (default: the YourMT3 HuggingFace Space)")
    sy.add_argument("--model", default="YPTF.MoE+Multi (noPS)",
                    help="model variant to verify a checkpoint for")
    sy.add_argument("--force", action="store_true",
                    help="update an existing checkout (git pull) instead of skipping")
    sy.set_defaults(func=cmd_setup_yourmt3)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
