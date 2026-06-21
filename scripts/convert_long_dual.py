#!/usr/bin/env python
"""Batch **dual-AY** conversion of the long samples: ``.ym`` (+ ``.ay2.ym``) + a mixed ``.mp3``.

One pipeline run per song produces BOTH:

  * the two single-chip ``.ym`` register dumps -- chip 0 -> ``<name>.ym``, chip 1 ->
    ``<name>.ay2.ym`` (``--chips 2`` writes one single-chip file per AY), and
  * a mixed ``.mp3`` -- both chips summed, rendered from the same in-memory dual-chip song.

Why a custom driver instead of the ``convert`` + ``validate`` pattern the single-chip
``convert_all*.bat`` scripts use: ``validate <name>.ym`` renders only **chip 0**, so for a
dual-AY song its ``.mp3`` would be missing chip 1. ``preview`` would mix both chips but re-runs
the (expensive) neural front-end a second time. This driver renders the in-memory ``n_chips=2``
song straight to ``.mp3`` right after writing the ``.ym`` files -- one neural pass, correct mix.

Pick the front-end with ``--separation`` / ``--transcription`` to mirror the single-chip scripts:

  * basic-pitch + Demucs (default):  ``--separation demucs``
  * basic-pitch + fine-tuned Demucs: ``--separation demucs-ft``
  * basic-pitch + 6-stem Demucs:     ``--separation demucs6``
  * YourMT3 multi-instrument:        ``--transcription yourmt3 --model YMT3+``

Run on a machine with the needed extras (``[neural]`` / ``[yourmt3]`` plus ``[mp3]``) installed.
Songs are processed strictly one at a time -- never launch a second heavy job alongside it.

    python scripts/convert_long_dual.py --separation demucs6 --out-dir results/demucs6_dual
    python scripts/convert_long_dual.py --transcription yourmt3 --model "YMT3+" \\
        --separation none --out-dir results/ymt3plus_dual
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch dual-AY conversion of the long samples -> .ym + mixed .mp3.",
    )
    ap.add_argument("--in-dir", default="samples/long", help="folder of input songs")
    ap.add_argument("--out-dir", default="results/dual", help="output folder")
    ap.add_argument("--separation", default="demucs",
                    choices=["demucs", "demucs-ft", "demucs6", "none"],
                    help="stem separation backend (ignored by --transcription mt3/yourmt3)")
    ap.add_argument("--transcription", default="basic-pitch",
                    choices=["basic-pitch", "mt3", "yourmt3"],
                    help="transcription backend")
    ap.add_argument("--model", default=None,
                    help="YourMT3 variant (only used with --transcription yourmt3)")
    ap.add_argument("--glob", default="*", help="filename glob within --in-dir")
    ap.add_argument("--force", action="store_true",
                    help="re-convert songs whose outputs already exist")
    args = ap.parse_args(argv)

    # Imported lazily (after arg parsing) so ``--help`` works without dragging in the heavy
    # neural / numba stack.
    from audio2ay3.cli import _write_multichip
    from audio2ay3.config import ChipConfig, RunConfig
    from audio2ay3.pipeline import convert
    from audio2ay3.render import Renderer
    from audio2ay3.ymformat import ym_writer

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        print(f"error: input folder not found: {in_dir}", file=sys.stderr)
        return 2

    inputs = sorted(
        p for p in in_dir.glob(args.glob)
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    )
    if not inputs:
        print(f"error: no audio files in {in_dir} matching {args.glob!r}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        chip=ChipConfig(n_chips=2),          # dual-AY: 6 tone channels
        separation=args.separation,
        transcription=args.transcription,
        yourmt3_model=args.model,
    )
    renderer = Renderer(render_sr=cfg.render_sr, oversample=cfg.oversample)

    front_end = args.transcription + (f"/{args.model}" if args.model else "")
    print(f"Converting {len(inputs)} song(s) from {in_dir} -> {out_dir}")
    print(f"  front-end: {front_end} + {args.separation}  |  chips: 2 (dual-AY)\n")

    failures: list[tuple[str, str]] = []
    for i, src in enumerate(inputs, 1):
        stem = src.stem
        out_ym = out_dir / f"{stem}.ym"
        out_mp3 = out_dir / f"{stem}.mp3"
        if out_ym.exists() and out_mp3.exists() and not args.force:
            print(f"[{i}/{len(inputs)}] skip (exists): {stem}")
            continue

        print(f"[{i}/{len(inputs)}] {src.name} ...", flush=True)
        t0 = time.time()
        try:
            song = convert(str(src), cfg)                       # one neural pass
            ym_paths = _write_multichip(song, str(out_ym), ym_writer)
            renderer.render_to_file(song, str(out_mp3), bitrate_kbps=cfg.mp3_bitrate_kbps)
        except Exception as exc:  # batch driver: report and keep going to the next song
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append((stem, f"{type(exc).__name__}: {exc}"))
            continue

        dt = (time.time() - t0) / 60.0
        print(f"    ok: {song.n_frames} frames ({song.duration_s:.1f}s) in {dt:.1f} min")
        print(f"    -> {', '.join(ym_paths)}")
        print(f"    -> {out_mp3}")

    done = len(inputs) - len(failures)
    print(f"\nDone: {done}/{len(inputs)} succeeded -> {out_dir}")
    if failures:
        print("Failed:")
        for stem, msg in failures:
            print(f"  - {stem}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
