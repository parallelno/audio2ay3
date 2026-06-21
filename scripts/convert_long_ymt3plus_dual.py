#!/usr/bin/env python
"""Batch-convert every long sample with YourMT3 (``YMT3+`` variant) in dual-AY mode.

Each song is run through **one** YourMT3 inference that yields BOTH:

  * the ``.ym`` register dumps -- chip 0 -> ``<name>.ym``, chip 1 -> ``<name>.ay2.ym``
    (``--chips 2`` writes one single-chip file per AY), and
  * a mixed ``.mp3`` preview -- both chips summed, rendered from the same in-memory
    dual-chip song (no second inference).

Outputs land in ``results/ymt3plus_dual/`` by default.

Why a custom driver instead of ``convert`` + ``preview``: ``preview`` would re-run the
(expensive) YourMT3 inference a second time, and ``validate`` on ``<name>.ym`` would render
only chip 0 -- so neither CLI combo gives both ``.ym`` files *and* a correctly mixed ``.mp3``
from a single inference. This driver renders the in-memory ``n_chips=2`` song straight to
``.mp3`` (exactly what ``preview`` does internally) right after writing the ``.ym`` files.

Run on a machine with the ``[yourmt3]`` + ``[mp3]`` extras installed and YourMT3 set up
(``audio2ay3 setup-yourmt3``). Songs are processed strictly one at a time -- never launch a
second heavy YourMT3 job alongside it.

    python scripts/convert_long_ymt3plus_dual.py
    python scripts/convert_long_ymt3plus_dual.py --force          # re-do existing outputs
    python scripts/convert_long_ymt3plus_dual.py --glob "Goblins*"  # a subset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch dual-AY YourMT3 (YMT3+) conversion -> .ym + mixed .mp3.",
    )
    ap.add_argument("--in-dir", default="samples/long", help="folder of input songs")
    ap.add_argument("--out-dir", default="results/ymt3plus_dual", help="output folder")
    ap.add_argument("--model", default="YMT3+", help="YourMT3 variant")
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
        chip=ChipConfig(n_chips=2),     # dual-AY: 6 tone channels
        transcription="yourmt3",
        yourmt3_model=args.model,        # YMT3+ by default
    )
    renderer = Renderer(render_sr=cfg.render_sr, oversample=cfg.oversample)

    print(f"Converting {len(inputs)} song(s) from {in_dir} -> {out_dir}")
    print(f"  backend: yourmt3 / {args.model}  |  chips: 2 (dual-AY)\n")

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
            song = convert(str(src), cfg)                       # one YourMT3 inference
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
