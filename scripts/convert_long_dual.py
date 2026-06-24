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
    ap.add_argument("--stems-dir", dest="stems_dir", default=None,
                    help="directory of pre-separated stems; skips Demucs and loads "
                         "<stems-dir>/<song>/<song> (Synth|Bass|Drums).mp3 directly "
                         "(--separation is ignored when this is set)")
    ap.add_argument("--stems-only", dest="stems_only", action="store_true",
                    help="derive the input list from --stems-dir subfolders directly; "
                         "no --in-dir needed. Each subfolder must contain a (Synth) stem.")
    ap.add_argument("--noise-volume", type=float, default=1.0, dest="noise_volume",
                    metavar="SCALE",
                    help="noise channel volume as a linear scale "
                         "(default 1.0; 0.5 = half as loud; 0.0 = muted)")
    ap.add_argument("--format", choices=["ym", "vtx"], default="ym",
                    help="output register-dump format: 'ym' (YM6, two files per song: "
                         "<name>.ym + <name>.ay2.ym; default) or 'vtx' (Vortex Tracker, "
                         "single file with turboAY chipType=2 for dual-AY)")
    args = ap.parse_args(argv)

    # Imported lazily (after arg parsing) so ``--help`` works without dragging in the heavy
    # neural / numba stack.
    from audio2ay3.cli import _write_song
    from audio2ay3.config import ChipConfig, RunConfig
    from audio2ay3.pipeline import convert
    from audio2ay3.render import Renderer

    stems_dir_path = Path(args.stems_dir) if args.stems_dir else None
    out_dir = Path(args.out_dir)

    # --- Build the (song_name, audio_path) input list ---
    # When --stems-only: discover songs from the stems directory itself; --in-dir is ignored.
    # Each subfolder must contain a ``<folder> (Synth).<ext>`` file.
    # When not --stems-only: use --in-dir as before (with optional Demucs fallback).
    if args.stems_only:
        if stems_dir_path is None:
            print("error: --stems-only requires --stems-dir", file=sys.stderr)
            return 2
        if not stems_dir_path.is_dir():
            print(f"error: stems directory not found: {stems_dir_path}", file=sys.stderr)
            return 2
        inputs: list[tuple[str, Path]] = []  # (song_name, synth_audio_path)
        for folder in sorted(stems_dir_path.iterdir()):
            if not folder.is_dir():
                continue
            # Accept any audio file whose stem contains "(Synth)" (case-insensitive).
            # The file name prefix need not match the folder name.
            synth = None
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in _AUDIO_EXTS \
                        and "(synth)" in f.stem.lower():
                    synth = f
                    break
            if synth is not None:
                inputs.append((folder.name, synth))
        if not inputs:
            print(f"error: no Synth stems found in {stems_dir_path}", file=sys.stderr)
            return 2
        source_label = str(stems_dir_path)
    else:
        in_dir = Path(args.in_dir)
        if not in_dir.is_dir():
            print(f"error: input folder not found: {in_dir}", file=sys.stderr)
            return 2
        raw = sorted(
            p for p in in_dir.glob(args.glob)
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
        )
        if not raw:
            print(f"error: no audio files in {in_dir} matching {args.glob!r}", file=sys.stderr)
            return 2
        inputs = [(p.stem, p) for p in raw]
        source_label = str(in_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    reg_ext = ".vtx" if fmt == "vtx" else ".ym"

    cfg = RunConfig(
        chip=ChipConfig(n_chips=2),          # dual-AY: 6 tone channels
        separation=args.separation,
        transcription=args.transcription,
        yourmt3_model=args.model,
        stems_dir=stems_dir_path,
        noise_volume=args.noise_volume,
    )
    renderer = Renderer(render_sr=cfg.render_sr, oversample=cfg.oversample)

    front_end = args.transcription + (f"/{args.model}" if args.model else "")
    separation_label = "pre-separated stems" if stems_dir_path is not None else args.separation
    print(f"Converting {len(inputs)} song(s) from {source_label} -> {out_dir}")
    print(f"  front-end: {front_end} + {separation_label}  |  chips: 2 (dual-AY)  |  format: {fmt}\n")

    failures: list[tuple[str, str]] = []
    for i, (song_name, src) in enumerate(inputs, 1):
        out_reg = out_dir / f"{song_name}{reg_ext}"
        out_mp3 = out_dir / f"{song_name}.mp3"
        if out_reg.exists() and out_mp3.exists() and not args.force:
            print(f"[{i}/{len(inputs)}] skip (exists): {song_name}")
            continue

        print(f"[{i}/{len(inputs)}] {src.name} ...", flush=True)
        t0 = time.time()
        try:
            song = convert(str(src), cfg, name=song_name)      # one neural pass
            reg_paths = _write_song(song, str(out_reg), fmt)
            renderer.render_to_file(song, str(out_mp3), bitrate_kbps=cfg.mp3_bitrate_kbps)
        except Exception as exc:  # batch driver: report and keep going to the next song
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append((song_name, f"{type(exc).__name__}: {exc}"))
            continue

        dt = (time.time() - t0) / 60.0
        print(f"    ok: {song.n_frames} frames ({song.duration_s:.1f}s) in {dt:.1f} min")
        print(f"    -> {', '.join(reg_paths)}")
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
