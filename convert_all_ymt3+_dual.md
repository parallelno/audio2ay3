Task: batch-convert all long samples to dual-AY YourMT3 (YMT3+), .ym + .mp3.

0) Pull latest (needs this session's YMT3+ preset + --yourmt3-model wiring + the new
   scripts). Confirm the driver is present:
     git pull
     Test-Path scripts\convert_long_ymt3plus_dual.py   # must be True

1) Deps (one-time): pure-PyTorch + mp3 encoder, plus the GPL YourMT3 backend in the cache:
     pip install -e ".[yourmt3,mp3]"
     python -m audio2ay3 setup-yourmt3        # clones the HF Space (YMT3+ ckpt via LFS)
   (If you hit the torchvision/torchmetrics import error: pip install "torchmetrics<1.7" wandb)

2) Sanity-check on ONE song first (end-to-end, ~tens of minutes — it's a heavy model):
     python scripts\convert_long_ymt3plus_dual.py --glob "Goblins_Lair*"
   Expect results\ymt3plus_dual\Goblins_Lair.ym + .ay2.ym + .mp3. Listen to the .mp3
   (it's both AY chips mixed). If good, do the rest.

3) Convert all 6 long songs (sequential — do NOT run anything else heavy alongside it):
     .\convert_all_ymt3plus_dual.bat
   (or: python scripts\convert_long_ymt3plus_dual.py)
   Already-done songs are skipped; pass --force to redo. If a song fails, the batch
   keeps going and lists failures at the end.

4) Report back per song: ym/ay2.ym/mp3 written? inference minutes? and a listening note
   vs the previous single-chip YMT3 results — does dual-AY fill in the missing notes
   (more simultaneous voices) without sounding muddy?