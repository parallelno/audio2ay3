# 09 — Performance & Acceleration

The brief asks us to **use multithreading where beneficial**, **explore GPU/CUDA for
compute-heavy stages**, and **optimise the audio analysis and transformation pipelines**. This
document maps those directives onto the architecture and sets a measurement-driven approach.

> Principle: parallelise the **embarrassingly parallel** and **compute-dense** stages; keep the
> inherently sequential chip emulation lean on CPU. Always profile before optimising.

## 9.1 Where the time goes (expected hotspots)

| Stage | Cost class | Parallelism | Accelerator |
|-------|-----------|-------------|-------------|
| Decode | I/O + light CPU | per-file | — |
| HPSS / STFT / CQT | **compute-dense** | per-frame batched | **GPU** (torch) |
| Multi-F0 (CQT salience / NMF / neural) | **compute-dense** | per-frame / batched | **GPU** |
| NN separation (Demucs) — optional | **very heavy** | batched | **GPU (required)** |
| Onset/percussion | medium | per-frame | GPU optional |
| Mapping (voice alloc, smoothing) | light, sequential-ish | per-segment | CPU (threads) |
| Quantise / assemble | trivial | vectorised | CPU/NumPy |
| **Emulation** | medium, **sequential** | per-frame-chunk | **CPU (Numba)** |
| MP3 encode | light | per-file | CPU |

## 9.2 GPU / CUDA acceleration

**Targets:** the spectral front-end and multi-pitch estimation, which dominate runtime on long
tracks and are naturally batched.

- Use **PyTorch / torchaudio** for STFT, CQT, mel/Constant-Q, and salience computation so the
  same code runs on CUDA or CPU by device selection.
- Batch many frames/Constant-Q columns into single GPU kernels; keep data on-device across
  consecutive spectral ops to avoid PCIe round-trips.
- **Optional NN stem separation (Demucs)** and **neural multi-F0** run on GPU when present;
  these are the biggest beneficiaries.
- **Graceful fallback:** `utils.gpu.select_device()` returns CUDA if available else CPU; the
  pipeline is identical either way. `--no-gpu` forces CPU. No correctness depends on the GPU.

```python
# utils/gpu.py
def select_device(prefer_gpu: bool) -> "torch.device":
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

CUDA specifics:
- Respect `CUDA_VISIBLE_DEVICES`; allow `--device cuda:N` later.
- Use mixed precision (fp16/bf16) for NN stages where it doesn't hurt pitch accuracy.
- Pin host buffers and use async copies for the decode→GPU hand-off on long files.

## 9.3 Multithreading & multiprocessing

| Technique | Used for | Why |
|-----------|----------|-----|
| **Thread pool** | I/O (decode, file writes), and GPU/Numba calls that release the GIL | overlaps I/O with compute |
| **Process pool** | CPU-bound pure-Python stages on independent **time segments** | sidesteps the GIL |
| **Chunked emulation** | render frame-ranges in parallel, carrying chip state per chunk | emulation is sequential *within* a chunk but chunks are independent |
| **Batch (GPU)** | spectral/NN stages | maximises device utilisation |

Design rules:
- The orchestrator (`pipeline.py`) owns concurrency; stages stay pure and thread-agnostic.
- **Determinism preserved:** parallelism is over independent data partitions (segments, frame
  chunks, files) and results are reassembled in order, so `--seed` reproducibility holds.
- `--threads N` (0 = auto = `os.cpu_count()`); GPU batch size auto-tuned to VRAM.

### Emulation parallelism detail

Because the emulator carries explicit state, a long song renders as:

```
split frames into K contiguous chunks
for each chunk in parallel:
    emu = Ay3Emulator(...); emu.load_state(state_at_chunk_start)
    pcm_chunk = emu.render(chunk)
concatenate pcm_chunks (with small cross-fade at seams to hide state warm-up)
```

State at chunk starts is obtained by a fast forward-pass (registers only, no audio) — cheap
because it's just counter math.

## 9.4 Pipeline-level parallelism

- **Stage pipelining:** while the GPU computes spectra for segment *n+1*, the CPU runs mapping
  for segment *n* — classic producer/consumer with bounded queues.
- **Batch CLI** (later): convert many files concurrently with a process pool, one track per
  worker, GPU shared/serialised for the spectral stage.

## 9.5 Memory strategy

- **Stream by segment** (e.g. 10–30 s windows) so minutes-long tracks never fully materialise as
  high-rate spectrograms in RAM/VRAM.
- Use float32 internally; downcast to fp16 only on GPU NN stages.
- Emulator works frame-by-frame, bounding its footprint regardless of song length.

## 9.6 Measurement & tuning

Optimisation is **evidence-driven**:

1. Built-in `--profile-run` flag times each stage and prints a breakdown.
2. Keep a small benchmark set (one short loop, one full track) tracked over time.
3. Optimise only stages that profiling shows dominate; avoid speculative complexity
   (consistent with the project's anti-over-engineering stance).

Targets (modern desktop + mid CUDA GPU):

| Workload | Target |
|----------|--------|
| Emulate (render) a 3-min YM | ≪ real-time (≥10×) on CPU |
| Convert a 3-min track (HPSS + CQT salience) | well under real-time on GPU |
| Convert with Demucs (optional) | bounded by the NN; still practical on GPU |

## 9.7 What we will not prematurely build

- No custom CUDA kernels initially — torch/torchaudio primitives suffice.
- No distributed/multi-GPU orchestration (single-GPU is the design point).
- No micro-optimisation of cold stages (decode, encode) until profiling justifies it.
