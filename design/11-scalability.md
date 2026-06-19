# 11 — Scalability (Dual-AY & 100 Hz)

The brief requires the architecture to support **dual-AY chip configurations** and **50 Hz and
100 Hz update rates** without a redesign. Both are expressed as **configuration**, not new code
paths, because the pipeline contracts were built around `ChipConfig`
(§[04](04-system-architecture.md#46-configuration-model)).

```python
@dataclass(frozen=True)
class ChipConfig:
    master_clock_hz: int = 1_773_400
    frame_rate_hz: int = 50      # → 100 supported
    n_chips: int = 1             # → 2 supported (dual-AY)
    tone_channels: int = 3       # per chip
```

The **voice/resource budget** the mapping stage targets is derived from this config:

```
total_tone_channels = n_chips * tone_channels      # 3 or 6
total_noise_gens    = n_chips                       # 1 or 2
total_envelopes     = n_chips                       # 1 or 2
frames_per_second   = frame_rate_hz                 # 50 or 100
```

Nothing upstream of mapping changes — analysis already detects more candidates than one chip can
use (§[06](06-conversion-pipeline.md#stage-3a--multi-f0-polyphonic-pitch-estimation)).

## 11.1 Dual-AY (6 tone channels)

Two AY-3-8910s, summed at output. This roughly **doubles the harmonic budget** and gives a
**second independent noise + envelope** generator — a large fidelity upgrade.

### What changes (and what doesn't)

| Concern | Single AY | Dual AY | Code impact |
|---------|-----------|---------|-------------|
| Tone channels | 3 | 6 | mapping budget param |
| Noise gens | 1 | 2 | percussion can run two drum voices |
| Envelopes | 1 | 2 | two simultaneous buzzer/decay roles |
| Emulation | one `Ay3Emulator` | two instances, summed | `render` loops over chips |
| Output file | one register stream | **two** register banks | YM writer / convention |

### Voice allocation with 6 channels

- The assignment problem (§[06](06-conversion-pipeline.md#41-voice-allocation-the-central-problem))
  generalises directly: 6 slots instead of 3, same cost function (continuity, salience, role).
- **Chip affinity:** keep a voice on the same *chip* (not just channel) for stable stereo image
  if the two chips are panned (common dual-AY "ABC/ABC stereo" layouts).
- **Role split option:** dedicate chip 1 to melody+harmony and chip 2 to bass+percussion+pads —
  a research lever exposed as a policy, not hard-coded.

### Output representation for dual-AY

A single mono `.ym` describes one PSG. Options, in preference order:

1. **Two `.ym` files** (`name.ay0.ym`, `name.ay1.ym`) — simplest, every player compatible; the
   pair is played in sync. Default.
2. **Interleaved/extended container** (e.g. an `.ay`/custom dual stream) — later, if a concrete
   target player needs it.

The in-memory model already supports this: `encode` produces a `list[YmSong]` (length =
`n_chips`); for `n_chips == 1` the list has one element, preserving today's behaviour.

### Emulator

`render` sums `n_chips` emulator instances. With panning, chip 0 → left bias, chip 1 → right
(or true stereo). No change to the per-chip core
(§[05](05-emulator-design.md#54-emulator-api)).

## 11.2 100 Hz update rate

Doubling the frame rate to 100 Hz (10 ms/frame) doubles temporal resolution for **faster
arpeggios, tighter percussion, and smoother envelopes** — at the cost of double the register
data and a play routine that can service 100 Hz on the target machine.

### What changes

| Concern | Impact |
|---------|--------|
| Analysis hop | frame grid becomes 10 ms; analysis already supports a configurable hop |
| Quantisation | identical math; periods computed from the same clock (period is rate-independent) |
| Smoothing thresholds | hysteresis/slew windows are expressed in **frames**, so re-tune (or specify in **ms** and convert) |
| YM header | `frameRate = 100`; players that honour the header reproduce correctly |
| Data size | ~2× frames; LHA packing mitigates |
| Performance | ~2× analysis/emulation work; covered by the perf plan (§[09](09-performance-acceleration.md)) |

### Important nuance

- **Tone/noise/envelope periods are NOT frame-rate dependent** — they derive from `master_clock`.
  Only *how often we may change them* doubles. So 100 Hz buys **finer control**, not finer pitch.
- Smoothing parameters should be authored in **milliseconds** and converted to frames at runtime,
  so the same profile behaves consistently at 50 and 100 Hz.

## 11.3 Combined: dual-AY @ 100 Hz

The maximal config (6 tone channels, 2 noise, 2 envelopes, 100 Hz) is just
`ChipConfig(n_chips=2, frame_rate_hz=100)`. All stages already parameterise on these fields;
performance scales as described in §[09](09-performance-acceleration.md). This is the headroom
the architecture deliberately reserves.

## 11.4 Guardrails so scalability stays free

To ensure these remain config-only forever:

- **No hard-coded "3"** anywhere — channel counts come from `ChipConfig`. (Lint/test check.)
- **No hard-coded "50"/"20 ms"** — frame timing derives from `frame_rate_hz`; smoothing in ms.
- **Encode returns a list of `YmSong`** even for one chip.
- **Tests run a dual-AY / 100 Hz smoke case** early (even before full tuning) to keep the paths
  alive and prevent accidental single-chip assumptions creeping in.
