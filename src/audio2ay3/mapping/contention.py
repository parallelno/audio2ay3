"""Quantify how many melodic notes the 3-channel AY budget forces the arranger to drop.

The arranger has three tone channels: A is the dedicated bass, B and C carry melody/harmony, and
C is *also* stolen by every drum hit. So whenever bass and drums sound together the melody is
squeezed onto a single channel (B), and any extra simultaneous notes are dropped for that frame.
This module measures that loss and estimates how much a second AY chip would recover. It re-runs
only the deterministic arrange-time policy (no neural transcription), so it is cheap and exact for
the question "given these notes, how many could the chip actually voice?".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis.model import Transcription
from ..config import RunConfig
from ..encode.quantize import frames_for_duration
from .percussion import PERCUSSION_CHANNEL, percussion_busy_frames
from .voices import (
    N_CHANNELS,
    _bucket_by_frame,
    _priority,
    _spans_from_notes,
    allocate_voices,
    place_bass,
)

# A second AY adds three channels. A natural dual-chip role split dedicates one channel to bass
# and one to percussion, leaving four free for melody/harmony (versus the one or two a single chip
# spares once bass and drums are both playing).
DUAL_MELODIC_CHANNELS = 4


@dataclass(frozen=True)
class ContentionStats:
    """Per-conversion accounting of melodic notes lost to the channel budget."""

    frames: int
    frame_rate_hz: int
    melodic_notes: int
    notes_silenced: int  # melodic notes that never sound on any frame (fully starved)
    demanded_note_frames: int  # sum over frames of simultaneously-active melodic notes
    sounded_note_frames: int  # of those, how many actually reach a channel and survive
    dropped_capacity: int  # note-frames the allocator had no free channel for
    dropped_to_drums: int  # melodic note-frames overwritten by a drum on channel C
    contention_frames: int  # frames where some active note could not be voiced
    bass_frames: int  # frames bass holds channel A
    drum_frames: int  # frames a drum decay occupies channel C
    demand_hist: tuple[int, ...]  # frames wanting 0,1,2,3,4,5+ simultaneous melodic notes
    dual_sounded_note_frames: int  # estimate under a 2nd AY (4 melodic channels, drums isolated)
    dual_notes_silenced: int  # melodic notes still never sounding under the dual-chip estimate

    @property
    def dropped_note_frames(self) -> int:
        return self.demanded_note_frames - self.sounded_note_frames


def _end_seconds(tr: Transcription) -> float:
    """Total length the arranger covers — must match :func:`pipeline.arrange`."""
    return max(
        tr.duration_s,
        max((n.offset_s for n in tr.notes), default=0.0),
        max((n.offset_s for n in tr.bass_notes), default=0.0),
        (max(p.onset_s for p in tr.percussion) + 0.1) if tr.percussion else 0.0,
    )


def voice_contention(tr: Transcription, cfg: RunConfig) -> ContentionStats:
    """Replay the arrange-time voice allocation and tally what the channel budget drops."""
    frame_rate = cfg.chip.frame_rate_hz
    n_frames = frames_for_duration(_end_seconds(tr), frame_rate)

    active_by_frame = _bucket_by_frame(
        _spans_from_notes(tr.notes, frame_rate, n_frames), n_frames
    )
    _, reserved = place_bass(tr.bass_notes, frame_rate, n_frames)
    assignment = allocate_voices(tr.notes, frame_rate, n_frames, reserved=reserved)
    drum_busy = percussion_busy_frames(tr.percussion, frame_rate, n_frames)

    demanded = sounded = dropped_to_drums = 0
    contention_frames = bass_frames = drum_frames = 0
    dual_sounded = 0
    sounded_ids: set[int] = set()
    dual_ids: set[int] = set()
    hist = [0, 0, 0, 0, 0, 0]

    for f in range(n_frames):
        active = active_by_frame[f]
        demand = len(active)
        demanded += demand
        hist[min(demand, 5)] += 1
        if reserved[f] is not None:
            bass_frames += 1
        if drum_busy[f]:
            drum_frames += 1

        # Real single-chip outcome: replay the allocator, then let a drum overwrite channel C.
        placed = 0
        for ch in range(N_CHANNELS):
            voice = assignment[f][ch]
            if voice is None:
                continue
            if ch == PERCUSSION_CHANNEL and drum_busy[f]:
                dropped_to_drums += 1  # placed, but the drum clobbers it this frame
                continue
            sounded += 1
            placed += 1
            sounded_ids.add(voice.note_id)
        if placed < demand:
            contention_frames += 1

        # Dual-chip estimate: four melodic channels by priority, drums on their own channel.
        if demand:
            ranked = sorted(active, key=_priority)[:DUAL_MELODIC_CHANNELS]
            dual_sounded += len(ranked)
            dual_ids.update(s.note_id for s in ranked)

    melodic = len(tr.notes)
    return ContentionStats(
        frames=n_frames,
        frame_rate_hz=frame_rate,
        melodic_notes=melodic,
        notes_silenced=sum(1 for i in range(melodic) if i not in sounded_ids),
        demanded_note_frames=demanded,
        sounded_note_frames=sounded,
        dropped_capacity=demanded - sounded - dropped_to_drums,
        dropped_to_drums=dropped_to_drums,
        contention_frames=contention_frames,
        bass_frames=bass_frames,
        drum_frames=drum_frames,
        demand_hist=tuple(hist),
        dual_sounded_note_frames=dual_sounded,
        dual_notes_silenced=sum(1 for i in range(melodic) if i not in dual_ids),
    )


def describe_contention(stats: ContentionStats) -> str:
    """Render :class:`ContentionStats` as a readable block for ``--explain``."""

    def pct(part: int, whole: int) -> str:
        return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"

    frames = stats.frames
    dem = stats.demanded_note_frames
    notes = stats.melodic_notes
    hist = stats.demand_hist
    recovered = stats.dual_sounded_note_frames - stats.sounded_note_frames
    return "\n".join(
        [
            "Voice contention (single AY: A=bass, B/C=melody, C also drums):",
            f"  melodic notes:             {notes}",
            f"  notes never sounded:       {stats.notes_silenced} ({pct(stats.notes_silenced, notes)})",
            f"  note-frames demanded:      {dem}",
            f"  note-frames sounded:       {stats.sounded_note_frames} ({pct(stats.sounded_note_frames, dem)})",
            f"  dropped, no free channel:  {stats.dropped_capacity} ({pct(stats.dropped_capacity, dem)})",
            f"  dropped, drum stole C:     {stats.dropped_to_drums} ({pct(stats.dropped_to_drums, dem)})",
            f"  frames with contention:    {stats.contention_frames} ({pct(stats.contention_frames, frames)})",
            f"  bass holds channel A:      {stats.bass_frames} ({pct(stats.bass_frames, frames)})",
            f"  drums steal channel C:     {stats.drum_frames} ({pct(stats.drum_frames, frames)})",
            "  simultaneous melodic demand: "
            + " ".join(f"{i}:{hist[i]}" for i in range(5))
            + f" 5+:{hist[5]}",
            "  estimate with a 2nd AY chip (4 melodic channels, drums isolated):",
            f"    note-frames would sound: {stats.dual_sounded_note_frames} ({pct(stats.dual_sounded_note_frames, dem)})",
            f"    notes still silent:      {stats.dual_notes_silenced} ({pct(stats.dual_notes_silenced, notes)})",
            f"    recovered vs single AY:  +{recovered} note-frames ({pct(recovered, dem)} of demand)",
        ]
    )
