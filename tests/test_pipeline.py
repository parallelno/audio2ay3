"""Tests for the deterministic arrange() core of the conversion pipeline.

These exercise the neural-free half end-to-end: a synthetic Transcription becomes a
hardware-legal YmSong that the emulator can render. The neural front-end (load/separate/
transcribe) needs the optional [neural] extra and is not unit-tested here.
"""

from __future__ import annotations

import numpy as np

from audio2ay3.analysis.model import Note, Percussion, Transcription
from audio2ay3.config import AmpEnvelope, RunConfig
from audio2ay3.encode.quantize import quantize_tone
from audio2ay3.pipeline import arrange
from audio2ay3.render import Renderer

CLOCK = 1_773_400


def test_arrange_produces_legal_song_with_expected_length():
    tr = Transcription(
        notes=[Note(0.0, 0.1, 440.0, 1.0)], percussion=[], duration_s=0.0
    )
    song = arrange(tr, RunConfig(), name="x")
    assert song.frames.shape == (5, 16)  # ceil(0.1s * 50)
    assert song.master_clock == CLOCK
    assert song.frame_rate == 50
    assert song.version == "YM6"
    # Reserved/IO registers stay clean, envelope left untouched.
    assert np.all(song.frames[:, 13] == 0xFF)


def test_arrange_maps_pitch_to_correct_tone_period():
    tr = Transcription(notes=[Note(0.0, 0.1, 440.0, 1.0)])
    song = arrange(tr, RunConfig())
    expected_tp = quantize_tone(440.0, CLOCK)  # 252
    assert song.frames[0, 0] == (expected_tp & 0xFF)
    assert song.frames[0, 1] == ((expected_tp >> 8) & 0x0F)
    assert song.frames[0, 8] == 15  # velocity 1.0 -> full amplitude
    assert song.frames[0, 7] & 0x01 == 0  # tone A enabled


def test_arrange_is_renderable_and_audible():
    tr = Transcription(notes=[Note(0.0, 0.3, 330.0, 1.0)])
    song = arrange(tr, RunConfig())
    pcm = Renderer(render_sr=22_050, oversample=1).render(song)
    assert pcm.size == int(song.n_frames * 22_050 / 50)
    assert np.max(np.abs(pcm)) > 0.1  # not silent


def test_arrange_percussion_only_programs_noise():
    tr = Transcription(
        notes=[], percussion=[Percussion(0.0, "snare", 1.0)], duration_s=0.2
    )
    song = arrange(tr, RunConfig())
    # The snare frame routes noise to channel C (mixer bit 5 cleared).
    assert song.frames[0, 7] & (1 << 5) == 0
    assert song.frames[0, 6] > 0  # noise period set


def test_arrange_empty_transcription_is_one_silent_frame():
    song = arrange(Transcription(), RunConfig())
    assert song.n_frames == 1
    assert np.all(song.frames[:, 7] == 0x3F)  # everything disabled


def test_arrange_bass_and_lead_land_on_separate_channels():
    # The Test-1 mono-collapse case: a low bass note and a high lead sounding together must
    # both survive — bass on its dedicated channel A, lead on a free channel.
    tr = Transcription(
        notes=[Note(0.0, 0.2, 440.0, 1.0)],  # lead
        bass_notes=[Note(0.0, 0.2, 80.0, 1.0)],  # bass
    )
    song = arrange(tr, RunConfig())

    def tone_period(frame, ch):
        return int(song.frames[frame, 2 * ch]) | (int(song.frames[frame, 2 * ch + 1]) << 8)

    bass_tp = quantize_tone(80.0, CLOCK)
    lead_tp = quantize_tone(440.0, CLOCK)

    # Bass owns channel A; the lead lives on B or C — two distinct audible voices, no collapse.
    assert tone_period(0, 0) == bass_tp
    lead_channels = [ch for ch in (1, 2) if tone_period(0, ch) == lead_tp]
    assert lead_channels, "lead note was dropped instead of getting its own channel"
    mixer = int(song.frames[0, 7])
    assert mixer & 0x01 == 0  # tone A (bass) enabled
    assert mixer & (1 << lead_channels[0]) == 0  # tone on the lead's channel enabled


def test_amp_envelope_level_strikes_then_decays_to_sustain():
    env = AmpEnvelope(attack_frames=0, decay_frames=10, sustain=0.6)
    assert env.level(0, 15) == 15  # onset strike at peak
    assert env.level(1, 15) < 15  # decaying
    # Non-increasing across the decay, settling at the sustain level (round(15*0.6)=9).
    levels = [env.level(a, 15) for a in range(20)]
    assert levels == sorted(levels, reverse=True)
    assert levels[-1] == 9
    assert env.level(0, 0) == 0  # a silent note stays silent


def test_amp_envelope_disabled_is_flat():
    env = AmpEnvelope(enabled=False)
    assert env.level(0, 12) == env.level(5, 12) == env.level(50, 12) == 12


def test_arrange_amplitude_envelope_decays_and_restrikes():
    # Two back-to-back lead notes: each should strike at full and decay, the second re-striking.
    tr = Transcription(notes=[Note(0.0, 0.4, 440.0, 1.0), Note(0.4, 0.4, 440.0, 1.0)])
    song = arrange(tr, RunConfig())  # no bass -> lead lands on channel A (R8)
    amp = song.frames[:, 8].astype(int)
    assert amp[0] == 15  # first strike
    assert amp[19] < amp[0]  # decayed by the end of the first note
    assert amp[20] == 15  # second note re-strikes to peak
    assert amp[20] > amp[19]


def test_arrange_flat_when_envelope_disabled():
    tr = Transcription(notes=[Note(0.0, 0.4, 440.0, 1.0)])
    cfg = RunConfig(amp_envelope=AmpEnvelope(enabled=False))
    song = arrange(tr, cfg)
    amp = song.frames[:20, 8].astype(int)
    assert set(amp.tolist()) == {15}  # constant, no shaping


def test_arrange_follows_source_amp_contour():
    # A note carrying a source loudness contour: the channel amplitude tracks it frame-by-frame,
    # overriding the synthetic envelope so the note keeps the original's character. The contour
    # is applied in the DAC's logarithmic domain (scale_amplitude), so a 0.4 voltage frame lands
    # near level 11 (~0.35 output), not level 6 a naive level*0.4 would give.
    note = Note(0.0, 0.08, 440.0, velocity=1.0, amp_contour=(1.0, 0.8, 0.4, 0.2))
    song = arrange(Transcription(notes=[note]), RunConfig())  # no bass -> channel A (R8)
    amp = song.frames[:4, 8].astype(int).tolist()
    assert amp == [15, 14, 11, 9]


def test_arrange_strikes_each_onset_at_full_peak():
    # The first frame of every note is struck at its full (velocity-scaled) peak regardless of
    # the source contour, so fast repeated notes keep a sharp attack instead of fusing into one
    # sustained tone. Subsequent frames follow the contour.
    note = Note(0.0, 0.06, 440.0, velocity=1.0, amp_contour=(0.5, 0.5, 0.5))
    song = arrange(Transcription(notes=[note]), RunConfig())
    amp = song.frames[:3, 8].astype(int).tolist()
    assert amp[0] == 15  # struck at peak even though contour[0] = 0.5
    assert amp[1] == 12 and amp[2] == 12  # then follows the contour (scale_amplitude(15, 0.5))


def test_arrange_flat_contour_still_decays_for_liveliness():
    # A held note whose (whole-stem) contour is flat must still get a struck attack-and-decay
    # shape, not a lifeless constant amplitude. The first frame strikes at peak; the synthetic
    # envelope applied on top of the flat contour pulls later frames down to the sustain level.
    note = Note(0.0, 0.3, 440.0, velocity=1.0, amp_contour=(1.0,) * 15)
    song = arrange(Transcription(notes=[note]), RunConfig())  # no bass -> channel A (R8)
    amp = song.frames[:15, 8].astype(int).tolist()
    assert amp[0] == 15  # struck at peak
    assert amp[-1] < amp[0]  # then decays -> not a flat, lifeless sustain
    assert amp == sorted(amp, reverse=True)  # monotonically non-increasing (clean decay)
    assert amp[-1] == 13  # settles at the sustain level (0.6 of peak in the log DAC)


def test_arrange_sustained_instrument_holds_legato_without_struck_decay():
    # The same held note as the decay test, but tagged with a sustained GM family (program 48 =
    # string ensemble), must keep its level for the whole note instead of decaying like a pluck,
    # so a bowed/legato line stays connected rather than fragmenting into short, isolated notes.
    note = Note(0.0, 0.3, 440.0, velocity=1.0, amp_contour=(1.0,) * 15, program=48)
    song = arrange(Transcription(notes=[note]), RunConfig())  # no bass -> channel A (R8)
    amp = song.frames[:15, 8].astype(int).tolist()
    assert amp[0] == 15  # struck at peak
    assert set(amp) == {15}  # held flat for the whole note -> no struck decay


def test_arrange_sustained_instrument_without_contour_holds_flat_at_peak():
    # A sustained instrument that carries no source contour must still hold (not decay), unlike
    # the program-less liveliness path that falls to the synthetic struck envelope.
    note = Note(0.0, 0.3, 440.0, velocity=1.0, program=73)  # flute, no contour
    song = arrange(Transcription(notes=[note]), RunConfig())
    amp = song.frames[:15, 8].astype(int).tolist()
    assert set(amp) == {15}  # legato hold, no per-note decay


def test_arrange_vibrato_wobbles_a_sustained_wind_voice():
    # A flute (GM 73) gets an idiomatic pitch LFO: its tone period must wobble over a long note
    # once the clean-attack delay has passed, staying within a few percent of the true period.
    flute = Note(0.0, 0.6, 440.0, velocity=1.0, program=73)
    song = arrange(Transcription(notes=[flute]), RunConfig())  # no bass -> channel A (R0/R1)
    tp = [int(song.frames[f, 0]) | (int(song.frames[f, 1]) << 8) for f in range(30)]
    base = quantize_tone(440.0, CLOCK)
    assert len(set(tp[10:30])) > 1  # wobbles once vibrato has ramped in
    assert all(abs(p - base) <= base // 15 + 2 for p in tp)  # stays near the true pitch


def test_arrange_no_vibrato_on_a_struck_instrument():
    # A piano (GM 0) is not a vibrato family, so its tone period is rock-steady.
    piano = Note(0.0, 0.6, 440.0, velocity=1.0, program=0)
    song = arrange(Transcription(notes=[piano]), RunConfig())
    tp = {int(song.frames[f, 0]) | (int(song.frames[f, 1]) << 8) for f in range(30)}
    assert len(tp) == 1  # steady pitch


def test_arrange_breath_chiff_adds_noise_at_a_wind_attack():
    # A flute (GM 73) gets a brief noise chiff at its attack: noise enabled on its own channel
    # for the first frames (mixer noise-A bit cleared, R6 set), then a clean tone.
    flute = Note(0.0, 0.4, 440.0, velocity=1.0, program=73)
    song = arrange(Transcription(notes=[flute]), RunConfig())  # no bass -> channel A
    r6 = song.frames[:, 6].astype(int)
    r7 = song.frames[:, 7].astype(int)
    assert (r7[0] >> 3) & 1 == 0 and (r7[1] >> 3) & 1 == 0  # noise on A during the chiff
    assert r6[0] == 4 and r6[1] == 4  # breath noise period
    assert (r7[2] >> 3) & 1 == 1  # chiff over -> noise off on A, clean tone after


def test_arrange_breath_chiff_yields_to_drums():
    # When a wind attack coincides with a drum, the drum owns the shared noise generator, so the
    # breath chiff is suppressed: channel A's noise bit is never enabled.
    flute = Note(0.0, 0.4, 440.0, velocity=1.0, program=73)
    tr = Transcription(notes=[flute], percussion=[Percussion(0.0, "snare")])
    song = arrange(tr, RunConfig())  # flute -> channel A, snare -> channel C
    r7 = song.frames[:4, 7].astype(int)
    assert all((v >> 3) & 1 == 1 for v in r7)  # noise-A never enabled (breath deferred)




def test_arrange_ignores_contour_when_envelope_disabled():
    note = Note(0.0, 0.08, 440.0, velocity=1.0, amp_contour=(1.0, 0.5, 0.5, 0.25))
    cfg = RunConfig(amp_envelope=AmpEnvelope(enabled=False))
    song = arrange(Transcription(notes=[note]), cfg)
    amp = song.frames[:4, 8].astype(int)
    assert set(amp.tolist()) == {15}  # flat: the contour is ignored when shaping is off


