Good question — it gets at the real ceiling: GM identity tells us what instrument it is, but the AY's palette (3 square waves + 1 noise + 1 shared envelope) is what limits how faithfully we can imitate it. Let me check what the design already plans for timbre/effects so I align with it rather than reinvent.

Yes — and your organ/pipe observation points exactly at where the remaining headroom is. Let me separate what MT3/GM actually gives us from what the AY can do with it, because the ceiling here is the chip, not the data.

What MT3/GM exposes (and its hard limits)
Per note: pitch, start/end, velocity, program (GM instrument), is_drum, and an instrument/track index. That's it — MT3 is note-level, so there is no pitch-bend, CC, portamento, or articulation, and its multi-instrument checkpoint uses a single velocity bin (every note floors to full, which is why we already derive dynamics from the source loudness contour, not velocity).

So the untapped MT3/GM signal is really two things: the GM program (instrument identity — we've started using it) and the instrument/track grouping (which notes belong to the same voice). What we can't get is expression; we have to synthesize idiomatic expression from the instrument identity.

Why organ/pipe sound "pure"
The AY plays a bare 50%-duty square — no filter, no PWM (fixed duty), one shared hardware envelope. The design is explicit (07-sound-quality-strategy.md:139): we don't chase spectral matching; perceived richness must come from arrangement + register-level chip tricks, not DSP. A static square will never be an organ — but the classic chiptune techniques make it read as one. GM identity is exactly the switch that picks the right trick per instrument:

GM family	AY technique	Cost	Touches a shared resource?
Pipe/Flute, Strings, Violin, Sax, Organ, Synth Lead	Vibrato (LFO on the tone period, ~5–7 Hz)	cheap — just R0–R5	No
Flute/Pan-flute, breathy reeds	Noise blend (mixer adds a touch of noise)	cheap	Noise gen (vs drums)
Organ, Reed, Brass, Synth Lead	Hardware-envelope "buzzer" (repeating 0x08/0x0A/0x0C/0x0E → reedy/buzzy tone)	medium	Envelope (vs drums) — and you wanted this opt-in
Strings/Pads/Ensemble	Detune/chorus (two channels a few cents apart)	expensive — 2 channels	Channels (needs dual-AY)
Chords/fuller texture	Arpeggio (cycle chord tones on one channel)	cheap but busy	No
These are all register-level states a real chip holds, so they're in-scope (distinct from the forbidden post-render chorus/EQ).

My recommendation
Instrument-idiomatic vibrato first. It's the natural next step after priority → sustain, it's the single best bang-for-buck for the exact instruments you flagged (pipe/organ/strings/lead), and — unlike the buzzer — it doesn't touch the contended envelope or noise generator, so there's no percussion conflict and it needs no opt-in flag. A 5–7 Hz pitch wobble at 50 fps is very achievable and is the trick that makes a dead square read as a flute or a living organ.

The hardware-envelope buzzer is the design's documented tool for genuine reedy/organ richness and would help even more on timbre — but it's the single shared envelope (contends with drums) and you specifically wanted it behind an opt-in flag, so I'd stage it second: auto-offer it to one sustained "feature" voice only on frames where no drum needs the envelope.