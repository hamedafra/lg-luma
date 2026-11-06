//! Quiet navigation clicks, synthesized once at start-up: no audio assets ship with Luma.
//! The idea follows funnymataleao/luma-webos-launcher (MIT): a short damped two-partial tone
//! queued through SDL. A missing audio device leaves Luma silent instead of failing.
use sdl2::audio::{AudioQueue, AudioSpecDesired};

const RATE: i32 = 48_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sound {
    /// Focus moved, a page or panel changed.
    Move,
    /// OK on a control.
    Select,
}

pub struct UiSound {
    _audio: sdl2::AudioSubsystem,
    queue: AudioQueue<i16>,
    tick: Vec<i16>,
    select: Vec<i16>,
    active: bool,
}

/// A soft "tock": a bright partial for the click and a lower one for body, each with a 1 ms
/// attack and an exponential decay, so it never pops or rings.
fn tone(high: f32, low: f32, decay_ms: f32, gain: f32) -> Vec<i16> {
    let length = (RATE as f32 * decay_ms * 5. / 1000.) as usize;
    (0..length)
        .map(|n| {
            let t = n as f32 / RATE as f32;
            let attack = (t / 0.001).min(1.);
            let body = (-t / (decay_ms / 1000.)).exp();
            let click = (-t / (decay_ms / 2500.)).exp();
            let wave = 0.6 * (std::f32::consts::TAU * high * t).sin() * click
                + 0.4 * (std::f32::consts::TAU * low * t).sin() * body;
            (wave * attack * gain * f32::from(i16::MAX)) as i16
        })
        .collect()
}

impl UiSound {
    pub fn new(sdl: &sdl2::Sdl) -> Option<Self> {
        let audio = sdl
            .audio()
            .map_err(|error| eprintln!("Luma navigation sounds unavailable: {error}"))
            .ok()?;
        let spec = AudioSpecDesired {
            freq: Some(RATE),
            channels: Some(1),
            samples: Some(512),
        };
        let queue = audio
            .open_queue::<i16, _>(None, &spec)
            .map_err(|error| eprintln!("Luma navigation sounds unavailable: {error}"))
            .ok()?;
        Some(Self {
            _audio: audio,
            queue,
            tick: tone(2100., 420., 9., 0.10),
            select: tone(1500., 300., 14., 0.13),
            active: false,
        })
    }

    /// Keeps the device paused while Luma is hidden, so it never holds the TV's audio.
    pub fn set_active(&mut self, active: bool) {
        if self.active != active {
            self.active = active;
            if active {
                self.queue.resume();
            } else {
                self.queue.clear();
                self.queue.pause();
            }
        }
    }

    pub fn play(&self, sound: Sound) {
        if !self.active {
            return;
        }
        // Rapid moves restart the click instead of building a backlog.
        self.queue.clear();
        let _ = self.queue.queue_audio(match sound {
            Sound::Move => &self.tick,
            Sound::Select => &self.select,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clicks_are_short_quiet_and_start_and_end_near_silence() {
        for samples in [tone(2100., 420., 9., 0.10), tone(1500., 300., 14., 0.13)] {
            assert!(samples.len() < RATE as usize / 10, "under 100 ms");
            let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap();
            assert!(peak > 500 && peak < i16::MAX as u16 / 6, "{peak}");
            assert_eq!(samples[0], 0);
            assert!(samples.last().unwrap().unsigned_abs() < 200);
        }
    }
}
