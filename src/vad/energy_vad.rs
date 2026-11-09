use crate::audio::calculate_rms;
use std::collections::VecDeque;

#[derive(Debug, PartialEq, Eq)]
pub enum VadState {
    Silence,
    Speaking,
}

/// Trailing silence kept at the end of an utterance (the rest is trimmed before ASR).
const TRAILING_SILENCE_KEEP_MS: u64 = 250;

pub struct SpeechSegmenter {
    sample_rate: u32,
    threshold: f32,
    silence_duration_samples: usize,
    min_speech_samples: usize,
    max_speech_samples: usize,

    state: VadState,
    pre_speech_ring: VecDeque<f32>,
    pre_speech_capacity: usize,
    speech_buffer: Vec<f32>,
    consecutive_silence_samples: usize,
    /// Samples classified as voice in the current utterance (pre-roll and silence excluded).
    voiced_samples: usize,
    adaptive_noise_floor: f32,
}

impl SpeechSegmenter {
    pub fn new(
        sample_rate: u32,
        threshold: f32,
        silence_duration_ms: u64,
        min_speech_ms: u64,
        max_speech_ms: u64,
    ) -> Self {
        let ms_to_samples = |ms: u64| (ms as f64 * sample_rate as f64 / 1000.0) as usize;
        let pre_speech_capacity = ms_to_samples(250); // 250ms pre-speech buffer

        Self {
            sample_rate,
            threshold,
            silence_duration_samples: ms_to_samples(silence_duration_ms),
            min_speech_samples: ms_to_samples(min_speech_ms),
            max_speech_samples: ms_to_samples(max_speech_ms),

            state: VadState::Silence,
            pre_speech_ring: VecDeque::with_capacity(pre_speech_capacity),
            pre_speech_capacity,
            speech_buffer: Vec::new(),
            consecutive_silence_samples: 0,
            voiced_samples: 0,
            adaptive_noise_floor: 0.005,
        }
    }

    pub fn update_params(
        &mut self,
        threshold: f32,
        silence_duration_ms: u64,
        min_speech_ms: u64,
        max_speech_ms: u64,
    ) {
        let rate = self.sample_rate as f64;
        let ms_to_samples = |ms: u64| (ms as f64 * rate / 1000.0) as usize;
        self.threshold = threshold.max(0.001);
        self.silence_duration_samples = ms_to_samples(silence_duration_ms);
        self.min_speech_samples = ms_to_samples(min_speech_ms);
        self.max_speech_samples = ms_to_samples(max_speech_ms);
    }

    /// Process incoming chunk of 16kHz mono audio.
    /// Returns Some(speech_segment) if a completed sentence/utterance was finalized.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        if chunk.is_empty() {
            return None;
        }

        let chunk_rms = calculate_rms(chunk);

        // Update noise floor slowly during silence
        if self.state == VadState::Silence && chunk_rms < self.threshold {
            self.adaptive_noise_floor = self.adaptive_noise_floor * 0.95 + chunk_rms * 0.05;
        }

        let dynamic_threshold = (self.threshold).max(self.adaptive_noise_floor * 2.5);
        let is_voice = chunk_rms > dynamic_threshold;

        let mut completed_segment: Option<Vec<f32>> = None;

        match self.state {
            VadState::Silence => {
                if is_voice {
                    // Speech begins!
                    self.state = VadState::Speaking;
                    self.consecutive_silence_samples = 0;
                    self.voiced_samples = chunk.len();
                    self.speech_buffer.clear();

                    // Prepend pre-roll buffer so words aren't truncated at start
                    self.speech_buffer.extend(self.pre_speech_ring.drain(..));
                    self.speech_buffer.extend_from_slice(chunk);
                } else {
                    // Still silence: keep rolling pre-speech buffer
                    for &sample in chunk {
                        if self.pre_speech_ring.len() >= self.pre_speech_capacity {
                            self.pre_speech_ring.pop_front();
                        }
                        self.pre_speech_ring.push_back(sample);
                    }
                }
            }
            VadState::Speaking => {
                self.speech_buffer.extend_from_slice(chunk);

                if is_voice {
                    self.consecutive_silence_samples = 0;
                    self.voiced_samples += chunk.len();
                } else {
                    self.consecutive_silence_samples += chunk.len();
                }

                // Check termination conditions
                let reached_silence_limit =
                    self.consecutive_silence_samples >= self.silence_duration_samples;
                let reached_max_length = self.speech_buffer.len() >= self.max_speech_samples;

                if reached_silence_limit || reached_max_length {
                    // Only genuine voiced audio counts toward the minimum duration, otherwise
                    // pre-roll + trailing silence would let every click through to the ASR.
                    if self.voiced_samples >= self.min_speech_samples {
                        let keep = (TRAILING_SILENCE_KEEP_MS * self.sample_rate as u64 / 1000) as usize;
                        let excess = self.consecutive_silence_samples.saturating_sub(keep);
                        let new_len = self.speech_buffer.len().saturating_sub(excess);
                        self.speech_buffer.truncate(new_len);
                        let segment = std::mem::take(&mut self.speech_buffer);
                        completed_segment = Some(segment);
                    } else {
                        // Discard too short noise/click
                        self.speech_buffer.clear();
                    }

                    self.state = VadState::Silence;
                    self.consecutive_silence_samples = 0;
                    self.voiced_samples = 0;
                    self.pre_speech_ring.clear();
                }
            }
        }

        completed_segment
    }

    #[cfg(test)]
    pub fn current_state(&self) -> &VadState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn chunk(level: f32) -> Vec<f32> {
        // 20 ms chunks, alternating sign so RMS == level.
        (0..320).map(|i| if i % 2 == 0 { level } else { -level }).collect()
    }

    fn feed(seg: &mut SpeechSegmenter, level: f32, ms: usize) -> Vec<Vec<f32>> {
        (0..ms / 20).filter_map(|_| seg.process_chunk(&chunk(level))).collect()
    }

    #[test]
    fn short_click_is_discarded() {
        let mut seg = SpeechSegmenter::new(RATE, 0.015, 700, 350, 15_000);
        feed(&mut seg, 0.0005, 500);
        // 100 ms of "voice" followed by 1 s of silence: shorter than the 350 ms minimum.
        let mut out = feed(&mut seg, 0.2, 100);
        out.extend(feed(&mut seg, 0.0005, 1000));
        assert!(out.is_empty(), "a 100 ms click must not produce a segment");
        assert_eq!(seg.current_state(), &VadState::Silence);
    }

    #[test]
    fn utterance_is_emitted_with_trimmed_trailing_silence() {
        let mut seg = SpeechSegmenter::new(RATE, 0.015, 700, 350, 15_000);
        feed(&mut seg, 0.0005, 500);
        let mut out = feed(&mut seg, 0.2, 1000);
        out.extend(feed(&mut seg, 0.0005, 1000));
        assert_eq!(out.len(), 1);
        let secs = out[0].len() as f32 / RATE as f32;
        // 1 s voice + <= 250 ms pre-roll + 250 ms kept trailing silence.
        assert!(secs > 1.0 && secs < 1.6, "segment length {secs}s");
    }

    #[test]
    fn long_speech_is_split_at_max_length() {
        let mut seg = SpeechSegmenter::new(RATE, 0.015, 700, 350, 2_000);
        let out = feed(&mut seg, 0.2, 5_000);
        assert_eq!(out.len(), 2);
    }
}
