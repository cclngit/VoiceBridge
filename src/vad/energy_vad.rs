use crate::audio::calculate_rms;
use std::collections::VecDeque;

#[derive(Debug, PartialEq, Eq)]
pub enum VadState {
    Silence,
    Speaking,
}

pub struct SpeechSegmenter {
    #[allow(dead_code)]
    _sample_rate: u32,
    threshold: f32,
    silence_duration_samples: usize,
    min_speech_samples: usize,
    max_speech_samples: usize,

    state: VadState,
    pre_speech_ring: VecDeque<f32>,
    pre_speech_capacity: usize,
    speech_buffer: Vec<f32>,
    consecutive_silence_samples: usize,
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
            _sample_rate: sample_rate,
            threshold,
            silence_duration_samples: ms_to_samples(silence_duration_ms),
            min_speech_samples: ms_to_samples(min_speech_ms),
            max_speech_samples: ms_to_samples(max_speech_ms),

            state: VadState::Silence,
            pre_speech_ring: VecDeque::with_capacity(pre_speech_capacity),
            pre_speech_capacity,
            speech_buffer: Vec::new(),
            consecutive_silence_samples: 0,
            adaptive_noise_floor: 0.005,
        }
    }

    #[allow(dead_code)]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(0.001);
    }

    pub fn update_params(
        &mut self,
        threshold: f32,
        silence_duration_ms: u64,
        min_speech_ms: u64,
        max_speech_ms: u64,
    ) {
        let ms_to_samples = |ms: u64| (ms as f64 * self._sample_rate as f64 / 1000.0) as usize;
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
                } else {
                    self.consecutive_silence_samples += chunk.len();
                }

                // Check termination conditions
                let reached_silence_limit =
                    self.consecutive_silence_samples >= self.silence_duration_samples;
                let reached_max_length = self.speech_buffer.len() >= self.max_speech_samples;

                if reached_silence_limit || reached_max_length {
                    let duration_samples = self.speech_buffer.len();

                    if duration_samples >= self.min_speech_samples {
                        // Utterance finalized!
                        let segment = std::mem::take(&mut self.speech_buffer);
                        completed_segment = Some(segment);
                    } else {
                        // Discard too short noise/click
                        self.speech_buffer.clear();
                    }

                    self.state = VadState::Silence;
                    self.consecutive_silence_samples = 0;
                    self.pre_speech_ring.clear();
                }
            }
        }

        completed_segment
    }

    #[allow(dead_code)]
    pub fn current_state(&self) -> &VadState {
        &self.state
    }
}
