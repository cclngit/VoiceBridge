use super::resampler::AudioConverter;
use std::collections::VecDeque;

/// A causal streaming audio buffer designed for real-time neural audio codecs (such as Mimi)
/// and streaming ASR systems (Delayed Streams Modeling / Moshi).
///
/// Converts incoming variable-sized PCM audio buffers (from CPAL) into fixed 80ms causal frames.
/// For 24,000 Hz (Mimi / Moshi standard), 80ms corresponds to exactly 1,920 mono f32 samples.
/// For 16,000 Hz (Whisper standard), 80ms corresponds to 1,280 samples.
#[allow(dead_code)]
pub struct StreamingCausalAudioBuffer {
    converter: AudioConverter,
    target_sample_rate: u32,
    frame_samples: usize,
    sample_queue: VecDeque<f32>,
    step_count: u64,
    total_samples_received: u64,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct AudioFrame80ms {
    pub step: u64,
    pub timestamp_ms: u64,
    pub samples: Vec<f32>,
    pub rms_energy: f32,
    pub is_silence: bool,
}

#[allow(dead_code)]
impl StreamingCausalAudioBuffer {
    /// Creates a new causal buffer with the given source and target sample rate.
    /// `frame_duration_ms` defaults to 80ms if None is specified (standard for Mimi at 12.5 Hz).
    pub fn new(source_rate: u32, target_sample_rate: u32, channels: usize) -> Self {
        let frame_duration_ms = 80;
        let frame_samples = ((target_sample_rate as u64 * frame_duration_ms) / 1000) as usize;
        let converter = AudioConverter::new(source_rate, target_sample_rate, channels);

        Self {
            converter,
            target_sample_rate,
            frame_samples: frame_samples.max(1),
            sample_queue: VecDeque::with_capacity(frame_samples * 8),
            step_count: 0,
            total_samples_received: 0,
        }
    }

    /// Target sample rate in Hz (e.g. 24000 or 16000).
    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    /// Number of samples in an 80ms frame (e.g. 1920 samples at 24kHz).
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Total number of 80ms frames emitted so far.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Push incoming interleaved audio samples from CPAL, resample to target rate,
    /// and pop all available complete 80ms causal frames.
    pub fn push_and_drain_frames(&mut self, interleaved: &[f32]) -> Vec<AudioFrame80ms> {
        let resampled = self.converter.process(interleaved);
        self.total_samples_received += resampled.len() as u64;
        self.sample_queue.extend(resampled);

        let mut frames = Vec::new();
        while self.sample_queue.len() >= self.frame_samples {
            let mut frame_samples = Vec::with_capacity(self.frame_samples);
            for _ in 0..self.frame_samples {
                if let Some(s) = self.sample_queue.pop_front() {
                    frame_samples.push(s);
                }
            }

            let rms = super::resampler::calculate_rms(&frame_samples);
            let timestamp_ms = self.step_count * 80;
            let is_silence = rms < 0.003;

            frames.push(AudioFrame80ms {
                step: self.step_count,
                timestamp_ms,
                samples: frame_samples,
                rms_energy: rms,
                is_silence,
            });

            self.step_count += 1;
        }

        frames
    }

    /// Resets the buffer state and sample queue.
    pub fn reset(&mut self) {
        self.sample_queue.clear();
        self.step_count = 0;
        self.total_samples_received = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_buffer_frame_size_at_24khz() {
        // 24000 Hz * 80ms / 1000 = 1920 samples
        let mut buffer = StreamingCausalAudioBuffer::new(24000, 24000, 1);
        assert_eq!(buffer.frame_samples(), 1920);

        // Feed exactly 1920 samples
        let input = vec![0.1f32; 1920];
        let frames = buffer.push_and_drain_frames(&input);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples.len(), 1920);
        assert_eq!(frames[0].step, 0);
        assert_eq!(frames[0].timestamp_ms, 0);
        assert!(!frames[0].is_silence);

        // Feed half a frame (960 samples) -> 0 frames emitted
        let half = vec![0.1f32; 960];
        let frames_half = buffer.push_and_drain_frames(&half);
        assert_eq!(frames_half.len(), 0);

        // Feed another half -> 1 frame emitted at step 1 (timestamp 80ms)
        let frames_next = buffer.push_and_drain_frames(&half);
        assert_eq!(frames_next.len(), 1);
        assert_eq!(frames_next[0].step, 1);
        assert_eq!(frames_next[0].timestamp_ms, 80);
    }

    #[test]
    fn test_causal_buffer_resampling_48k_to_24k() {
        let mut buffer = StreamingCausalAudioBuffer::new(48000, 24000, 2);
        assert_eq!(buffer.frame_samples(), 1920);

        // 48kHz stereo for 80ms = 3840 stereo frames = 7680 interleaved floats
        let input = vec![0.2f32; 7680];
        let frames = buffer.push_and_drain_frames(&input);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples.len(), 1920);
    }
}
