/// Audio resampler and format converter to 16kHz Mono f32 for Whisper
pub struct AudioConverter {
    source_rate: u32,
    target_rate: u32,
    channels: usize,
}

impl AudioConverter {
    pub fn new(source_rate: u32, target_rate: u32, channels: usize) -> Self {
        Self {
            source_rate,
            target_rate,
            channels: channels.max(1),
        }
    }

    /// Downmix multi-channel interleaved f32 to mono f32
    pub fn downmix_to_mono(&self, interleaved: &[f32]) -> Vec<f32> {
        if self.channels == 1 {
            return interleaved.to_vec();
        }
        let frame_count = interleaved.len() / self.channels;
        let mut mono = Vec::with_capacity(frame_count);
        let channel_count_f = self.channels as f32;

        for frame in interleaved.chunks_exact(self.channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channel_count_f);
        }
        mono
    }

    /// Resample mono audio using linear interpolation
    pub fn resample_mono(&self, input: &[f32]) -> Vec<f32> {
        if self.source_rate == self.target_rate || input.is_empty() {
            return input.to_vec();
        }

        let ratio = self.source_rate as f64 / self.target_rate as f64;
        let output_len = (input.len() as f64 / ratio).floor() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(input.len() - 1);
            let frac = (src_idx - idx0 as f64) as f32;

            let sample = input[idx0] * (1.0 - frac) + input[idx1] * frac;
            output.push(sample.clamp(-1.0, 1.0));
        }

        output
    }

    /// Full pipeline: downmix interleaved input and resample to target rate
    pub fn process(&self, interleaved: &[f32]) -> Vec<f32> {
        let mono = self.downmix_to_mono(interleaved);
        self.resample_mono(&mono)
    }
}

/// Compute Root Mean Square (RMS) energy of a buffer
pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downmix() {
        let converter = AudioConverter::new(48000, 16000, 2);
        let stereo = vec![0.5, 0.5, -0.5, 0.5, 1.0, -1.0];
        let mono = converter.downmix_to_mono(&stereo);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.0).abs() < 1e-6);
        assert!((mono[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_resampling() {
        let converter = AudioConverter::new(32000, 16000, 1);
        let input = vec![0.0, 0.5, 1.0, 0.5];
        let output = converter.resample_mono(&input);
        assert_eq!(output.len(), 2);
    }
}
