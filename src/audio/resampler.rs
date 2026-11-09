//! Audio format helpers: down-mixing, band-limited streaming resampling and RMS.
//!
//! The streaming resampler keeps state between calls, so arbitrarily sized input
//! blocks (as delivered by CPAL callbacks) are converted without dropping
//! fractional samples or creating discontinuities at block boundaries, and it
//! applies proper anti-aliasing (FFT based, via `rubato`) when downsampling.

use rubato::{FftFixedIn, Resampler};

/// Number of input frames processed per FFT block.
const CHUNK_FRAMES: usize = 1024;

/// Downmix interleaved multi-channel samples to mono by averaging channels.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let channels = channels.max(1);
    if channels == 1 {
        return interleaved.to_vec();
    }
    let inv = 1.0 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * inv)
        .collect()
}

/// Stateful, band-limited mono resampler for continuous streams.
pub struct StreamingResampler {
    inner: Option<FftFixedIn<f32>>,
    pending: Vec<f32>,
}

impl StreamingResampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        let inner = if source_rate == target_rate || source_rate == 0 || target_rate == 0 {
            None
        } else {
            match FftFixedIn::<f32>::new(
                source_rate as usize,
                target_rate as usize,
                CHUNK_FRAMES,
                2,
                1,
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::error!(
                        "Unable to build resampler {} Hz -> {} Hz ({}); audio passed through unchanged.",
                        source_rate,
                        target_rate,
                        e
                    );
                    None
                }
            }
        };
        Self {
            inner,
            pending: Vec::with_capacity(CHUNK_FRAMES * 2),
        }
    }

    /// Push mono samples and return every resampled sample that is ready.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let Some(resampler) = self.inner.as_mut() else {
            return input.to_vec();
        };
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();
        loop {
            let needed = resampler.input_frames_next();
            if self.pending.len() < needed {
                break;
            }
            match resampler.process(&[&self.pending[..needed]], None) {
                Ok(mut chunks) => {
                    if let Some(chunk) = chunks.pop() {
                        out.extend(chunk);
                    }
                }
                Err(e) => {
                    tracing::error!("Resampling error: {}", e);
                }
            }
            self.pending.drain(..needed);
        }
        out
    }

    /// Flush the remaining buffered samples (zero padded) at the end of a stream.
    #[cfg(test)]
    pub fn flush(&mut self) -> Vec<f32> {
        let Some(resampler) = self.inner.as_mut() else {
            return std::mem::take(&mut self.pending);
        };
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            if let Ok(mut chunks) = resampler.process_partial(Some(&[&self.pending[..]]), None)
                && let Some(chunk) = chunks.pop()
            {
                out.extend(chunk);
            }
            self.pending.clear();
        }
        out
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        if let Some(r) = self.inner.as_mut() {
            r.reset();
        }
    }
}

/// One-shot resampling of a complete mono buffer (keeps the expected output length).
#[cfg(test)]
pub fn resample_buffer(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || input.is_empty() {
        return input.to_vec();
    }
    let expected = (input.len() as u64 * target_rate as u64 / source_rate as u64) as usize;
    let mut r = StreamingResampler::new(source_rate, target_rate);
    let delay = r
        .inner
        .as_ref()
        .map(|i| i.output_delay())
        .unwrap_or(0);
    let mut out = r.process(input);
    // Feed enough silence to push the resampler's internal delay out.
    let tail = vec![0.0f32; CHUNK_FRAMES * 2];
    out.extend(r.process(&tail));
    out.extend(r.flush());
    let start = delay.min(out.len());
    let mut out: Vec<f32> = out.split_off(start);
    out.resize(expected, 0.0);
    out
}

/// Compute Root Mean Square (RMS) energy of a buffer.
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

    fn sine(freq: f32, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 / rate as f32 * freq * std::f32::consts::TAU).sin() * 0.5)
            .collect()
    }

    #[test]
    fn downmix_averages_channels() {
        let stereo = vec![0.5, 0.5, -0.5, 0.5, 1.0, -1.0];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.5, 0.0, 0.0]);
    }

    #[test]
    fn streaming_resampler_keeps_sample_count_across_odd_blocks() {
        // 44.1 kHz -> 16 kHz with irregular CPAL-like block sizes.
        let input = sine(440.0, 44_100, 44_100 * 2);
        let mut r = StreamingResampler::new(44_100, 16_000);
        let mut out = Vec::new();
        for block in input.chunks(441) {
            out.extend(r.process(block));
        }
        out.extend(r.flush());
        let expected = 32_000i64;
        assert!(
            (out.len() as i64 - expected).abs() < 1100,
            "got {} samples, expected about {}",
            out.len(),
            expected
        );
    }

    #[test]
    fn downsampling_attenuates_aliasing_content() {
        // A 12 kHz tone cannot be represented at 16 kHz (Nyquist 8 kHz): it must be filtered out.
        let input = sine(12_000.0, 48_000, 48_000);
        let out = resample_buffer(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        let rms = calculate_rms(&out[2000..14_000]);
        assert!(rms < 0.02, "aliased energy too high: {rms}");
    }

    #[test]
    fn passband_tone_is_preserved() {
        let input = sine(440.0, 16_000, 16_000);
        let out = resample_buffer(&input, 16_000, 24_000);
        assert_eq!(out.len(), 24_000);
        let rms = calculate_rms(&out[2000..22_000]);
        assert!((rms - 0.3535).abs() < 0.03, "rms {rms}");
    }
}
