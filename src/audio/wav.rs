//! Minimal WAV (16-bit PCM / 32-bit float) reader used by `--input-wav` and tests.

use std::path::Path;

/// Read a WAV file and return mono samples with the file sample rate.
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{} n'est pas un fichier WAV", path.display()));
    }
    let mut pos = 12;
    let (mut format, mut channels, mut rate, mut bits) = (1u16, 1u16, 16_000u32, 16u16);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            format = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]).max(1);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
            if format == 0xFFFE && body.len() >= 26 {
                // WAVE_FORMAT_EXTENSIBLE: the real format is in the sub-format GUID.
                format = u16::from_le_bytes([body[24], body[25]]);
            }
        } else if id == b"data" {
            let samples: Vec<f32> = match (format, bits) {
                (1, 16) => body
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                    .collect(),
                (3, 32) => body
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
                _ => return Err(format!("format WAV non pris en charge (format {format}, {bits} bits)")),
            };
            return Ok((super::resampler::downmix_to_mono(&samples, channels as usize), rate));
        }
        pos += 8 + len + (len & 1);
    }
    Err("aucun bloc audio dans le fichier WAV".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_generated_pcm16_wav() {
        let samples = [0.0f32, 0.5, -0.5, 0.25];
        let bytes = crate::stt::engine::samples_to_wav_bytes(&samples, 16_000);
        let path = std::env::temp_dir().join(format!("vb-wav-{}.wav", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let (pcm, rate) = read_wav_mono(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(rate, 16_000);
        assert_eq!(pcm.len(), 4);
        assert!((pcm[1] - 0.5).abs() < 1e-3);
    }
}
