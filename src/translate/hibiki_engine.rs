//! Kyutai Hibiki — real-time simultaneous speech translation (French → English),
//! running in-process with Candle through the official `moshi` crate and the
//! `kyutai/hibiki-1b-rs-bf16` / `kyutai/hibiki-2b-rs-bf16` weights.
//!
//! Hibiki listens to 24 kHz French speech in 80 ms frames and produces the English
//! translation as a text stream (alongside translated speech tokens, which are
//! generated because the model needs them but are not decoded to audio here).

use crate::ml::spm::SentencePieceDecoder;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_SAMPLES: usize = 1920;

/// Steps allocated per session (2500 steps = 200 s); the session is recycled before.
const MAX_SESSION_STEPS: usize = 2500;
/// Recycle the session once this many steps have elapsed and the speaker is silent.
const RECYCLE_AFTER_STEPS: usize = 1500;

#[derive(Debug, Clone, serde::Deserialize)]
struct HibikiFileConfig {
    mimi_name: String,
    moshi_name: String,
    tokenizer_name: String,
    model: moshi::lm::Config,
}

pub fn required_files(dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
    let config_path = dir.join("config.toml");
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let cfg: HibikiFileConfig =
        toml::from_str(&raw).map_err(|e| format!("invalid Hibiki config: {e}"))?;
    Ok((
        config_path,
        dir.join(&cfg.moshi_name),
        dir.join(&cfg.mimi_name),
        dir.join(&cfg.tokenizer_name),
    ))
}

pub struct HibikiTranslator {
    base_model: moshi::lm::LmModel,
    mimi: moshi::mimi::Mimi,
    tokenizer: SentencePieceDecoder,
    device: Device,
    gen_config: moshi::lm_generate_multistream::Config,
    conditions: Option<moshi::conditioner::Condition>,
    state: moshi::lm_generate_multistream::State,
    prev_text_token: u32,
    pending: Vec<f32>,
    session_steps: usize,
    seed: u64,
}

impl HibikiTranslator {
    pub fn load(dir: &Path, device: &Device) -> Result<Self, String> {
        let (config_path, lm_path, mimi_path, tokenizer_path) = required_files(dir)?;
        let raw = std::fs::read_to_string(&config_path).map_err(|e| e.to_string())?;
        let cfg: HibikiFileConfig = toml::from_str(&raw).map_err(|e| e.to_string())?;
        for p in [&lm_path, &mimi_path, &tokenizer_path] {
            if !p.is_file() {
                return Err(format!("missing model file {}", p.display()));
            }
        }

        let dtype = device.bf16_default_to_f32();
        let base_model = moshi::lm::load_lm_model(cfg.model.clone(), &lm_path, dtype, device)
            .map_err(|e| format!("cannot load Hibiki LM: {e}"))?;
        let generated_audio_codebooks = cfg.model.depformer.as_ref().map_or(8, |d| d.num_slices);
        let mimi_path_str = mimi_path.to_str().ok_or("non UTF-8 model path")?;
        let mimi = moshi::mimi::load(mimi_path_str, Some(generated_audio_codebooks), device)
            .map_err(|e| format!("cannot load Mimi codec: {e}"))?;
        let tokenizer = SentencePieceDecoder::open(&tokenizer_path)?;

        let conditions = match base_model.condition_provider() {
            None => None,
            Some(cp) => Some(
                cp.condition_lut("description", "very_good")
                    .map_err(|e| format!("cannot build Hibiki condition: {e}"))?,
            ),
        };
        let gen_config = moshi::lm_generate_multistream::Config {
            acoustic_delay: 2,
            audio_vocab_size: cfg.model.audio_vocab_size,
            generated_audio_codebooks,
            input_audio_codebooks: cfg.model.audio_codebooks - generated_audio_codebooks,
            text_start_token: cfg.model.text_out_vocab_size as u32,
            text_eop_token: 0,
            text_pad_token: 3,
        };
        let seed = 299_792_458;
        let state = Self::new_state(&base_model, &gen_config, seed);
        let prev_text_token = gen_config.text_start_token;
        Ok(Self {
            base_model,
            mimi,
            tokenizer,
            device: device.clone(),
            gen_config,
            conditions,
            state,
            prev_text_token,
            pending: Vec::with_capacity(FRAME_SAMPLES * 4),
            session_steps: 0,
            seed,
        })
    }

    fn new_state(
        model: &moshi::lm::LmModel,
        config: &moshi::lm_generate_multistream::Config,
        seed: u64,
    ) -> moshi::lm_generate_multistream::State {
        let mut model = model.clone();
        model.reset_state();
        // Sampling parameters from the reference hibiki-rs implementation.
        let audio_lp = LogitsProcessor::from_sampling(
            seed,
            Sampling::TopK { k: 250, temperature: 0.8 },
        );
        let text_lp = LogitsProcessor::from_sampling(
            seed,
            Sampling::TopK { k: 25, temperature: 0.8 },
        );
        moshi::lm_generate_multistream::State::new(
            model,
            MAX_SESSION_STEPS + 20,
            audio_lp,
            text_lp,
            None,
            None,
            None,
            config.clone(),
        )
    }

    /// Start a fresh translation session (clears the model and codec states).
    pub fn reset(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.state = Self::new_state(&self.base_model, &self.gen_config, self.seed);
        self.mimi.reset_state();
        self.prev_text_token = self.gen_config.text_start_token;
        self.pending.clear();
        self.session_steps = 0;
    }

    pub fn session_steps(&self) -> usize {
        self.session_steps
    }

    /// Whether the session should be recycled now (long session and speaker silent).
    pub fn should_recycle(&self, silent: bool) -> bool {
        (silent && self.session_steps >= RECYCLE_AFTER_STEPS)
            || self.session_steps >= MAX_SESSION_STEPS
    }

    /// Feed 24 kHz mono French speech and return newly produced English text pieces.
    pub fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<String>, String> {
        self.pending.extend_from_slice(pcm);
        let mut out = Vec::new();
        let mut offset = 0;
        while self.pending.len() - offset >= FRAME_SAMPLES {
            if self.session_steps >= MAX_SESSION_STEPS {
                self.pending.drain(..offset);
                offset = 0;
                tracing::info!("Hibiki: session length limit reached, starting a new session.");
                let keep = std::mem::take(&mut self.pending);
                self.reset();
                self.pending = keep;
                continue;
            }
            let frame = &self.pending[offset..offset + FRAME_SAMPLES];
            offset += FRAME_SAMPLES;
            let pcm = Tensor::new(frame, &self.device)
                .and_then(|t| t.reshape((1, 1, FRAME_SAMPLES)))
                .map_err(|e| e.to_string())?;
            let codes = self
                .mimi
                .encode_step(&pcm.into(), &().into())
                .map_err(|e| format!("Mimi encoding error: {e}"))?;
            let Some(codes) = codes.as_option() else {
                continue;
            };
            let (_b, _q, steps) = codes.dims3().map_err(|e| e.to_string())?;
            for step in 0..steps {
                let step_codes = codes
                    .i((0, .., step))
                    .and_then(|t| t.to_vec1::<u32>())
                    .map_err(|e| e.to_string())?;
                let text_token = self
                    .state
                    .step_(
                        Some(self.prev_text_token),
                        &step_codes,
                        None,
                        None,
                        self.conditions.as_ref(),
                    )
                    .map_err(|e| format!("Hibiki inference error: {e}"))?;
                self.session_steps += 1;
                if text_token != self.gen_config.text_eop_token
                    && text_token != self.gen_config.text_pad_token
                    && text_token != self.gen_config.text_start_token
                {
                    let piece = self.tokenizer.decode_piece(text_token);
                    if !piece.is_empty() {
                        out.push(piece);
                    }
                }
                self.prev_text_token = text_token;
            }
        }
        self.pending.drain(..offset);
        Ok(out)
    }

    /// Feed silence so the translation of the last words (Hibiki lags a few seconds
    /// behind the speaker) is produced.
    pub fn flush(&mut self, seconds: f32) -> Result<Vec<String>, String> {
        let n = (seconds.max(0.0) * SAMPLE_RATE as f32) as usize;
        self.push_pcm(&vec![0.0; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-model test — needs `models/hibiki-1b-rs-bf16` and a French WAV file:
    /// `VB_TEST_WAV_FR=/path/fr.wav cargo test --release -- --ignored hibiki`
    #[test]
    #[ignore]
    fn hibiki_translates_french_speech_to_english() {
        let wav = std::env::var("VB_TEST_WAV_FR").expect("set VB_TEST_WAV_FR");
        let (pcm, rate) = crate::audio::wav::read_wav_mono(Path::new(&wav)).unwrap();
        let pcm24 = crate::audio::resampler::resample_buffer(&pcm, rate, SAMPLE_RATE);
        let device = crate::ml::resolve_device(crate::ml::DeviceSlot::Streaming, std::env::var("VB_TEST_CPU").is_err(), 0);
        let t0 = std::time::Instant::now();
        let mut hibiki = HibikiTranslator::load(Path::new("models/hibiki-1b-rs-bf16"), &device).unwrap();
        println!("loaded on {} in {:?}", crate::ml::device_label(&device), t0.elapsed());
        let t0 = std::time::Instant::now();
        let mut text = String::new();
        for block in pcm24.chunks(1000) {
            for piece in hibiki.push_pcm(block).unwrap() {
                text.push_str(&piece);
            }
        }
        for piece in hibiki.flush(4.0).unwrap() {
            text.push_str(&piece);
        }
        let audio_s = pcm24.len() as f32 / SAMPLE_RATE as f32 + 4.0;
        let el = t0.elapsed().as_secs_f32();
        println!("translation: {}\nRTF: {:.2} ({el:.1}s for {audio_s:.1}s)", text.trim(), el / audio_s);
        let lower = text.to_lowercase();
        assert!(lower.contains("hello") || lower.contains("everyone") || lower.contains("today"), "{text}");
        assert!(lower.contains("translation") || lower.contains("sound") || lower.contains("quality"), "{text}");

        // A new session after reset must work as well.
        hibiki.reset();
        assert_eq!(hibiki.session_steps(), 0);
        let _ = hibiki.push_pcm(&pcm24[..SAMPLE_RATE as usize]).unwrap();
        assert!(hibiki.session_steps() > 0);
    }
}
