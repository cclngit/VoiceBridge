//! Kyutai streaming speech-to-text (Delayed Streams Modeling) running in-process
//! with Candle, using the official `moshi` crate and the `kyutai/stt-1b-en_fr-candle`
//! weights (Mimi neural codec + 1B decoder, French/English, 0.5 s text delay).
//!
//! The model consumes 80 ms frames of 24 kHz mono audio and emits words with
//! timestamps, plus voice-activity probabilities from its semantic VAD heads that
//! are used to detect end of turn.

use crate::ml::spm::SentencePieceDecoder;
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};

pub const SAMPLE_RATE: u32 = 24_000;
pub const FRAME_SAMPLES: usize = 1920;

#[derive(Debug, serde::Deserialize)]
struct SttSection {
    audio_silence_prefix_seconds: f64,
    audio_delay_seconds: f64,
}

#[derive(Debug, serde::Deserialize)]
struct ModelConfig {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
    #[serde(default)]
    extra_heads_num_heads: Option<usize>,
    #[serde(default)]
    extra_heads_dim: Option<usize>,
    stt_config: SttSection,
}

impl ModelConfig {
    fn lm_config(&self) -> moshi::lm::Config {
        let transformer = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        let extra_heads = match (self.extra_heads_num_heads, self.extra_heads_dim) {
            (Some(num_heads), Some(dim)) if num_heads > 0 => {
                Some(moshi::lm::ExtraHeadsConfig { num_heads, dim })
            }
            _ => None,
        };
        moshi::lm::Config {
            transformer,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads,
        }
    }
}

/// Events produced by the streaming recognizer.
#[derive(Debug, Clone, PartialEq)]
pub enum KyutaiEvent {
    /// A newly recognised word, with its start time (seconds, stream relative).
    Word { text: String, start_s: f64 },
    /// The semantic VAD predicts the speaker finished their turn.
    EndOfTurn,
}

pub struct KyutaiStt {
    state: moshi::asr::State,
    tokenizer: SentencePieceDecoder,
    device: Device,
    audio_delay_s: f64,
    silence_prefix_s: f64,
    pending: Vec<f32>,
    has_vad_heads: bool,
    end_of_turn_reported: bool,
    words_since_turn: usize,
    /// Steps left before reporting a detected end of turn. The VAD heads look at the
    /// current audio while words come out `audio_delay` later, so the turn is closed only
    /// once the delayed words have been emitted.
    pending_end_of_turn: Option<usize>,
    delay_steps: usize,
}

/// Files that make up the model directory.
pub fn required_files(dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
    let config_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let cfg: ModelConfig =
        serde_json::from_str(&raw).map_err(|e| format!("invalid Kyutai config: {e}"))?;
    Ok((
        config_path,
        dir.join("model.safetensors"),
        dir.join(&cfg.mimi_name),
        dir.join(&cfg.tokenizer_name),
    ))
}

impl KyutaiStt {
    /// Load the model from a directory containing `config.json`, `model.safetensors`,
    /// the Mimi codec weights and the SentencePiece tokenizer.
    pub fn load(dir: &Path, device: &Device) -> Result<Self, String> {
        let (config_path, model_path, mimi_path, tokenizer_path) = required_files(dir)?;
        let raw = std::fs::read_to_string(&config_path).map_err(|e| e.to_string())?;
        let cfg: ModelConfig = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        for p in [&model_path, &mimi_path, &tokenizer_path] {
            if !p.is_file() {
                return Err(format!("missing model file {}", p.display()));
            }
        }

        let tokenizer = SentencePieceDecoder::open(&tokenizer_path)?;
        let dtype = device.bf16_default_to_f32();
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, device)
                .map_err(|e| format!("cannot map Kyutai weights: {e}"))?
        };
        let lm_cfg = cfg.lm_config();
        let has_vad_heads = lm_cfg.extra_heads.is_some();
        let lm = moshi::lm::LmModel::new(&lm_cfg, moshi::nn::MaybeQuantizedVarBuilder::Real(vb))
            .map_err(|e| format!("cannot build Kyutai LM: {e}"))?;
        let mimi_path_str = mimi_path.to_str().ok_or("non UTF-8 model path")?;
        let mimi = moshi::mimi::load(mimi_path_str, Some(cfg.n_q), device)
            .map_err(|e| format!("cannot load Mimi codec: {e}"))?;
        let asr_delay_in_tokens = (cfg.stt_config.audio_delay_seconds * 12.5) as usize;
        let state = moshi::asr::State::new(1, asr_delay_in_tokens, 0.0, mimi, lm)
            .map_err(|e| format!("cannot init Kyutai state: {e}"))?;

        let mut engine = Self {
            state,
            tokenizer,
            device: device.clone(),
            audio_delay_s: cfg.stt_config.audio_delay_seconds,
            silence_prefix_s: cfg.stt_config.audio_silence_prefix_seconds,
            pending: Vec::with_capacity(FRAME_SAMPLES * 4),
            has_vad_heads,
            end_of_turn_reported: true,
            words_since_turn: 0,
            pending_end_of_turn: None,
            delay_steps: asr_delay_in_tokens + 2,
        };
        engine.prime()?;
        Ok(engine)
    }

    fn prime(&mut self) -> Result<(), String> {
        if self.silence_prefix_s > 0.0 {
            let n = (self.silence_prefix_s * SAMPLE_RATE as f64) as usize;
            self.push_pcm(&vec![0.0; n])?;
        }
        Ok(())
    }

    /// Text delay of the model (time between audio and the emitted words).
    pub fn audio_delay_s(&self) -> f64 {
        self.audio_delay_s
    }

    /// Feed 24 kHz mono samples (any length) and collect recognised events.
    pub fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<KyutaiEvent>, String> {
        self.pending.extend_from_slice(pcm);
        let mut events = Vec::new();
        let mut offset = 0;
        while self.pending.len() - offset >= FRAME_SAMPLES {
            let frame = &self.pending[offset..offset + FRAME_SAMPLES];
            offset += FRAME_SAMPLES;
            let tensor = Tensor::new(frame, &self.device)
                .and_then(|t| t.reshape((1, 1, FRAME_SAMPLES)))
                .map_err(|e| e.to_string())?;
            let msgs = self
                .state
                .step_pcm(tensor, None, &().into(), |_, _, _| ())
                .map_err(|e| format!("Kyutai inference error: {e}"))?;
            for msg in msgs {
                match msg {
                    moshi::asr::AsrMsg::Word { tokens, start_time, .. } => {
                        let text = self.tokenizer.decode(&tokens);
                        let text = text.trim();
                        if !text.is_empty() {
                            self.end_of_turn_reported = false;
                            self.words_since_turn += 1;
                            events.push(KyutaiEvent::Word {
                                text: text.to_string(),
                                start_s: start_time,
                            });
                        }
                    }
                    moshi::asr::AsrMsg::Step { prs, .. } => {
                        // prs[h][0] = probability of no voice activity within horizon h
                        // (0.5 s, 1 s, 2 s, 3 s for stt-1b-en_fr). Horizon 2 s marks end of turn.
                        if self.has_vad_heads
                            && !self.end_of_turn_reported
                            && self.pending_end_of_turn.is_none()
                            && self.words_since_turn > 0
                            && prs.get(2).and_then(|p| p.first()).is_some_and(|&p| p > 0.5)
                        {
                            self.pending_end_of_turn = Some(self.delay_steps);
                        }
                        if let Some(left) = self.pending_end_of_turn {
                            if left == 0 {
                                self.pending_end_of_turn = None;
                                self.end_of_turn_reported = true;
                                self.words_since_turn = 0;
                                events.push(KyutaiEvent::EndOfTurn);
                            } else {
                                self.pending_end_of_turn = Some(left - 1);
                            }
                        }
                    }
                    moshi::asr::AsrMsg::EndWord { .. } => {}
                }
            }
        }
        self.pending.drain(..offset);
        Ok(events)
    }

    /// Feed enough silence to flush the words still inside the model's delay window.
    pub fn flush(&mut self) -> Result<Vec<KyutaiEvent>, String> {
        let n = ((self.audio_delay_s + 0.5) * SAMPLE_RATE as f64) as usize;
        let mut events = self.push_pcm(&vec![0.0; n])?;
        if self.words_since_turn > 0 && !events.last().is_some_and(|e| *e == KyutaiEvent::EndOfTurn) {
            self.words_since_turn = 0;
            self.end_of_turn_reported = true;
            events.push(KyutaiEvent::EndOfTurn);
        }
        self.pending_end_of_turn = None;
        Ok(events)
    }

    /// Reset the streaming state (KV cache, codec state) for a fresh stream.
    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset().map_err(|e| e.to_string())?;
        self.pending.clear();
        self.end_of_turn_reported = true;
        self.words_since_turn = 0;
        self.pending_end_of_turn = None;
        self.prime()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-model test — needs `models/kyutai-stt-1b-en_fr-candle` and a French WAV file:
    /// `VB_TEST_WAV_FR=/path/fr.wav cargo test --release -- --ignored kyutai`
    #[test]
    #[ignore]
    fn kyutai_transcribes_french_speech() {
        let wav = std::env::var("VB_TEST_WAV_FR").expect("set VB_TEST_WAV_FR");
        let (pcm, rate) = crate::audio::wav::read_wav_mono(Path::new(&wav)).unwrap();
        let pcm24 = crate::audio::resampler::resample_buffer(&pcm, rate, SAMPLE_RATE);
        let device = crate::ml::resolve_device(crate::ml::DeviceSlot::Streaming, std::env::var("VB_TEST_CPU").is_err(), 0);
        let started = std::time::Instant::now();
        let mut stt = KyutaiStt::load(Path::new("models/kyutai-stt-1b-en_fr-candle"), &device).unwrap();
        println!("loaded on {} in {:?}", crate::ml::device_label(&device), started.elapsed());
        let started = std::time::Instant::now();
        let mut words = Vec::new();
        let mut turns = 0;
        let mut turn_texts: Vec<String> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut handle = |ev: KyutaiEvent, words: &mut Vec<String>, turns: &mut usize| match ev {
            KyutaiEvent::Word { text, .. } => {
                words.push(text.clone());
                current.push(text);
            }
            KyutaiEvent::EndOfTurn => {
                *turns += 1;
                turn_texts.push(current.join(" "));
                current.clear();
            }
        };
        // Feed in irregular blocks like the live pipeline does, with a 3 s pause between
        // two repetitions (a real end of turn).
        let mut input = pcm24.clone();
        input.extend(std::iter::repeat_n(0.0, 3 * SAMPLE_RATE as usize));
        input.extend_from_slice(&pcm24);
        for block in input.chunks(1000) {
            for ev in stt.push_pcm(block).unwrap() {
                handle(ev, &mut words, &mut turns);
            }
        }
        for ev in stt.flush().unwrap() {
            handle(ev, &mut words, &mut turns);
        }
        println!("turns: {turn_texts:#?}");
        let audio_s = input.len() as f32 / SAMPLE_RATE as f32;
        let elapsed = started.elapsed().as_secs_f32();
        let text = words.join(" ");
        println!("transcript: {text}\nend-of-turns: {turns}\nRTF: {:.2} ({elapsed:.1}s for {audio_s:.1}s)", elapsed / audio_s);
        let lower = text.to_lowercase();
        assert!(lower.contains("bonjour"), "{text}");
        assert!(lower.contains("traduction"), "{text}");
        assert!(turns >= 2);
        // End of turn must follow real pauses, never cut a phrase between two words
        // (the semantic VAD is ahead of the delayed text stream).
        for t in &turn_texts {
            assert!(t.trim_end().ends_with([',', '.', '?', '!']), "turn cut mid-phrase: {t:?}");
        }
    }

    /// Regression test: Kyutai (streaming slot) and MADLAD (translation slot) running at
    /// the same time on the GPU must both produce correct output.
    /// `VB_TEST_WAV_FR=/path/fr.wav cargo test --release -- --ignored concurrently`
    #[test]
    #[ignore]
    fn kyutai_and_madlad_run_concurrently() {
        let wav = std::env::var("VB_TEST_WAV_FR").expect("set VB_TEST_WAV_FR");
        let (pcm, rate) = crate::audio::wav::read_wav_mono(Path::new(&wav)).unwrap();
        let pcm24 = crate::audio::resampler::resample_buffer(&pcm, rate, SAMPLE_RATE);
        let mt = std::thread::spawn(|| {
            let dev = crate::ml::resolve_device(crate::ml::DeviceSlot::Translation, true, 0);
            let mut m = crate::translate::madlad_engine::MadladTranslator::load(Path::new("models/madlad400-3b-mt"), &dev).unwrap();
            (0..6)
                .map(|_| m.translate("Le chat dort sur le canapé. Il fait beau aujourd'hui.", "en").unwrap())
                .collect::<Vec<_>>()
        });
        let dev = crate::ml::resolve_device(crate::ml::DeviceSlot::Streaming, true, 0);
        let mut stt = KyutaiStt::load(Path::new("models/kyutai-stt-1b-en_fr-candle"), &dev).unwrap();
        let mut words = Vec::new();
        for _ in 0..2 {
            for block in pcm24.chunks(960) {
                for ev in stt.push_pcm(block).unwrap() {
                    if let KyutaiEvent::Word { text, .. } = ev {
                        words.push(text);
                    }
                }
            }
        }
        for ev in stt.flush().unwrap() {
            if let KyutaiEvent::Word { text, .. } = ev {
                words.push(text);
            }
        }
        let translations = mt.join().unwrap();
        let text = words.join(" ").to_lowercase();
        println!("stt: {text}\nmt: {translations:?}");
        assert!(text.matches("bonjour").count() >= 2, "{text}");
        assert!(text.contains("importante"), "{text}");
        for t in &translations {
            let l = t.to_lowercase();
            assert!(l.contains("cat") && (l.contains("sofa") || l.contains("couch")), "{t}");
        }
    }
}
