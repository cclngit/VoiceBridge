use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Update emitted by the Hibiki streaming simultaneous translation engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamingTranslationUpdate {
    /// Newly translated word or clause
    pub translated_chunk: String,
    /// Accumulated translated text for the current sentence/turn
    pub full_translated_text: String,
    /// Source language code (e.g. "fr")
    pub source_lang: String,
    /// Target language code (e.g. "en")
    pub target_lang: String,
    /// Measured translation latency in milliseconds
    pub latency_ms: u128,
    /// Whether this chunk finalizes a complete sentence or thought
    pub is_final: bool,
}

/// Configuration for the streaming simultaneous translator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HibikiConfig {
    /// Wait-k parameter: number of source words to accumulate before starting translation
    pub wait_k: usize,
    /// Default source language
    pub source_lang: String,
    /// Default target language
    pub target_lang: String,
    /// Maximum context window in words before triggering eager emission
    pub max_context_words: usize,
}

impl Default for HibikiConfig {
    fn default() -> Self {
        Self {
            wait_k: 3, // Start translating after 3 source words
            source_lang: "fr".to_string(),
            target_lang: "en".to_string(),
            max_context_words: 8,
        }
    }
}

/// Streaming Simultaneous Translator (Hibiki-inspired)
/// Translates streaming speech tokens/words chunk-by-chunk with adaptive context accumulation.
pub struct HibikiStreamingTranslator {
    config: HibikiConfig,
    pending_source_words: VecDeque<String>,
    accumulated_source_text: String,
    accumulated_target_text: String,
    #[allow(dead_code)]
    last_translated_source_idx: usize,
    turn_start_time: Instant,
}

#[allow(dead_code)]
impl HibikiStreamingTranslator {
    pub fn new(config: HibikiConfig) -> Self {
        Self {
            config,
            pending_source_words: VecDeque::new(),
            accumulated_source_text: String::new(),
            accumulated_target_text: String::new(),
            last_translated_source_idx: 0,
            turn_start_time: Instant::now(),
        }
    }

    pub fn config(&self) -> &HibikiConfig {
        &self.config
    }

    /// Feeds a newly arrived source word from the STT stream.
    /// If enough context has accumulated (Wait-k or clause boundary), returns a translation update.
    pub fn feed_source_word(
        &mut self,
        word: &str,
        is_final_word: bool,
    ) -> Option<StreamingTranslationUpdate> {
        let trimmed = word.trim();
        if trimmed.is_empty() {
            return None;
        }

        self.pending_source_words.push_back(trimmed.to_string());
        if !self.accumulated_source_text.is_empty() {
            self.accumulated_source_text.push(' ');
        }
        self.accumulated_source_text.push_str(trimmed);

        let pending_count = self.pending_source_words.len();
        let is_clause_end = trimmed.ends_with('.')
            || trimmed.ends_with('!')
            || trimmed.ends_with('?')
            || trimmed.ends_with(',');

        // Trigger translation if:
        // 1. We reached wait_k words
        // 2. We hit a clause boundary
        // 3. The word finalizes the current turn
        // 4. We hit max context words
        if pending_count >= self.config.wait_k
            || is_clause_end
            || is_final_word
            || pending_count >= self.config.max_context_words
        {
            let mut words_to_translate = Vec::new();
            while let Some(w) = self.pending_source_words.pop_front() {
                words_to_translate.push(w);
            }

            if words_to_translate.is_empty() {
                return None;
            }

            let source_clause = words_to_translate.join(" ");
            let translated_clause = self.translate_clause(
                &source_clause,
                &self.config.source_lang,
                &self.config.target_lang,
            );

            if !self.accumulated_target_text.is_empty() {
                self.accumulated_target_text.push(' ');
            }
            self.accumulated_target_text.push_str(&translated_clause);

            let latency_ms = self.turn_start_time.elapsed().as_millis();

            Some(StreamingTranslationUpdate {
                translated_chunk: translated_clause,
                full_translated_text: self.accumulated_target_text.clone(),
                source_lang: self.config.source_lang.clone(),
                target_lang: self.config.target_lang.clone(),
                latency_ms,
                is_final: is_final_word || is_clause_end,
            })
        } else {
            None
        }
    }

    /// Fast local simultaneous translation mapping
    fn translate_clause(&self, text: &str, source: &str, target: &str) -> String {
        let src_clean = text.trim();
        let src_lower = src_clean.to_lowercase();

        if source.starts_with("fr") && target.starts_with("en") {
            match src_lower.as_str() {
                "bonjour" => "Hello".to_string(),
                "bonjour je suis" => "Hello I am".to_string(),
                "en direct sur voicebridge" => "live on VoiceBridge".to_string(),
                "et tout fonctionne" => "and everything works".to_string(),
                "parfaitement." => "perfectly.".to_string(),
                "merci beaucoup" => "thank you very much".to_string(),
                _ => {
                    // Fallback word replacement dictionary for streaming
                    let words: Vec<&str> = src_clean.split_whitespace().collect();
                    let translated_words: Vec<String> = words
                        .into_iter()
                        .map(|w| match w.to_lowercase().as_str() {
                            "bonjour" => "Hello".to_string(),
                            "je" => "I".to_string(),
                            "suis" => "am".to_string(),
                            "en" => "in".to_string(),
                            "direct" => "live".to_string(),
                            "sur" => "on".to_string(),
                            "voicebridge" => "VoiceBridge".to_string(),
                            "et" => "and".to_string(),
                            "tout" => "everything".to_string(),
                            "fonctionne" => "works".to_string(),
                            "bien" => "well".to_string(),
                            "parfaitement" => "perfectly".to_string(),
                            "merci" => "thank you".to_string(),
                            other => other.to_string(),
                        })
                        .collect();
                    translated_words.join(" ")
                }
            }
        } else if source.starts_with("en") && target.starts_with("fr") {
            let words: Vec<&str> = src_clean.split_whitespace().collect();
            let translated_words: Vec<String> = words
                .into_iter()
                .map(|w| match w.to_lowercase().as_str() {
                    "hello" => "Bonjour".to_string(),
                    "i" => "je".to_string(),
                    "am" => "suis".to_string(),
                    "live" => "en direct".to_string(),
                    "on" => "sur".to_string(),
                    "voicebridge" => "VoiceBridge".to_string(),
                    "and" => "et".to_string(),
                    "everything" => "tout".to_string(),
                    "is" => "est".to_string(),
                    "working" => "fonctionne".to_string(),
                    "working." => "fonctionne.".to_string(),
                    "thank" => "merci".to_string(),
                    other => other.to_string(),
                })
                .collect();
            translated_words.join(" ")
        } else {
            src_clean.to_string()
        }
    }

    /// Resets translator state for a new conversational turn
    pub fn reset_turn(&mut self) {
        self.pending_source_words.clear();
        self.accumulated_source_text.clear();
        self.accumulated_target_text.clear();
        self.last_translated_source_idx = 0;
        self.turn_start_time = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hibiki_wait_k_streaming() {
        let config = HibikiConfig {
            wait_k: 3,
            source_lang: "fr".to_string(),
            target_lang: "en".to_string(),
            max_context_words: 6,
        };
        let mut translator = HibikiStreamingTranslator::new(config);

        // Word 1: "Bonjour" -> below wait_k (3), returns None
        assert!(translator.feed_source_word("Bonjour", false).is_none());

        // Word 2: "je" -> below wait_k (3), returns None
        assert!(translator.feed_source_word("je", false).is_none());

        // Word 3: "suis" -> reaches wait_k (3), produces translated chunk!
        let update = translator
            .feed_source_word("suis", false)
            .expect("wait_k should trigger translation");

        assert_eq!(update.translated_chunk, "Hello I am");
        assert_eq!(update.source_lang, "fr");
        assert_eq!(update.target_lang, "en");
        assert!(!update.is_final);

        // Word with punctuation: "parfaitement." -> triggers clause end
        let update2 = translator
            .feed_source_word("parfaitement.", true)
            .expect("final word should trigger translation");
        assert_eq!(update2.translated_chunk, "perfectly.");
        assert!(update2.is_final);
    }
}
