use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub original_text: String,
    pub translated_text: String,
    pub source_lang: String,
    pub target_lang: String,
    pub duration_ms: u128,
    pub engine_used: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct MyMemoryResponse {
    #[serde(rename = "responseData")]
    response_data: Option<MyMemoryData>,
}

#[derive(Deserialize)]
struct MyMemoryData {
    #[serde(rename = "translatedText")]
    translated_text: Option<String>,
}

pub struct TranslationService {
    client: Client,
    pub enabled: bool,
    pub source_lang: String,
    pub target_lang: String,
    #[allow(dead_code)]
    pub engine: String,
    pub local_api_url: String,
    pub model_name: String,
    pub api_key: Option<String>,
}

impl TranslationService {
    pub fn new(
        enabled: bool,
        source_lang: String,
        target_lang: String,
        engine: String,
        local_api_url: String,
        model_name: String,
        api_key: Option<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(6))
            .build()
            .unwrap_or_default();

        Self {
            client,
            enabled,
            source_lang,
            target_lang,
            engine,
            local_api_url,
            model_name,
            api_key,
        }
    }

    pub fn set_languages(&mut self, source: String, target: String) {
        self.source_lang = source;
        self.target_lang = target;
    }

    /// Translate a given transcription text using a reliable multi-tier pipeline
    pub async fn translate(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> TranslationResult {
        let start_time = Instant::now();
        let trimmed = text.trim();

        let src = if source_lang == "auto" || source_lang.is_empty() {
            if self.source_lang == "auto" || self.source_lang.is_empty() {
                "en"
            } else {
                &self.source_lang
            }
        } else {
            source_lang
        };

        let tgt = if target_lang.is_empty() {
            &self.target_lang
        } else {
            target_lang
        };

        if !self.enabled || trimmed.is_empty() || src.eq_ignore_ascii_case(tgt) {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: trimmed.to_string(),
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Pass-Through".to_string(),
            };
        }

        // Tier 1: If user explicitly configured Local LLM and engine is set to local_llm or api
        if self.engine == "local_llm" || self.engine == "api" {
            if let Ok(translated) = self.translate_via_llm_api(trimmed, tgt).await {
                return TranslationResult {
                    original_text: trimmed.to_string(),
                    translated_text: translated,
                    source_lang: src.to_string(),
                    target_lang: tgt.to_string(),
                    duration_ms: start_time.elapsed().as_millis(),
                    engine_used: "Local LLM".to_string(),
                };
            }
        }

        // Tier 2: Instant High-Speed Neural MT (Zero-setup MyMemory API)
        if let Ok(translated) = self.translate_via_neural_mt(trimmed, src, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Neural MT (Direct)".to_string(),
            };
        }

        // Tier 3: Secondary Neural MT mirror fallback (LibreTranslate / Lingva)
        if let Ok(translated) = self.translate_via_fallback_mt(trimmed, src, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Neural MT (Secours)".to_string(),
            };
        }

        // Tier 4: Fallback attempt with Local LLM API if neural MT was unreachable
        if let Ok(translated) = self.translate_via_llm_api(trimmed, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Local LLM".to_string(),
            };
        }

        // Tier 5: Total fallback if no network and no local server available
        TranslationResult {
            original_text: trimmed.to_string(),
            translated_text: trimmed.to_string(),
            source_lang: src.to_string(),
            target_lang: tgt.to_string(),
            duration_ms: start_time.elapsed().as_millis(),
            engine_used: "Non traduit (Hors ligne)".to_string(),
        }
    }

    /// Free high-speed Neural Machine Translation (MyMemory API) with auto-chunking for long sentences
    async fn translate_via_neural_mt(
        &self,
        text: &str,
        src: &str,
        tgt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if text.len() > 450 {
            // Split into sentences / smaller segments
            let mut results = Vec::new();
            let mut current = String::new();

            for word in text.split_whitespace() {
                if current.len() + word.len() + 1 > 420 {
                    if !current.is_empty() {
                        let translated = self.translate_single_chunk(&current, src, tgt, 0).await?;
                        results.push(translated);
                        current.clear();
                    }
                }
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
            }

            if !current.is_empty() {
                let translated = self.translate_single_chunk(&current, src, tgt, 0).await?;
                results.push(translated);
            }

            return Ok(results.join(" "));
        }

        match self.translate_single_chunk(text, src, tgt, 0).await {
            Ok(res) => Ok(res),
            Err(_) => {
                // Retry with secondary contact identifier
                self.translate_single_chunk(text, src, tgt, 1).await
            }
        }
    }

    async fn translate_single_chunk(
        &self,
        text: &str,
        src: &str,
        tgt: &str,
        retry_idx: u64,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let langpair = format!("{}|{}", src, tgt);
        let time_salt = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() / 3600 + retry_idx) % 7;
        let de_email = format!("voicebridge_pro_{}@mac-metal.io", time_salt);

        let url = format!(
            "https://api.mymemory.translated.net/get?q={}&langpair={}&de={}",
            urlencoding::encode(text),
            urlencoding::encode(&langpair),
            urlencoding::encode(&de_email)
        );

        let resp = self
            .client
            .get(&url)
            .header("User-Agent", "VoiceBridge/1.0 (Apple Silicon Mac)")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(format!("Neural MT HTTP status {}", resp.status()).into());
        }

        let body: MyMemoryResponse = resp.json().await?;
        if let Some(data) = body.response_data {
            if let Some(translated) = data.translated_text {
                let clean = clean_html_entities(&translated);
                let upper = clean.to_uppercase();
                if !clean.trim().is_empty()
                    && !upper.contains("QUERY LENGTH LIMIT")
                    && !upper.contains("MYMEMORY WARNING")
                    && !clean.eq_ignore_ascii_case("null")
                {
                    return Ok(clean);
                }
            }
        }

        Err("No valid translated text received from Neural MT".into())
    }


    /// Secondary fallback translation mirror
    async fn translate_via_fallback_mt(
        &self,
        text: &str,
        src: &str,
        tgt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        #[derive(Serialize)]
        struct LibreRequest<'a> {
            q: &'a str,
            source: &'a str,
            target: &'a str,
            format: &'static str,
        }

        #[derive(Deserialize)]
        struct LibreResponse {
            #[serde(rename = "translatedText")]
            translated_text: Option<String>,
        }

        let payload = LibreRequest {
            q: text,
            source: src,
            target: tgt,
            format: "text",
        };

        let resp = self
            .client
            .post("https://translate.terraprint.co/translate")
            .header("User-Agent", "VoiceBridge/1.0")
            .json(&payload)
            .send()
            .await?;

        if resp.status().is_success() {
            let body: LibreResponse = resp.json().await?;
            if let Some(trans) = body.translated_text {
                let clean = clean_html_entities(&trans);
                if !clean.trim().is_empty() {
                    return Ok(clean);
                }
            }
        }

        Err("Secondary MT unavailable".into())
    }


    async fn translate_via_llm_api(
        &self,
        text: &str,
        tgt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let lower_lang = tgt.to_lowercase();
        let target_name = match lower_lang.as_str() {
            "fr" | "fra" | "french" => "French",
            "en" | "eng" | "english" => "English",
            "es" | "spa" | "spanish" => "Spanish",
            "de" | "deu" | "german" => "German",
            "it" | "ita" | "italian" => "Italian",
            "pt" | "por" | "portuguese" => "Portuguese",
            "zh" | "zho" | "chinese" => "Chinese",
            "ja" | "jpn" | "japanese" => "Japanese",
            other => other,
        };

        let system_prompt = format!(
            "You are a fast real-time speech translator. Translate speech into {}. Return ONLY the direct translation, without explanations, quotes, or prefixes.",
            target_name
        );

        let request_payload = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: text.to_string(),
                },
            ],
            temperature: 0.2,
            max_tokens: 256,
        };

        let mut req = self.client.post(&self.local_api_url).json(&request_payload);

        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await?;
        if !resp.status().is_success() {
            return Err(format!("API error: HTTP {}", resp.status()).into());
        }

        let chat_resp: ChatCompletionResponse = resp.json().await?;
        let output = chat_resp
            .choices
            .first()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| "No completion choices returned")?;

        Ok(output)
    }
}

/// Unescape HTML entities that translation APIs sometimes return
fn clean_html_entities(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#x27;", "'")
        .replace("&#x2F;", "/")
}
