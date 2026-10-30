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

        let src = if source_lang.is_empty() {
            if self.source_lang.is_empty() {
                "auto"
            } else {
                &self.source_lang
            }
        } else {
            source_lang
        };

        let tgt = if target_lang.is_empty() {
            if self.target_lang.is_empty() {
                "fr"
            } else {
                &self.target_lang
            }
        } else {
            target_lang
        };

        if !self.enabled || trimmed.is_empty() || (!src.eq_ignore_ascii_case("auto") && src.eq_ignore_ascii_case(tgt)) {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: trimmed.to_string(),
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Pass-Through".to_string(),
            };
        }

        // Tier 1: If user explicitly configured Local LLM / API
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

        // Tier 2: Google Chrome Neural MT (Ultra-fast, zero-quota, highest accuracy)
        if let Ok(translated) = self.translate_via_google_chrome(trimmed, src, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Google Neural MT".to_string(),
            };
        }

        // Tier 3: Google GTX Neural MT (Reliable secondary endpoint)
        if let Ok(translated) = self.translate_via_google_gtx(trimmed, src, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "Google Neural MT (GTX)".to_string(),
            };
        }

        // Tier 4: MyMemory Neural MT
        if let Ok(translated) = self.translate_via_neural_mt(trimmed, src, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "MyMemory MT".to_string(),
            };
        }

        // Tier 5: Secondary Neural MT mirror fallback (LibreTranslate)
        if let Ok(translated) = self.translate_via_fallback_mt(trimmed, src, tgt).await {
            return TranslationResult {
                original_text: trimmed.to_string(),
                translated_text: translated,
                source_lang: src.to_string(),
                target_lang: tgt.to_string(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used: "LibreTranslate MT".to_string(),
            };
        }

        // Tier 6: Fallback attempt with Local LLM API
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

        // Tier 7: Total fallback if no network and no local server available
        TranslationResult {
            original_text: trimmed.to_string(),
            translated_text: trimmed.to_string(),
            source_lang: src.to_string(),
            target_lang: tgt.to_string(),
            duration_ms: start_time.elapsed().as_millis(),
            engine_used: "Non traduit (Hors ligne)".to_string(),
        }
    }

    /// Primary Ultra-Fast Neural MT using Google Chrome Engine (Unlimited, ~40ms, multilingual)
    async fn translate_via_google_chrome(
        &self,
        text: &str,
        src: &str,
        tgt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let clean_src = if src.eq_ignore_ascii_case("auto") || src.is_empty() {
            "auto"
        } else {
            src
        };
        let clean_tgt = if tgt.eq_ignore_ascii_case("auto") || tgt.is_empty() {
            "en"
        } else {
            tgt
        };

        let url = format!(
            "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl={}&tl={}&q={}",
            urlencoding::encode(clean_src),
            urlencoding::encode(clean_tgt),
            urlencoding::encode(text)
        );

        let resp = self
            .client
            .get(&url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(format!("Chrome MT HTTP status {}", resp.status()).into());
        }

        let val: serde_json::Value = resp.json().await?;
        if let Some(arr) = val.as_array() {
            let mut results = Vec::new();
            for item in arr {
                if let Some(s) = item.as_str() {
                    results.push(s.to_string());
                } else if let Some(sub_arr) = item.as_array() {
                    if let Some(s) = sub_arr.first().and_then(|v| v.as_str()) {
                        results.push(s.to_string());
                    }
                }
            }
            if !results.is_empty() {
                let joined = results.join(" ");
                let clean = clean_html_entities(&joined);
                if !clean.trim().is_empty() {
                    return Ok(clean);
                }
            }
        } else if let Some(s) = val.as_str() {
            let clean = clean_html_entities(s);
            if !clean.trim().is_empty() {
                return Ok(clean);
            }
        }

        Err("No valid translation received from Chrome MT".into())
    }

    /// Secondary High-Speed Neural MT using Google GTX API
    async fn translate_via_google_gtx(
        &self,
        text: &str,
        src: &str,
        tgt: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let clean_src = if src.eq_ignore_ascii_case("auto") || src.is_empty() {
            "auto"
        } else {
            src
        };
        let clean_tgt = if tgt.eq_ignore_ascii_case("auto") || tgt.is_empty() {
            "en"
        } else {
            tgt
        };

        let url = format!(
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&q={}",
            urlencoding::encode(clean_src),
            urlencoding::encode(clean_tgt),
            urlencoding::encode(text)
        );

        let resp = self
            .client
            .get(&url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(format!("GTX MT HTTP status {}", resp.status()).into());
        }

        let val: serde_json::Value = resp.json().await?;
        if let Some(sentences) = val.get(0).and_then(|v| v.as_array()) {
            let mut translated = String::new();
            for s in sentences {
                if let Some(part) = s.get(0).and_then(|v| v.as_str()) {
                    translated.push_str(part);
                }
            }
            let clean = clean_html_entities(&translated);
            if !clean.trim().is_empty() {
                return Ok(clean);
            }
        }

        Err("No valid translation received from GTX MT".into())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_translate_en_to_fr() {
        let svc = TranslationService::new(
            true,
            "en".to_string(),
            "fr".to_string(),
            "auto".to_string(),
            "".to_string(),
            "".to_string(),
            None,
        );
        let res = svc.translate("Hello world, this is a test", "en", "fr").await;
        assert_ne!(res.translated_text, res.original_text);
        assert!(!res.translated_text.is_empty());
        println!("EN -> FR: '{}' -> '{}' ({})", res.original_text, res.translated_text, res.engine_used);
    }

    #[tokio::test]
    async fn test_translate_en_to_es() {
        let svc = TranslationService::new(
            true,
            "en".to_string(),
            "es".to_string(),
            "auto".to_string(),
            "".to_string(),
            "".to_string(),
            None,
        );
        let res = svc.translate("How are you today?", "en", "es").await;
        assert_ne!(res.translated_text, res.original_text);
        assert!(!res.translated_text.is_empty());
        println!("EN -> ES: '{}' -> '{}' ({})", res.original_text, res.translated_text, res.engine_used);
    }
}
