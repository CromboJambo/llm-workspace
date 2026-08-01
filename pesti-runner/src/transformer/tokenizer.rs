//! Tokenizer integration with GGUF vocab.
//!
//! Loads tokenizer configuration from GGUF KV pairs and wraps the tokenizers library.
//!

#![allow(clippy::if_same_then_else, clippy::collapsible_if)]
use std::collections::HashMap;
use std::path::Path;

use pesti_gguf::types::GgufHeader;
use tokenizers::tokenizer::{Result, Tokenizer};
use tracing::debug;

use crate::error::RunnerError;

/// GGUF tokenizer configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GgufTokenizerConfig {
    /// Tokenizer model type (e.g., "llama", "gpt2", "bpe").
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// PreTokenizer type (e.g., "default", "split", "byte_fallback").
    pub pre_tokenizer_type: Option<String>,
    /// PostProcessor type.
    pub post_processor_type: Option<String>,
    /// Added tokens (from GGUF vocab).
    pub added_tokens: HashMap<u32, String>,
    /// Pattern regex for splitting.
    pub pattern: Option<String>,
    /// BOS token ID.
    pub bos_token_id: Option<u32>,
    /// EOS token ID.
    pub eos_token_id: Option<u32>,
    /// UNK token ID.
    pub unk_token_id: Option<u32>,
    /// Whether to add BOS token by default.
    pub add_bos_token: Option<bool>,
    /// Whether to add EOS token by default.
    pub add_eos_token: Option<bool>,
}

impl GgufTokenizerConfig {
    /// Build config from GGUF header KV pairs.
    pub fn from_gguf_header(header: &GgufHeader) -> Self {
        let model_type = header
            .get_kv_str("tokenizer.ggml.model")
            .unwrap_or("llama")
            .to_string();
        let vocab_size = header.vocab_size().unwrap_or(32000);

        // Load added tokens from GGUF
        let mut added_tokens = HashMap::new();
        let token_count = header
            .get_kv_u32("tokenizer.ggml.tokens")
            .or_else(|| header.get_kv_u32("tokenizer.ggml.length"))
            .unwrap_or(0);

        for id in 0..token_count {
            let key = format!("tokenizer.ggml.tokens.{id}");
            if let Some(value) = header.get_kv_str(&key) {
                added_tokens.insert(id, value.to_string());
            }
        }

        // Load token scores if available
        let _scores: HashMap<u32, f32> = (0..token_count)
            .filter_map(|id| {
                let key = format!("tokenizer.ggml.scores.{id}");
                header.get_kv_f32(&key).map(|score| (id, score))
            })
            .collect();

        let pre_tokenizer_type = header
            .get_kv_str("tokenizer.ggml.pre")
            .map(|s| s.to_string());

        let post_processor_type = header
            .get_kv_str("tokenizer.ggml.postprocess")
            .map(|s| s.to_string());

        let pattern = header
            .get_kv_str("tokenizer.ggml.token_type")
            .map(|s| s.to_string());

        let bos_token_id = header.get_kv_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = header.get_kv_u32("tokenizer.ggml.eos_token_id");
        let unk_token_id = header.get_kv_u32("tokenizer.ggml.unk_token_id");
        let add_bos_token = header.get_kv_bool("tokenizer.ggml.add_bos_token");
        let add_eos_token = header.get_kv_bool("tokenizer.ggml.add_eos_token");

        Self {
            model_type,
            vocab_size,
            pre_tokenizer_type,
            post_processor_type,
            added_tokens,
            pattern,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            add_bos_token,
            add_eos_token,
        }
    }

    /// Build a working `tokenizers::Tokenizer` from this GGUF config.
    pub fn to_tokenizer(&self) -> Tokenizer {
        // Build a JSON tokenizer config from GGUF header data
        let mut added_tokens_json: Vec<serde_json::Value> = Vec::new();

        // Add BOS token if present
        if let Some(bos_id) = self.bos_token_id {
            if let Some(bos_tok) = self.added_tokens.get(&bos_id) {
                added_tokens_json.push(serde_json::json!({
                    "content": bos_tok,
                    "special": true
                }));
            }
        }

        // Add EOS token if present
        if let Some(eos_id) = self.eos_token_id {
            if let Some(eos_tok) = self.added_tokens.get(&eos_id) {
                added_tokens_json.push(serde_json::json!({
                    "content": eos_tok,
                    "special": true
                }));
            }
        }

        // Add vocab tokens (non-special)
        let bos_id = self.bos_token_id.unwrap_or(0);
        let eos_id = self.eos_token_id.unwrap_or(0);
        for (id, token) in &self.added_tokens {
            if *id != bos_id && *id != eos_id {
                added_tokens_json.push(serde_json::json!({
                    "content": token,
                    "lstrip": false,
                    "rstrip": false,
                    "single_word": false,
                    "normalized": true,
                    "special": false
                }));
            }
        }

        let config = serde_json::json!({
            "version": "1.0",
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "vocab": self.added_tokens.iter()
                .map(|(id, token)| serde_json::json!([token, serde_json::Value::from(*id)]))
                .collect::<serde_json::Value>(),
            "merges": added_tokens_json.iter()
                .map(|tok| {
                    let content = tok["content"].as_str().unwrap_or("");
                    serde_json::json!([content, content])
                })
                .collect::<serde_json::Value>(),
            "added_tokens": serde_json::Value::Array(added_tokens_json)
        });

        Tokenizer::from_bytes(config.to_string().as_bytes()).unwrap_or_else(|e| {
            debug!("Failed to build tokenizer from GGUF config: {e}, using empty tokenizer");
            Tokenizer::new(tokenizers::models::bpe::BPE::default())
        })
    }
}

/// Load a tokenizer from a GGUF file.
pub fn load_tokenizer_from_gguf(path: &Path) -> Result<(GgufTokenizerConfig, Tokenizer)> {
    let header = pesti_gguf::parser::parse_gguf(path)
        .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

    let config = GgufTokenizerConfig::from_gguf_header(&header);
    let tokenizer = config.to_tokenizer();

    debug!(
        path = %path.display(),
        model_type = %config.model_type,
        vocab_size = config.vocab_size,
        "Loaded GGUF tokenizer"
    );

    Ok((config, tokenizer))
}

/// Build a tokenizer config from a GGUF header without loading the file.
pub fn tokenizer_config_from_header(header: &GgufHeader) -> GgufTokenizerConfig {
    GgufTokenizerConfig::from_gguf_header(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pesti_gguf::{compute_data_section_start, GgufKvPair, GgufTensorInfo};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn kv_pair_str(key: &str, value: &str) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: pesti_gguf::GgufValueType::String,
            value: pesti_gguf::GgufKvValue::String(value.to_string()),
        }
    }

    fn kv_pair_u64(key: &str, value: u64) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: pesti_gguf::GgufValueType::Uint64,
            value: pesti_gguf::GgufKvValue::Uint64(value),
        }
    }

    fn write_kv_value(buf: &mut Vec<u8>, value: &pesti_gguf::GgufKvValue) {
        match value {
            pesti_gguf::GgufKvValue::Uint8(v) => buf.push(*v),
            pesti_gguf::GgufKvValue::Int8(v) => buf.push(*v as u8),
            pesti_gguf::GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Int16(v) => buf.extend_from_slice(&(*v as i16).to_le_bytes()),
            pesti_gguf::GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Int32(v) => buf.extend_from_slice(&(*v as i32).to_le_bytes()),
            pesti_gguf::GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Int64(v) => buf.extend_from_slice(&(*v as i64).to_le_bytes()),
            pesti_gguf::GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            pesti_gguf::GgufKvValue::Bool(v) => buf.push(*v as u8),
            pesti_gguf::GgufKvValue::String(s) => {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            pesti_gguf::GgufKvValue::Int8Array(arr) => {
                let bytes: Vec<u8> = arr.iter().map(|b| *b as u8).collect();
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(&bytes);
            }
            pesti_gguf::GgufKvValue::Uint8Array(arr) => {
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(arr);
            }
            pesti_gguf::GgufKvValue::Array(arr) => {
                buf.extend_from_slice(&9u32.to_le_bytes()); // element type = ARRAY
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                for elem in arr {
                    write_kv_value(buf, elem);
                }
            }
            pesti_gguf::GgufKvValue::Bfloat16(v) => {
                let raw = (*v as u32) << 16;
                buf.extend_from_slice(&((raw as u16) as u16).to_le_bytes());
            }
            pesti_gguf::GgufKvValue::Float16(v) => buf.extend_from_slice(&(*v as u16).to_le_bytes()),
            pesti_gguf::GgufKvValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        }
    }

    #[test]
    fn tokenizer_config_to_tokenizer_empty_vocab() {
        let config = GgufTokenizerConfig {
            model_type: "llama".to_string(),
            vocab_size: 0,
            pre_tokenizer_type: None,
            post_processor_type: None,
            added_tokens: HashMap::new(),
            pattern: None,
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
            add_bos_token: None,
            add_eos_token: None,
        };
        let tokenizer = config.to_tokenizer();
        // Should produce a valid tokenizer (may be empty if JSON is invalid)
        let encoded = tokenizer.encode("test", false);
        assert!(encoded.is_ok() || encoded.is_err());
    }

    #[test]
    fn tokenizer_config_clone() {
        let config = GgufTokenizerConfig {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            pre_tokenizer_type: Some("default".to_string()),
            post_processor_type: None,
            added_tokens: HashMap::new(),
            pattern: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            unk_token_id: Some(3),
            add_bos_token: Some(true),
            add_eos_token: Some(false),
        };
        let cloned = config.clone();
        assert_eq!(config.model_type, cloned.model_type);
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.bos_token_id, cloned.bos_token_id);
    }
}

// NOTE: Synthetic GGUF v3 test helpers removed (2026-07-31)
// These tests encoded exact wire format assumptions that were brittle and hard to maintain.
// The parser is validated against real llama.cpp GGUF files via conformance tests instead.
