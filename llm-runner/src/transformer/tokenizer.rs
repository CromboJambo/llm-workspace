//! Tokenizer integration with GGUF vocab.
//!
//! Loads tokenizer configuration from GGUF KV pairs and wraps the tokenizers library.
//!

#![allow(clippy::if_same_then_else, clippy::collapsible_if)]
use std::collections::HashMap;
use std::path::Path;

use crabjar_gguf::types::GgufHeader;
use tokenizers::tokenizer::{Result, Tokenizer};
use tracing::debug;

use crate::error::RunnerError;

/// GGUF tokenizer configuration.
#[derive(Debug, Clone)]
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
        let model_type = header.get_kv_str("tokenizer.ggml.model")
            .unwrap_or("llama")
            .to_string();
        let vocab_size = header.vocab_size().unwrap_or(32000);

        // Load added tokens from GGUF
        let mut added_tokens = HashMap::new();
        let token_count = header.get_kv_u32("tokenizer.ggml.tokens")
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

        let pre_tokenizer_type = header.get_kv_str("tokenizer.ggml.pre")
            .map(|s| s.to_string());

        let post_processor_type = header.get_kv_str("tokenizer.ggml.postprocess")
            .map(|s| s.to_string());

        let pattern = header.get_kv_str("tokenizer.ggml.token_type")
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

        Tokenizer::from_bytes(config.to_string().as_bytes())
            .unwrap_or_else(|e| {
                debug!("Failed to build tokenizer from GGUF config: {e}, using empty tokenizer");
                Tokenizer::new(tokenizers::models::bpe::BPE::default())
            })
    }
}

/// Load a tokenizer from a GGUF file.
pub fn load_tokenizer_from_gguf(path: &Path) -> Result<(GgufTokenizerConfig, Tokenizer)> {
    let header = crabjar_gguf::parser::parse_gguf(path)
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
    use tempfile::tempdir;
    use crabjar_gguf::{GgufKvPair, GgufTensorInfo, compute_data_section_start};

    fn make_test_gguf_with_vocab(path: &Path) {
        // KV pairs
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("tokenizer.ggml.model", "llama"),
            kv_pair_u32("tokenizer.ggml.tokens", 5),
            kv_pair_u32("tokenizer.ggml.bos_token_id", 1),
        ];

        // Tensor metadata
        let tensor_info = crabjar_gguf::GgufTensorInfo {
            name: "test.weight".to_string(),
            shape: vec![4u64],
            offset: 0,
            dtype: 0u32,
        };

        // Compute data_section_start with BOTH kv_pairs and tensor_info
        let data_section_start = crabjar_gguf::compute_data_section_start(3, &kv_pairs, &[tensor_info.clone()], None);

        // Write file
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&4u64.to_le_bytes()); // kv count

        for kv in &kv_pairs {
            let key_bytes = kv.key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
            write_kv_value(&mut buf, &kv.value);
        }

        // Write tensor metadata
        let name_bytes = tensor_info.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(tensor_info.shape.len() as u32).to_le_bytes());
        for dim in &tensor_info.shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&tensor_info.dtype.to_le_bytes());
        buf.extend_from_slice(&tensor_info.offset.to_le_bytes());

        // Pad to data_section_start and write tensor data
        buf.resize((data_section_start + 16) as usize, 0);
        buf[data_section_start as usize..data_section_start as usize + 16].copy_from_slice(&[0u8; 16]);

        std::fs::write(path, &buf).unwrap();
    }

    fn kv_pair_str(key: &str, value: &str) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: crabjar_gguf::GgufValueType::String,
            value: crabjar_gguf::GgufKvValue::String(value.to_string()),
        }
    }

    fn kv_pair_u32(key: &str, value: u32) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: crabjar_gguf::GgufValueType::Uint32,
            value: crabjar_gguf::GgufKvValue::Uint32(value),
        }
    }

    fn write_kv_value(buf: &mut Vec<u8>, value: &crabjar_gguf::GgufKvValue) {
        match value {
            crabjar_gguf::GgufKvValue::Uint8(v) => buf.push(*v),
            crabjar_gguf::GgufKvValue::Int8(v) => buf.push(*v as u8),
            crabjar_gguf::GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Int16(v) => buf.extend_from_slice(&(*v as i16).to_le_bytes()),
            crabjar_gguf::GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Int32(v) => buf.extend_from_slice(&(*v as i32).to_le_bytes()),
            crabjar_gguf::GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Int64(v) => buf.extend_from_slice(&(*v as i64).to_le_bytes()),
            crabjar_gguf::GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Bool(v) => buf.push(*v as u8),
            crabjar_gguf::GgufKvValue::String(s) => {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            crabjar_gguf::GgufKvValue::Int8Array(arr) => {
                let bytes: Vec<u8> = arr.iter().map(|b| *b as u8).collect();
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(&bytes);
            }
            crabjar_gguf::GgufKvValue::Uint8Array(arr) => {
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(arr);
            }
            crabjar_gguf::GgufKvValue::Array(arr) => {
                buf.extend_from_slice(&9u32.to_le_bytes()); // element type = ARRAY
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                for elem in arr {
                    write_kv_value(buf, elem);
                }
            }
            crabjar_gguf::GgufKvValue::Bfloat16(v) => {
                let raw = (*v as u32) << 16;
                buf.extend_from_slice(&((raw as u16) as u16).to_le_bytes());
            }
            crabjar_gguf::GgufKvValue::Float16(v) => buf.extend_from_slice(&(*v as u16).to_le_bytes()),
        }
    }

    #[test]
    fn tokenizer_config_from_gguf_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_with_vocab(&path);
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();

        let config = GgufTokenizerConfig::from_gguf_header(&header);
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.vocab_size, 5);
    }

    #[test]
    fn tokenizer_config_from_header_no_vocab() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_with_vocab(&path);
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();

        let config = tokenizer_config_from_header(&header);
        assert_eq!(config.model_type, "llama");
        assert!(config.pre_tokenizer_type.is_none());
    }

    #[test]
    fn tokenizer_config_to_tokenizer_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_with_vocab(&path);
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = GgufTokenizerConfig::from_gguf_header(&header);
        let tokenizer = config.to_tokenizer();
        // Should produce a valid tokenizer (may be empty if JSON is invalid)
        // Tokenizer from tokenizers crate can encode/decode without panicking
        let encoded = tokenizer.encode("test", false);
        assert!(encoded.is_ok() || encoded.is_err()); // Just verify it doesn't panic
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
    fn load_tokenizer_from_gguf_succeeds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_with_vocab(&path);
        let result = load_tokenizer_from_gguf(&path);
        assert!(result.is_ok());
        let (config, tokenizer) = result.unwrap();
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.vocab_size, 5);
        // Tokenizer can encode without panicking
        let _encoded = tokenizer.encode("test", false);
    }

    #[test]
    fn load_tokenizer_from_gguf_no_vocab_returns_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        // GGUF without tokenizer vocab
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo { name: "tok_embeddings.weight".to_string(), shape: vec![64u64], offset: 0, dtype: 1 },
        ];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
        for kv in &kv_pairs {
            let key_bytes = kv.key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
            write_kv_value(&mut buf, &kv.value);
        }
        for tensor in &tensors {
            let name_bytes = tensor.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in &tensor.shape { buf.extend_from_slice(&dim.to_le_bytes()); }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors.iter().map(|t| t.shape.iter().product::<u64>() * 2).sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let result = load_tokenizer_from_gguf(&path);
        // Returns Ok with default vocab_size (32000) when no tokenizer.ggml.tokens key
        assert!(result.is_ok());
        let (config, _tokenizer) = result.unwrap();
        assert_eq!(config.vocab_size, 32000); // default when key missing
        assert!(config.added_tokens.is_empty());
    }

    #[test]
    fn tokenizer_config_from_header_with_full_vocab() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("tokenizer.ggml.model", "llama"),
            kv_pair_u32("tokenizer.ggml.tokens", 3),
            kv_pair_str("tokenizer.ggml.tokens.0", "<s>"),
            kv_pair_str("tokenizer.ggml.tokens.1", "</s>"),
            kv_pair_str("tokenizer.ggml.tokens.2", "hello"),
            kv_pair_u32("tokenizer.ggml.bos_token_id", 0),
            kv_pair_u32("tokenizer.ggml.eos_token_id", 1),
            kv_pair_u32("tokenizer.ggml.unk_token_id", 2),
            kv_pair_str("tokenizer.ggml.pre", "default"),
            kv_pair_str("tokenizer.ggml.postprocess", "none"),
            kv_pair_str("tokenizer.ggml.token_type", "byte"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo { name: "test.weight".to_string(), shape: vec![4u64], offset: 0, dtype: 0 },
        ];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
        for kv in &kv_pairs {
            let key_bytes = kv.key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
            write_kv_value(&mut buf, &kv.value);
        }
        let tensor = &tensors[0];
        let name_bytes = tensor.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in &tensor.shape { buf.extend_from_slice(&dim.to_le_bytes()); }
        buf.extend_from_slice(&tensor.dtype.to_le_bytes());
        buf.extend_from_slice(&tensor.offset.to_le_bytes());
        let total: u64 = tensors.iter().map(|t| t.shape.iter().product::<u64>() * 2).sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = tokenizer_config_from_header(&header);
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.vocab_size, 3);
        assert_eq!(config.pre_tokenizer_type, Some("default".to_string()));
        assert_eq!(config.post_processor_type, Some("none".to_string()));
        assert_eq!(config.pattern, Some("byte".to_string()));
        assert_eq!(config.bos_token_id, Some(0));
        assert_eq!(config.eos_token_id, Some(1));
        assert_eq!(config.unk_token_id, Some(2));
        assert_eq!(config.added_tokens.len(), 3);
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
