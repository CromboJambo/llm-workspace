//! Tokenizer integration with GGUF vocab.
//!
//! Loads tokenizer configuration from GGUF KV pairs and wraps the tokenizers library.

use std::collections::HashMap;
use std::path::Path;

use crabjar_gguf::types::GgufHeader;

use crate::error::{Result, RunnerError};
use crate::tokenizer::Tokenizer as BaseTokenizer;

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

        Self {
            model_type,
            vocab_size,
            pre_tokenizer_type,
            post_processor_type,
            added_tokens,
            pattern,
        }
    }
}

/// Load a tokenizer from a GGUF file.
pub fn load_tokenizer_from_gguf(path: &Path) -> Result<(GgufTokenizerConfig, BaseTokenizer)> {
    let header = crabjar_gguf::parser::parse_gguf(path)
        .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

    let config = GgufTokenizerConfig::from_gguf_header(&header);
    let mut tokenizer = BaseTokenizer::new(&config.model_type);

    // Attempt to initialize BPE tokenizer
    tokenizer.init_bpe()
        .map_err(|e| RunnerError::Tokenizer(format!("BPE init failed: {e}")))?;

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

    fn make_test_gguf_with_vocab(path: &Path) {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 KV pairs
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor

        // KV: general.architecture = "llama"
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(8u32).to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"llama");

        // KV: tokenizer.ggml.model = "llama"
        let key = "tokenizer.ggml.model";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(8u32).to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"llama");

        // KV: tokenizer.ggml.tokens = 5
        let key = "tokenizer.ggml.tokens";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(4u32).to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes());

        // KV: tokenizer.ggml.bos_token_id = 1
        let key = "tokenizer.ggml.bos_token_id";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(4u32).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());

        // Tensor: test.weight [4] F32
        let name = "test.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Pad to data section
        let data_start = 256u64;
        buf.resize(data_start as usize, 0);
        buf.extend_from_slice(&[0u8; 16]); // tensor data

        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn tokenizer_config_from_gguf_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_with_vocab(&path);
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();

        let config = GgufTokenizerConfig::from_gguf_header(&header);
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.vocab_size, 32000); // default from header.vocab_size()
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
}
