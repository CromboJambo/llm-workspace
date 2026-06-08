//! Llama-style model: loads GGUF weights, wires transformer layers.
//!
//! Supports the llama architecture family (llama, mistral, phi3, etc.)
//! with standard tensor naming conventions.

use std::path::Path;

use crabjar_gguf::types::GgufHeader;
use tracing::debug;

use crate::error::{Result, RunnerError};
use crate::gguf_weight_loader::{GgufWeights, load_gguf_weights};
use crate::transformer::GgufTokenizer;
use crate::transformer::layer::{Attention, FeedForward, TransformerLayer};
use crate::transformer::linear::Linear;
use crate::transformer::rms_norm::RmsNorm;
use crate::transformer::rope::RopeConfig;
use crate::transformer::tokenizer::{GgufTokenizerConfig, load_tokenizer_from_gguf};

/// Model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelArch {
    #[default]
    Llama,
    Gemma,
    Qwen2,
    Qwen3,
    Phi3,
    Mixtral,
    Starcoder2,
}

/// Llama-style model configuration.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: ModelArch,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rms_norm_eps: f32,
}

impl LlamaConfig {
    /// Build config from a GGUF header.
    pub fn from_gguf_header(header: &GgufHeader) -> Self {
        let arch_str = header.architecture().unwrap_or("llama");
        let arch = match arch_str {
            "gemma" => ModelArch::Gemma,
            "qwen2" => ModelArch::Qwen2,
            "qwen3" => ModelArch::Qwen3,
            "phi3" => ModelArch::Phi3,
            "mixtral" => ModelArch::Mixtral,
            "starcoder2" => ModelArch::Starcoder2,
            _ => ModelArch::Llama,
        };

        let embed_dim = header.embedding_length().unwrap_or(4096) as usize;
        let num_heads = header.attention_head_count().unwrap_or(32) as usize;

        let num_kv_heads = match arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => header
                .get_kv_u32(&format!("{arch_str}.num_key_value_heads"))
                .unwrap_or(8) as usize,
            _ => header.attention_head_count_kv().unwrap_or(num_heads as u32) as usize,
        };

        let num_layers = header.block_count().unwrap_or(32) as usize;
        let head_dim = if num_heads > 0 {
            embed_dim / num_heads
        } else {
            64
        };
        let intermediate_dim = match arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => header
                .get_kv_u32(&format!("{arch_str}.feed_forward_length"))
                .unwrap_or(11008) as usize,
            _ => header.feed_forward_length().unwrap_or(11008) as usize,
        };
        let max_seq_len = header.context_length().unwrap_or(4096) as usize;
        let rope_base = 10000.0;
        let rope_dim = header.rope_dimension_count().unwrap_or(head_dim as i32) as usize;
        let rms_norm_eps = header.normalization_epsilon().unwrap_or(1e-5);

        let actual_head_dim = if rope_dim > 0 { rope_dim } else { head_dim };

        Self {
            arch,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim: actual_head_dim,
            embed_dim,
            intermediate_dim,
            max_seq_len,
            rope_base,
            rope_scaling_factor: None,
            rope_scaling_type: None,
            rms_norm_eps,
        }
    }

    /// Get the layer prefix for this architecture.
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        match self.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => {
                format!("model.layers.{layer_idx}.")
            }
            _ => format!("layers.{layer_idx}."),
        }
    }

    /// Check if this architecture uses `gate_proj` / `up_proj` / `down_proj` naming.
    pub fn uses_gate_up_down(&self) -> bool {
        matches!(self.arch, ModelArch::Qwen2 | ModelArch::Qwen3)
    }

    /// Check if this architecture uses `q_proj` / `k_proj` / `v_proj` / `o_proj` naming.
    pub fn uses_proj_naming(&self) -> bool {
        matches!(
            self.arch,
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3
        )
    }

    /// Get the attention weight suffix for this architecture.
    pub fn attn_weight_suffix(&self) -> &str {
        match self.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => "proj.weight",
            _ => ".weight",
        }
    }

    /// Get the embedding tensor name for this architecture.
    pub fn embedding_name(&self) -> &str {
        match self.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => "model.embed_tokens.weight",
            ModelArch::Gemma => "model.embed_tokens.weight",
            _ => "tok_embeddings.weight",
        }
    }

    /// Get the output/LM-head tensor name for this architecture.
    pub fn output_name(&self) -> &str {
        match self.arch {
            ModelArch::Gemma => "lm_head.weight",
            ModelArch::Qwen2 | ModelArch::Qwen3 => "lm_head.weight",
            _ => "output.weight",
        }
    }

    /// Get the final norm tensor name for this architecture (if any).
    pub fn final_norm_name(&self) -> Option<&str> {
        match self.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => Some("model.norm.weight"),
            _ => None,
        }
    }
}

/// A loaded Llama-style model ready for inference.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub token_embeddings: Option<Linear>,
    pub output: Option<Linear>,
    pub final_norm: Option<RmsNorm>,
    pub layers: Vec<TransformerLayer>,
    pub vocab_size: u32,
    pub tokenizer: Option<GgufTokenizer>,
    pub tokenizer_config: Option<GgufTokenizerConfig>,
}

impl LlamaModel {
    /// Load a Llama-style model from a GGUF file.
    pub fn load_gguf(path: &Path) -> Result<Self> {
        let _header = crabjar_gguf::parser::parse_gguf(path)
            .map_err(|e| RunnerError::ModelLoad(e.to_string()))?;
        let weights = load_gguf_weights(path).map_err(|e| RunnerError::ModelLoad(e.to_string()))?;
        let mut model = Self::from_gguf_weights(weights)?;

        // Load tokenizer from GGUF file
        if let Ok((tokenizer_config, tokenizer)) = load_tokenizer_from_gguf(path) {
            model.tokenizer_config = Some(tokenizer_config);
            model.tokenizer = Some(tokenizer);
            debug!(path = %path.display(), "Loaded model with tokenizer");
        } else {
            debug!(path = %path.display(), "No tokenizer found in GGUF file");
        }

        Ok(model)
    }

    /// Build a model from already-loaded GGUF weights.
    pub fn from_gguf_weights(weights: GgufWeights) -> Result<Self> {
        let header = &weights.header;
        let config = LlamaConfig::from_gguf_header(header);

        let vocab_size = header.vocab_size().unwrap_or(32000);
        let rope_config = RopeConfig::new(config.head_dim, config.rope_base, config.max_seq_len);

        // Load token embeddings — architecture-dependent name
        let embedding_name = config.embedding_name();
        let token_embeddings = weights
            .tensors
            .get(embedding_name)
            .map(|tensor_data| Linear::from_f16_weight(tensor_data, None));

        // Load output (LM head) — architecture-dependent name
        let output_name = config.output_name();
        let output = weights
            .tensors
            .get(output_name)
            .map(|tensor_data| Linear::from_f16_weight(tensor_data, None));

        // Build transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_layer(&weights, layer_idx, &config, &rope_config)?;
            layers.push(layer);
        }

        // Load final norm for architectures that have it (qwen2/qwen3)
        let final_norm = if let Some(norm_name) = config.final_norm_name() {
            weights
                .tensors
                .get(norm_name)
                .map(|tensor_data| RmsNorm::new(f16_bytes_to_f32(tensor_data), config.rms_norm_eps))
        } else {
            None
        };

        Ok(Self {
            config,
            token_embeddings,
            output,
            final_norm,
            layers,
            vocab_size,
            tokenizer: None,
            tokenizer_config: None,
        })
    }

    /// Load a single transformer layer from GGUF weights.
    fn load_layer(
        weights: &GgufWeights,
        layer_idx: usize,
        config: &LlamaConfig,
        _rope: &RopeConfig,
    ) -> Result<TransformerLayer> {
        let prefix = config.layer_prefix(layer_idx);

        // RMSNorm weights — architecture-dependent names
        let attention_norm_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}input_layernorm.weight"),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}norm_1.weight"),
            _ => format!("{prefix}attention_norm.weight"),
        };
        let attention_norm_data = weights
            .tensors
            .get(&attention_norm_name)
            .ok_or_else(|| RunnerError::ModelLoad(format!("missing {attention_norm_name}")))?;
        let attention_norm =
            RmsNorm::new(f16_bytes_to_f32(attention_norm_data), config.rms_norm_eps);

        let ffn_norm_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}post_attention_layernorm.weight"),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}norm_2.weight"),
            _ => format!("{prefix}ffn_norm.weight"),
        };
        let ffn_norm_data = weights
            .tensors
            .get(&ffn_norm_name)
            .ok_or_else(|| RunnerError::ModelLoad(format!("missing {ffn_norm_name}")))?;
        let ffn_norm = RmsNorm::new(f16_bytes_to_f32(ffn_norm_data), config.rms_norm_eps);

        // Attention weights — architecture-dependent naming
        let suffix = config.attn_weight_suffix();
        let wq_data = match config.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}q_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad(format!("missing q_proj weight"))),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wq.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wq.weight"))
                }),
        }?;
        let wk_data = match config.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}k_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad(format!("missing k_proj weight"))),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wk.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wk.weight"))
                }),
        }?;
        let wv_data = match config.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}v_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad(format!("missing v_proj weight"))),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wv.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wv.weight"))
                }),
        }?;
        let wo_data = match config.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}o_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad(format!("missing o_proj weight"))),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wo.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wo.weight"))
                }),
        }?;

        let wq = Linear::from_f16_weight(wq_data, None);
        let wk = Linear::from_f16_weight(wk_data, None);
        let wv = Linear::from_f16_weight(wv_data, None);
        let wo = Linear::from_f16_weight(wo_data, None);

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.head_dim,
            config.num_heads,
            config.num_kv_heads,
        );

        // FFN weights — architecture-dependent naming
        let (w1_data, w2_data, w3_data) = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => {
                let w1 = weights
                    .tensors
                    .get(&format!("{prefix}gate.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing gate weight".into()))?;
                let w2 = weights
                    .tensors
                    .get(&format!("{prefix}down_proj.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing down_proj weight".into()))?;
                let w3 = weights
                    .tensors
                    .get(&format!("{prefix}up.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing up weight".into()))?;
                (w1, w2, w3)
            }
            _ => {
                let w1 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w1.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w1.weight"))
                    })?;
                let w2 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w2.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w2.weight"))
                    })?;
                let w3 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w3.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w3.weight"))
                    })?;
                (w1, w2, w3)
            }
        };

        let w1 = Linear::from_f16_weight(w1_data, None);
        let w2 = Linear::from_f16_weight(w2_data, None);
        let w3 = Linear::from_f16_weight(w3_data, None);

        let feed_forward = FeedForward::new(w1, w2, w3, config.intermediate_dim);

        Ok(TransformerLayer::new(
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        ))
    }

    /// Run the model on a single token input.
    ///
    /// `token` — input token ID
    /// `start_pos` — position in the sequence (for RoPE)
    /// Returns: logits over vocabulary [vocab_size]
    pub fn forward(&self, token: u32, start_pos: usize) -> Result<Vec<f32>> {
        let logits = self.embed(token, start_pos)?;
        self.apply_output_head(&logits)
    }

    /// Embed a single token ID into its embedding vector.
    pub fn embed(&self, token: u32, _start_pos: usize) -> Result<Vec<f32>> {
        let emb = self
            .token_embeddings
            .as_ref()
            .ok_or_else(|| RunnerError::ModelLoad("missing token embeddings".to_string()))?;

        let token_idx = token as usize;
        let emb_dim = emb.in_features;
        let start = token_idx * emb_dim;
        let x = emb.weight[start..start + emb_dim].to_vec();
        Ok(x)
    }

    /// Apply the output (LM head) to get logits from hidden states.
    pub fn apply_output_head(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| RunnerError::ModelLoad("missing output layer".to_string()))?;

        let logits = output.forward(hidden, 1);
        Ok(logits)
    }

    /// Pass hidden states through all transformer layers.
    pub fn forward_layers(&self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        let _embed_dim = hidden.len();
        let mut h = hidden.to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, 1, 1, start_pos + layer_idx);
        }

        // Apply final norm for architectures that have it (qwen2/qwen3)
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h, 1);
        }

        Ok(h)
    }

    /// Get the model architecture string from GGUF header.
    pub fn architecture(header: &GgufHeader) -> Option<&str> {
        header.architecture()
    }

    /// Check if this model supports the given architecture.
    pub fn is_supported_architecture(arch: &str) -> bool {
        matches!(
            arch,
            "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2" | "qwen3" | "starcoder2"
        )
    }

    /// Sample a token from logits using the configured sampling strategy.
    pub fn sample_from_logits(
        logits: &[f32],
        config: &crate::transformer::SamplingConfig,
        rng: &mut rand::rngs::StdRng,
    ) -> u32 {
        crate::transformer::sample(logits, config, rng)
    }

    /// Greedy decode: argmax over logits.
    pub fn argmax_from_logits(logits: &[f32]) -> u32 {
        crate::transformer::argmax(logits)
    }

    /// Generate tokens autoregressively.
    ///
    /// `prompt` — input token IDs
    /// `max_tokens` — maximum tokens to generate
    /// `sampling_config` — sampling parameters (temperature, top-p, top-k)
    /// `rng` — random number generator
    /// `stop_tokens` — token IDs that stop generation
    ///
    /// Returns: generated token IDs (excluding prompt)
    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        sampling_config: &crate::transformer::SamplingConfig,
        rng: &mut rand::rngs::StdRng,
        stop_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();

        // Process prompt: for each token, run forward and update position
        let mut context = prompt.to_vec();
        let mut pos = 0;

        // Use the last token for each forward pass (autoregressive)
        for _ in 0..max_tokens {
            let last_token = *context
                .last()
                .ok_or_else(|| RunnerError::ModelLoad("empty context".to_string()))?;

            // Get logits for the last token
            let logits = self.forward(last_token, pos)?;

            // Sample next token
            let next_token = if sampling_config.temperature == 0.0 {
                Self::argmax_from_logits(&logits)
            } else {
                Self::sample_from_logits(&logits, sampling_config, rng)
            };

            // Check for stop tokens
            if stop_tokens.contains(&next_token) {
                break;
            }

            generated.push(next_token);
            context.push(next_token);
            pos += 1;

            if pos >= self.config.max_seq_len {
                break;
            }
        }

        Ok(generated)
    }
}

/// Convert f16 tensor bytes to f32 Vec.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1F) as i32;
            let frac = (bits & 0x3FF) as u32;

            if exp == 0 {
                if frac == 0 {
                    f32::from_bits(sign << 31)
                } else {
                    let f32_bits = (sign << 31) | (frac << 13);
                    f32::from_bits(f32_bits)
                }
            } else if exp == 31 {
                f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
            } else {
                let f32_exp = (exp - 15 + 127) as u32;
                let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
                f32::from_bits(f32_bits)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crabjar_gguf::{GgufKvPair, GgufTensorInfo, compute_data_section_start};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn make_test_gguf_llama(path: &Path) {
        // KV pairs — numeric values must use correct type tags, not String
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
            kv_pair_u32("llama.context_length", 4096),
            kv_pair_u32("llama.embedding_length", 64),
            kv_pair_u32("llama.block_count", 2),
            kv_pair_u32("llama.attention.head_count", 4),
            kv_pair_u32("llama.attention.head_count_kv", 2),
            kv_pair_u32("llama.feed_forward_length", 128),
            kv_pair_i32("llama.rope.dimension_count", 64),
            kv_pair_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
            kv_pair_u32("tokenizer.ggml.tokens", 32000),
        ];

        // Tensor metadata — compute sizes first
        let tensor_shapes: Vec<Vec<u64>> = vec![
            vec![64],        // tok_embeddings
            vec![32000, 64], // output
            vec![64, 64],    // layers.0.attention.wq
            vec![64, 64],    // layers.0.attention.wk
            vec![64, 64],    // layers.0.attention.wv
            vec![64, 64],    // layers.0.attention.wo
            vec![64],        // layers.0.attention_norm
            vec![64],        // layers.0.ffn_norm
            vec![64, 128],   // layers.0.feed_forward.w1
            vec![128, 64],   // layers.0.feed_forward.w2
            vec![64, 128],   // layers.0.feed_forward.w3
        ];
        let tensor_names: Vec<&str> = vec![
            "tok_embeddings.weight",
            "output.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.feed_forward.w3.weight",
        ];

        // Compute offsets
        let mut offset = 0u64;
        let tensors: Vec<GgufTensorInfo> = tensor_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                let tensor_info = GgufTensorInfo {
                    name: tensor_names[i].to_string(),
                    shape: shape.clone(),
                    offset,
                    dtype: 1,
                };
                let elems: u64 = shape.iter().product();
                offset += elems * 2; // F16 = 2 bytes
                tensor_info
            })
            .collect();

        // Tensor metadata — compute sizes first
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo {
                name: "tok_embeddings.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "output.weight".to_string(),
                shape: vec![32000u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wq.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wk.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wv.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wo.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention_norm.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.feed_forward.w1.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.feed_forward.w2.weight".to_string(),
                shape: vec![128u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.feed_forward.w3.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.ffn_norm.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
        ];

        let data_section_start =
            crabjar_gguf::compute_data_section_start(3, &kv_pairs, &tensors, None);

        // Write file
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

        // Write tensor metadata
        for tensor in &tensors {
            let name_bytes = tensor.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }

        // Pad to data_section_start and write tensor data
        let total_tensor_bytes: u64 = tensors
            .iter()
            .map(|t| {
                let elems: u64 = t.shape.iter().product();
                elems * 2 // F16 = 2 bytes
            })
            .sum();
        buf.resize((data_section_start + total_tensor_bytes) as usize, 0);
        for i in 0..total_tensor_bytes as usize {
            buf[data_section_start as usize + i] = if i % 2 == 0 { 0x00 } else { 0x3F };
        }

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

    fn kv_pair_f32(key: &str, value: f32) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: crabjar_gguf::GgufValueType::Float32,
            value: crabjar_gguf::GgufKvValue::Float32(value),
        }
    }

    fn kv_pair_i32(key: &str, value: i32) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: crabjar_gguf::GgufValueType::Int32,
            value: crabjar_gguf::GgufKvValue::Int32(value),
        }
    }

    fn write_kv_value(buf: &mut Vec<u8>, value: &crabjar_gguf::GgufKvValue) {
        match value {
            crabjar_gguf::GgufKvValue::Uint8(v) => buf.push(*v),
            crabjar_gguf::GgufKvValue::Int8(v) => buf.push(*v as u8),
            crabjar_gguf::GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Int16(v) => {
                buf.extend_from_slice(&(*v as i16).to_le_bytes())
            }
            crabjar_gguf::GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Int32(v) => {
                buf.extend_from_slice(&(*v as i32).to_le_bytes())
            }
            crabjar_gguf::GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            crabjar_gguf::GgufKvValue::Int64(v) => {
                buf.extend_from_slice(&(*v as i64).to_le_bytes())
            }
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
                buf.extend_from_slice(&9u32.to_le_bytes());
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                for elem in arr {
                    write_kv_value(buf, elem);
                }
            }
            crabjar_gguf::GgufKvValue::Bfloat16(v) => {
                let raw = (*v as u32) << 16;
                buf.extend_from_slice(&((raw as u16) as u16).to_le_bytes());
            }
            crabjar_gguf::GgufKvValue::Float16(v) => {
                buf.extend_from_slice(&(*v as u16).to_le_bytes())
            }
        }
    }

    #[test]
    fn llama_config_from_header() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        make_test_gguf_llama(&path);
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();

        let config = LlamaConfig::from_gguf_header(&header);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.embed_dim, 64);
        assert_eq!(config.intermediate_dim, 128);
    }

    #[test]
    fn is_supported_architecture() {
        assert!(LlamaModel::is_supported_architecture("llama"));
        assert!(LlamaModel::is_supported_architecture("mistral"));
        assert!(LlamaModel::is_supported_architecture("qwen2"));
        assert!(!LlamaModel::is_supported_architecture("unknown"));
    }

    #[test]
    fn f16_bytes_to_f32_known() {
        let pack = |v: f32| -> [u8; 2] {
            let bits = v.to_bits();
            let sign = (bits >> 31) & 1;
            let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
            let frac = ((bits >> 13) & 0x3FF) as u16;
            if exp <= 0 {
                let biased = ((sign << 15) as u16) | frac;
                biased.to_le_bytes()
            } else if exp >= 31 {
                ((sign << 15) as u16 | 0x7C00).to_le_bytes()
            } else {
                (((sign << 15) as u16) | ((exp as u16) << 10) | frac).to_le_bytes()
            }
        };

        let data: Vec<u8> = vec![pack(1.0), pack(2.0), pack(0.5), pack(-1.0)]
            .into_iter()
            .flatten()
            .collect();
        let result = f16_bytes_to_f32(&data);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 0.5).abs() < 1e-5);
        assert!((result[3] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn llama_model_from_gguf_weights_builds_layers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_llama(&path);
        let weights = load_gguf_weights(&path).unwrap();
        // Verify tensors were loaded
        assert!(weights.tensors.contains_key("tok_embeddings.weight"));
        assert!(weights.tensors.contains_key("output.weight"));
        // F16 tensor dequantized to f32: 64 elements * 4 bytes = 256 bytes
        assert_eq!(weights.tensors["tok_embeddings.weight"].len(), 256);
    }

    #[test]
    fn llama_model_config_defaults_on_missing_keys() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 1,
        }];
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = LlamaConfig::from_gguf_header(&header);
        // Should use defaults when keys are missing
        assert_eq!(config.num_layers, 32); // default block_count
        assert_eq!(config.num_heads, 32); // default attention_head_count
        assert_eq!(config.embed_dim, 4096); // default embedding_length
        assert_eq!(config.max_seq_len, 4096); // default context_length
        assert_eq!(config.rope_base, 10000.0); // hardcoded default
    }

    #[test]
    fn llama_model_is_supported_arch_variants() {
        assert!(LlamaModel::is_supported_architecture("llama"));
        assert!(LlamaModel::is_supported_architecture("mistral"));
        assert!(LlamaModel::is_supported_architecture("mixtral"));
        assert!(LlamaModel::is_supported_architecture("gemma"));
        assert!(LlamaModel::is_supported_architecture("phi3"));
        assert!(LlamaModel::is_supported_architecture("qwen2"));
        assert!(LlamaModel::is_supported_architecture("qwen3"));
        assert!(LlamaModel::is_supported_architecture("starcoder2"));
        assert!(!LlamaModel::is_supported_architecture(""));
        assert!(!LlamaModel::is_supported_architecture("bert"));
        assert!(!LlamaModel::is_supported_architecture("gpt2"));
    }

    #[test]
    fn llama_model_architecture_from_header() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "phi3")];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        assert_eq!(LlamaModel::architecture(&header), Some("phi3"));
    }

    #[test]
    fn llama_config_rope_dimension_fallback() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
            kv_pair_u32("llama.context_length", 4096),
            kv_pair_u32("llama.embedding_length", 64),
            kv_pair_u32("llama.block_count", 2),
            kv_pair_u32("llama.attention.head_count", 4),
            kv_pair_u32("llama.attention.head_count_kv", 2),
            kv_pair_u32("llama.feed_forward_length", 128),
            // No rope.dimension_count — should fall back to head_dim
            kv_pair_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 1,
        }];
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = LlamaConfig::from_gguf_header(&header);
        // Without rope.dimension_count, should fall back to head_dim (64/4 = 16)
        assert_eq!(config.head_dim, 16);
    }

    #[test]
    fn llama_config_detects_gemma_architecture() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "gemma"),
            kv_pair_str("general.file_type", "F16"),
            kv_pair_u32("gemma.context_length", 4096),
            kv_pair_u32("gemma.embedding_length", 64),
            kv_pair_u32("gemma.block_count", 2),
            kv_pair_u32("gemma.attention.head_count", 4),
            kv_pair_u32("gemma.attention.head_count_kv", 2),
            kv_pair_u32("gemma.feed_forward_length", 128),
            kv_pair_u32("gemma.attention.layer_norm_rms_epsilon", 1000000u32), // gemma uses 1e6 scaled by 1e6
            kv_pair_i32("gemma.rope.dimension_count", 64),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo {
                name: "model.embed_tokens.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "lm_head.weight".to_string(),
                shape: vec![32000u64, 64u64],
                offset: 128,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.input_layernorm.weight".to_string(),
                shape: vec![64u64],
                offset: 6553600,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.q_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 6553664,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.k_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 13107328,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.v_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 19660992,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.o_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 26214656,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.post_attention_layernorm.weight".to_string(),
                shape: vec![64u64],
                offset: 32768320,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.gate_proj.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 32768384,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.up_proj.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 32768512,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.down_proj.weight".to_string(),
                shape: vec![128u64, 64u64],
                offset: 32768768,
                dtype: 1,
            },
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = LlamaConfig::from_gguf_header(&header);
        assert_eq!(config.arch, ModelArch::Gemma);
        assert_eq!(config.layer_prefix(0), "model.layers.0.");
        assert_eq!(config.embedding_name(), "model.embed_tokens.weight");
        assert_eq!(config.output_name(), "lm_head.weight");
        assert!(config.final_norm_name().is_none());
        assert!(config.uses_proj_naming());
        assert_eq!(config.attn_weight_suffix(), "proj.weight");
    }

    #[test]
    fn llama_config_detects_qwen2_architecture() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "qwen2"),
            kv_pair_str("general.file_type", "F16"),
            kv_pair_u32("qwen2.context_length", 4096),
            kv_pair_u32("qwen2.embedding_length", 64),
            kv_pair_u32("qwen2.block_count", 2),
            kv_pair_u32("qwen2.attention.head_count", 4),
            kv_pair_u32("qwen2.attention.head_count_kv", 2),
            kv_pair_u32("qwen2.feed_forward_length", 128),
            kv_pair_u32("qwen2.attention.layer_norm_rms_epsilon", 1000000u32),
            kv_pair_i32("qwen2.rope.dimension_count", 64),
            kv_pair_u32("qwen2.num_key_value_heads", 2),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo {
                name: "model.embed_tokens.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.norm.weight".to_string(),
                shape: vec![64u64],
                offset: 64,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "lm_head.weight".to_string(),
                shape: vec![32000u64, 64u64],
                offset: 128,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.norm_1.weight".to_string(),
                shape: vec![64u64],
                offset: 6553600,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.q_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 6553664,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.k_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 13107328,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.v_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 19660992,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.o_proj.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 26214656,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.norm_2.weight".to_string(),
                shape: vec![64u64],
                offset: 32768320,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.gate.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 32768384,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.up.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 32768512,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "model.layers.0.down_proj.weight".to_string(),
                shape: vec![128u64, 64u64],
                offset: 32768768,
                dtype: 1,
            },
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = LlamaConfig::from_gguf_header(&header);
        assert_eq!(config.arch, ModelArch::Qwen2);
        assert_eq!(config.layer_prefix(0), "model.layers.0.");
        assert_eq!(config.embedding_name(), "model.embed_tokens.weight");
        assert_eq!(config.output_name(), "lm_head.weight");
        assert_eq!(config.final_norm_name(), Some("model.norm.weight"));
        assert!(config.uses_proj_naming());
        assert!(config.uses_gate_up_down());
        assert_eq!(config.attn_weight_suffix(), "proj.weight");
        // num_kv_heads should come from qwen2.num_key_value_heads
        assert_eq!(config.num_kv_heads, 2);
    }

    #[test]
    fn llama_config_detects_phi3_architecture() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "phi3"),
            kv_pair_str("general.file_type", "F16"),
            kv_pair_u32("phi3.context_length", 4096),
            kv_pair_u32("phi3.embedding_length", 64),
            kv_pair_u32("phi3.block_count", 2),
            kv_pair_u32("phi3.attention.head_count", 4),
            kv_pair_u32("phi3.attention.head_count_kv", 2),
            kv_pair_u32("phi3.feed_forward_length", 128),
            kv_pair_f32("phi3.attention.layer_norm_epsilon", 1e-5),
            kv_pair_i32("phi3.rope.dimension_count", 64),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo {
                name: "tok_embeddings.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "output.weight".to_string(),
                shape: vec![32000u64, 64u64],
                offset: 128,
                dtype: 1,
            },
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = crabjar_gguf::parser::parse_gguf(&path).unwrap();
        let config = LlamaConfig::from_gguf_header(&header);
        assert_eq!(config.arch, ModelArch::Phi3);
        assert_eq!(config.layer_prefix(0), "layers.0.");
        assert_eq!(config.embedding_name(), "tok_embeddings.weight");
        assert_eq!(config.output_name(), "output.weight");
        assert!(config.final_norm_name().is_none());
        assert!(!config.uses_proj_naming());
        assert!(!config.uses_gate_up_down());
    }

    #[test]
    fn llama_config_layer_prefix_per_arch() {
        let kv_pairs: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "llama")];
        let tensors: Vec<GgufTensorInfo> = vec![];
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
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&PathBuf::from("/tmp/_arch_test.gguf"), &buf).unwrap();

        // Test llama prefix
        let kv_llama: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "llama")];
        let h_llama = GgufHeader {
            version: 3,
            kv_pairs: kv_llama,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(
            LlamaConfig::from_gguf_header(&h_llama).layer_prefix(5),
            "layers.5."
        );

        // Test gemma prefix
        let kv_gemma: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "gemma")];
        let h_gemma = GgufHeader {
            version: 3,
            kv_pairs: kv_gemma,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(
            LlamaConfig::from_gguf_header(&h_gemma).layer_prefix(5),
            "model.layers.5."
        );

        // Test qwen2 prefix
        let kv_qwen: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "qwen2")];
        let h_qwen = GgufHeader {
            version: 3,
            kv_pairs: kv_qwen,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(
            LlamaConfig::from_gguf_header(&h_qwen).layer_prefix(5),
            "model.layers.5."
        );

        // Test phi3 prefix (llama-style)
        let kv_phi3: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "phi3")];
        let h_phi3 = GgufHeader {
            version: 3,
            kv_pairs: kv_phi3,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(
            LlamaConfig::from_gguf_header(&h_phi3).layer_prefix(5),
            "layers.5."
        );
    }

    #[test]
    fn llama_config_embedding_output_names_per_arch() {
        let kv_llama: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "llama")];
        let h_llama = GgufHeader {
            version: 3,
            kv_pairs: kv_llama,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        let c_llama = LlamaConfig::from_gguf_header(&h_llama);
        assert_eq!(c_llama.embedding_name(), "tok_embeddings.weight");
        assert_eq!(c_llama.output_name(), "output.weight");

        let kv_gemma: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "gemma")];
        let h_gemma = GgufHeader {
            version: 3,
            kv_pairs: kv_gemma,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        let c_gemma = LlamaConfig::from_gguf_header(&h_gemma);
        assert_eq!(c_gemma.embedding_name(), "model.embed_tokens.weight");
        assert_eq!(c_gemma.output_name(), "lm_head.weight");

        let kv_qwen: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "qwen2")];
        let h_qwen = GgufHeader {
            version: 3,
            kv_pairs: kv_qwen,
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        let c_qwen = LlamaConfig::from_gguf_header(&h_qwen);
        assert_eq!(c_qwen.embedding_name(), "model.embed_tokens.weight");
        assert_eq!(c_qwen.output_name(), "lm_head.weight");
    }
}
