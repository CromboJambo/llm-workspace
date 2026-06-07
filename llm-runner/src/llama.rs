//! llama.rs — High-level Rust API over llama.cpp via llama-cpp-2.
//!
//! This module provides a clean, idiomatic Rust interface for:
//! - Loading GGUF models
//! - Creating inference contexts with KV cache
//! - Tokenizing / detokenizing
//! - Running inference (prefill + decode)
//! - Sampling with configurable parameters (temperature, top-k, top-p, etc.)
//! - Chat template application (system, user, assistant)
//! - Session save/load
//! - Embeddings
//! - Grammar-constrained decoding
//! - Function calling (OpenAI compat)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use crabjar_llm_runner::llama::{LlamaRunner, SamplingConfig};
//!
//! let runner = LlamaRunner::builder()
//!     .model_path("/path/to/model.gguf")
//!     .n_ctx(4096)
//!     .n_batch(512)
//!     .build()?;
//!
//! let result = runner.generate(
//!     "Explain quantum computing in one sentence.",
//!     SamplingConfig::default(),
//! )?;
//! println!("{}", result.text);
//! ```

pub mod sampler;
pub mod model;
pub mod context;
pub mod session;

use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::{
    llama_backend,
    model::{LlamaModel, LlamaModelParams},
    context::{
        LlamaContext,
        params::{LlamaContextParams, KvCacheType},
    },
    llama_batch,
    token::LlamaToken,
    sampling::{LlamaSampler, components::*},
    gguf::GgufReader,
};

use anyhow::Result;
use tracing::{info, warn};

pub use sampler::SamplingConfig;
pub use model::ModelInfo;
pub use context::ContextConfig;
pub use session::SessionManager;

// ── Re-exported types from llama-cpp-2 that users may need ──

pub use llama_cpp_2::model::LlamaChatMessage;
pub use llama_cpp_2::context::session::LlamaStateSeqFlags;

/// Result of a generation run.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated text.
    pub text: String,
    /// All tokens produced (including the prompt).
    pub tokens: Vec<LlamaToken>,
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
    /// Number of new tokens generated.
    pub generated_tokens: usize,
    /// Time stats from llama.cpp.
    pub load_time_ms: f64,
    pub prompt_eval_ms: f64,
    pub eval_ms: f64,
}

/// Builder for [`LlamaRunner`].
pub struct LlamaRunnerBuilder {
    model_path: String,
    context: LlamaContext,
    model: LlamaModel,
    sampling: SamplingConfig,
}

impl LlamaRunnerBuilder {
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        let model_path = model_path.as_ref().to_string_lossy().to_string();
        Self {
            model_path,
            context: panic!("call build() to initialize"),
            model: panic!("call build() to initialize"),
            sampling: SamplingConfig::default(),
        }
    }

    /// Set the context size (number of tokens in KV cache).
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.sampling.n_ctx = n_ctx;
        self
    }

    /// Set the batch size for token processing.
    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.sampling.n_batch = n_batch;
        self
    }

    /// Set the number of threads to use.
    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.sampling.n_threads = n_threads;
        self
    }

    /// Set the number of layers to offload to GPU. -1 = auto, 0 = CPU only.
    pub fn n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.sampling.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Set the KV cache type.
    pub fn kv_cache_type(mut self, kv_cache_type: KvCacheType) -> Self {
        self.sampling.kv_cache_type = kv_cache_type;
        self
    }

    /// Set the sampling configuration.
    pub fn sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Build the runner. This loads the model and creates the context.
    pub fn build(self) -> Result<LlamaRunner> {
        // Initialize the llama.cpp backend (CPU/GPU)
        llama_backend::init();
        info!("llama.cpp backend initialized");

        // Load the model
        let model_params = LlamaModelParams::default().with_n_gpu_layers(self.sampling.n_gpu_layers);
        let model = LlamaModel::load_from_file(&model_params, &self.model_path)?;
        info!(
            "Loaded model: {} params={}, n_embd={}, n_layer={}, n_head={}, n_head_kv={}",
            self.model_path,
            model.n_params(),
            model.n_embd(),
            model.n_layer(),
            model.n_head(),
            model.n_head_kv(),
        );

        // Create the context
        let mut ctx_params = LlamaContextParams::default()
            .with_seed(42)
            .with_n_ctx(self.sampling.n_ctx)
            .with_n_batch(self.sampling.n_batch)
            .with_n_threads(self.sampling.n_threads)
            .with_kv_cache_type(self.sampling.kv_cache_type);

        let context = model.new_context(&ctx_params)?;
        info!(
            "Context created: n_ctx={}, n_batch={}, n_ubatch={}",
            context.n_ctx(),
            context.n_batch(),
            context.n_ubatch(),
        );

        Ok(LlamaRunner {
            model,
            context: Arc::new(context),
            sampling: self.sampling,
        })
    }
}

/// High-level llama.cpp runner. Wraps model loading, context management, and inference.
pub struct LlamaRunner {
    model: LlamaModel,
    context: Arc<LlamaContext>,
    sampling: SamplingConfig,
}

impl LlamaRunner {
    /// Create a new builder for constructing a runner.
    pub fn builder(model_path: impl AsRef<Path>) -> LlamaRunnerBuilder {
        LlamaRunnerBuilder::new(model_path)
    }

    /// Get model information.
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            n_params: self.model.n_params(),
            n_embd: self.model.n_embd(),
            n_layer: self.model.n_layer(),
            n_head: self.model.n_head(),
            n_head_kv: self.model.n_head_kv(),
            n_ctx_train: self.model.n_ctx_train(),
            n_vocab: self.model.n_vocab(),
            rope_type: format!("{:?}", self.model.rope_type()),
            is_hybrid: self.model.is_hybrid(),
            is_recurrent: self.model.is_recurrent(),
            vocab_type: format!("{:?}", self.model.vocab_type()),
        }
    }

    /// Get the model's BOS, EOS, and NL token IDs.
    pub fn special_tokens(&self) -> (LlamaToken, LlamaToken, LlamaToken) {
        (
            self.model.token_bos(),
            self.model.token_eos(),
            self.model.token_nl(),
        )
    }

    /// Encode text to tokens using the model's built-in tokenizer.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
        let tokens = self.model.tokens(text, add_bos.into())?;
        Ok(tokens)
    }

    /// Decode a single token to a string piece.
    pub fn token_to_piece(&self, token: LlamaToken) -> Result<String> {
        let piece = self.model.token_to_piece(token, true)?;
        Ok(piece)
    }

    /// Decode a token to string.
    pub fn token_to_str(&self, token: LlamaToken) -> Result<String> {
        let s = self.model.token_to_str(token)?;
        Ok(s)
    }

    /// Apply a chat template to a list of messages.
    pub fn apply_chat_template(
        &self,
        messages: &[LlamaChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let result = self.model.apply_chat_template(messages, add_generation_prompt)?;
        Ok(result.text)
    }

    /// Run inference on a batch of tokens.
    /// Returns the logits for the last token in each sequence.
    pub fn decode(&self, batch: &llama_batch::LlamaBatch) -> Result<()> {
        self.context.decode(batch)?;
        Ok(())
    }

    /// Get logits for the entire batch (for classification / embeddings).
    pub fn get_logits(&self) -> Result<Vec<f32>> {
        let logits = self.context.get_logits();
        Ok(logits)
    }

    /// Get logits for a specific sequence.
    pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>> {
        let logits = self.context.get_logits_ith(i)?;
        Ok(logits)
    }

    /// Run a full generation loop: encode prompt, prefill, then decode token by token.
    pub fn generate(&self, prompt: &str, config: &SamplingConfig) -> Result<GenerationResult> {
        // Build sampler
        let mut sampler = self.build_sampler(config);

        // Encode prompt
        let prompt_tokens = self.encode(prompt, true)?;
        let prompt_len = prompt_tokens.len();

        info!("Prompt: {} tokens", prompt_len);

        // Create batch
        let mut batch = llama_batch::new(prompt_len as i32, 0, 1);

        // Add prompt tokens to batch
        for (i, tok) in prompt_tokens.iter().enumerate() {
            batch.add1(*tok, i as i32, &[0], false)?;
        }

        // Prefill
        let t_start = std::time::Instant::now();
        self.decode(&batch)?;
        let prompt_time = t_start.elapsed().as_secs_f64() * 1000.0;

        // Sample first token
        let mut tokens: Vec<LlamaToken> = vec![];
        let mut logits = self.context.get_logits_ith((prompt_len - 1) as i32)?;
        let token = sampler.accept(&mut logits, self.context.token_data_array(prompt_len - 1))?;
        tokens.push(token);

        info!("First token sampled: {:?}", token);

        // Decode loop
        let t_gen_start = std::time::Instant::now();
        let mut gen_count = 0;

        for pos in prompt_len..(prompt_len + config.max_tokens as usize) {
            // Clear KV cache for new position
            self.context.clear_kv_cache_seq_all(0, 0)?;

            // Create new batch for single token
            let mut new_batch = llama_batch::new(1, pos as i32, 1);
            new_batch.add1(token, pos as i32, &[0], false)?;

            // Decode
            self.decode(&new_batch)?;

            // Sample next token
            logits = self.context.get_logits_ith(pos as i32)?;
            token = sampler.accept(&mut logits, self.context.token_data_array(pos))?;
            tokens.push(token);
            gen_count += 1;

            // Check for EOS
            if self.model.is_eog_token(token) {
                info!("EOS token reached at position {}, generated {} tokens", pos + 1, gen_count);
                break;
            }
        }

        let gen_time = t_gen_start.elapsed().as_secs_f64() * 1000.0;

        // Decode tokens to text
        let text: String = tokens
            .iter()
            .filter_map(|t| self.model.token_to_str(*t).ok())
            .collect();

        // Get timings
        let timings = self.context.timings();

        info!(
            "Generation complete: {} tokens in {:.2}ms ({:.2} tok/s)",
            gen_count,
            gen_time,
            if gen_time > 0.0 { gen_count as f64 / gen_time * 1000.0 } else { 0.0 }
        );

        Ok(GenerationResult {
            text,
            tokens,
            prompt_tokens: prompt_len,
            generated_tokens: gen_count,
            load_time_ms: timings.load_ms,
            prompt_eval_ms: prompt_time,
            eval_ms: gen_time,
        })
    }

    /// Run generation with a chat template.
    pub fn generate_chat(
        &self,
        messages: &[LlamaChatMessage],
        config: &SamplingConfig,
    ) -> Result<GenerationResult> {
        let prompt = self.apply_chat_template(messages, true)?;
        self.generate(&prompt, config)
    }

    /// Compute embeddings for a prompt.
    pub fn embeddings(&self, prompt: &str) -> Result<Vec<f32>> {
        let tokens = self.encode(prompt, true)?;
        let mut batch = llama_batch::new(tokens.len() as i32, 0, 1);

        for (i, tok) in tokens.iter().enumerate() {
            batch.add1(*tok, i as i32, &[0], false)?;
        }

        self.decode(&batch)?;

        // Get embeddings from the last token of the sequence
        let emb = self.context.embeddings_ith(tokens.len() - 1)?;
        Ok(emb)
    }

    /// Clear the KV cache.
    pub fn clear_kv_cache(&self) -> Result<()> {
        self.context.clear_kv_cache()?;
        Ok(())
    }

    /// Reset the model's timings.
    pub fn reset_timings(&self) {
        self.context.reset_timings();
    }

    /// Print memory breakdown.
    pub fn print_memory_breakdown(&self) {
        self.context.print_memory_breakdown();
    }

    /// Get the context's KV cache size.
    pub fn n_ctx(&self) -> u32 {
        self.context.n_ctx()
    }

    /// Get the context's batch size.
    pub fn n_batch(&self) -> u32 {
        self.context.n_batch()
    }

    /// Get the context's ubatch size.
    pub fn n_ubatch(&self) -> u32 {
        self.context.n_ubatch()
    }

    /// Build a sampler from the sampling config.
    fn build_sampler(&self, config: &SamplingConfig) -> LlamaSampler {
        let mut sampler = LlamaSampler::new();

        // Temperature
        if config.temperature > 0.0 {
            sampler = sampler.chain(LlamaSamplerTemperature::new(config.temperature));
        }

        // Top-k
        if config.top_k > 0 {
            sampler = sampler.chain(LlamaSamplerTopK::new(config.top_k as i32, -100.0));
        }

        // Top-p
        if config.top_p > 0.0 {
            sampler = sampler.chain(LlamaSamplerTopP::new(config.top_p, -100.0));
        }

        // Min-p
        if config.min_p > 0.0 {
            sampler = sampler.chain(LlamaSamplerMinP::new(config.min_p, 256));
        }

        // TFS (tail free sampling)
        if config.tfs > 0.0 {
            sampler = sampler.chain(LlamaSamplerTailFree::new(config.tfs, 256));
        }

        // Typical p
        if config.typical_p > 0.0 {
            sampler = sampler.chain(LlamaSamplerTypical::new(config.typical_p, 256));
        }

        // MIROSTAT
        if config.mirostat {
            sampler = sampler.chain(LlamaSamplerMirostat::new(
                2,
                (config.mirostat_tau as f32).into(),
                (config.mirostat_tau as f32).into(),
                0.01f32.into(),
            ));
        }

        // Repetition penalty
        if config.repetition_penalty != 1.0 {
            sampler = sampler.chain(LlamaSamplerRepeatPenalty::new(
                config.repetition_penalty,
                config.repeat_last_n as i32,
            ));
        }

        // Penalties
        if config.penalty_alpha > 0.0 {
            sampler = sampler.chain(LlamaSamplerLoopyPenalty::new(
                config.penalty_alpha,
                config.penalty_beta,
                config.repeat_last_n as i32,
            ));
        }

        // Finalize
        sampler = sampler.finalize();

        sampler
    }

    /// Get a session manager for this runner.
    pub fn session_manager(&self) -> SessionManager {
        SessionManager::new(self.context.clone())
    }
}

// ── GGUF inspection (uses your gguf-parser, not llama-cpp-2's GGUF) ──

/// Inspect a GGUF file using the GGUF reader.
pub fn inspect_gguf(path: &str) -> Result<GgufReader> {
    let reader = GgufReader::from_path(path)?;
    Ok(reader)
}
