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
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use crabjar_llm_runner::llama::{LlamaRunner, SamplingConfig};
//!
//! let runner = LlamaRunner::builder("/path/to/model.gguf")
//!     .n_ctx(4096)
//!     .n_batch(512)
//!     .build()?;
//!
//! let result = runner.generate(
//!     "Explain quantum computing in one sentence.",
//!     &SamplingConfig::balanced(),
//! )?;
//! println!("{}", result.text);
//! ```

pub mod sampler;
pub mod model;
pub mod context;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use llama_cpp_2::{
    llama_backend,
    model::{LlamaModel, LlamaModelParams},
    context::{LlamaContext, params::LlamaContextParams},
    llama_batch::LlamaBatch,
    token::LlamaToken,
    sampling::LlamaSampler,
    json_schema_to_grammar,
};

use anyhow::Result;
use tracing::{info, warn};

pub use sampler::SamplingConfig;
pub use model::ModelInfo;
pub use context::ContextConfig;

// ── Re-exported types from llama-cpp-2 that users may need ──

pub use llama_cpp_2::model::LlamaChatMessage;
pub use llama_cpp_2::context::session::LlamaStateSeqFlags;
pub use llama_cpp_2::LlamaBackendDevice;

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
    /// Load time in milliseconds.
    pub load_time_ms: f64,
    /// Prompt evaluation time in milliseconds.
    pub prompt_eval_ms: f64,
    /// Token evaluation time in milliseconds.
    pub eval_ms: f64,
}

/// Builder for [`LlamaRunner`].
pub struct LlamaRunnerBuilder {
    model_path: String,
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_threads: i32,
    seed: u32,
    kv_cache_type: KvCacheType,
    n_gpu_layers: i32,
}

impl LlamaRunnerBuilder {
    pub fn new(model_path: impl AsRef<std::path::Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_string_lossy().to_string(),
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 512,
            n_threads: 0,
            seed: 42,
            kv_cache_type: KvCacheType::F32,
            n_gpu_layers: -1,
        }
    }

    /// Set the context size (number of tokens in KV cache).
    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Set the batch size for token processing.
    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    /// Set the ubatch size (micro-batch for compute).
    pub fn n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.n_ubatch = n_ubatch;
        self
    }

    /// Set the number of threads to use. 0 = auto.
    pub fn n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Set the KV cache type.
    pub fn kv_cache_type(mut self, kv_cache_type: KvCacheType) -> Self {
        self.kv_cache_type = kv_cache_type;
        self
    }

    /// Set the number of layers to offload to GPU.
    /// -1 = auto, 0 = CPU only, positive = specific layer count.
    pub fn n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Build the runner. This loads the model and creates the context.
    pub fn build(self) -> Result<LlamaRunner> {
        // Initialize the llama.cpp backend (CPU/GPU)
        let backend = llama_backend::LlamaBackend::init()?;
        info!("llama.cpp backend initialized");

        // Load the model
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(self.n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &model_params, &self.model_path)?;
        info!(
            "Loaded model: {} params={} n_embd={} n_layer={} n_head={} n_head_kv={} n_ctx_train={}",
            self.model_path,
            model.n_params(),
            model.n_embd(),
            model.n_layer(),
            model.n_head(),
            model.n_head_kv(),
            model.n_ctx_train(),
        );

        // Create the context
        let ctx_params = LlamaContextParams::default()
            .with_seed(self.seed)
            .with_n_ctx(self.n_ctx)
            .with_n_batch(self.n_batch)
            .with_n_ubatch(self.n_ubatch)
            .with_n_threads(self.n_threads)
            .with_kv_cache_type(self.kv_cache_type.into());

        let context = model.new_context(ctx_params)?;
        info!(
            "Context created: n_ctx={} n_batch={} n_ubatch={}",
            context.n_ctx(),
            context.n_batch(),
            context.n_ubatch(),
        );

        Ok(LlamaRunner {
            model,
            context: Rc::new(RefCell::new(context)),
        })
    }
}

/// High-level llama.cpp runner. Wraps model loading, context management, and inference.
pub struct LlamaRunner {
    model: LlamaModel,
    context: Rc<RefCell<LlamaContext<'static>>>,
}

impl LlamaRunner {
    /// Create a new builder for constructing a runner.
    pub fn builder(model_path: impl AsRef<std::path::Path>) -> LlamaRunnerBuilder {
        LlamaRunnerBuilder::new(model_path)
    }

    /// Get model information.
    pub fn model_info(&self) -> ModelInfo {
        let rope = self.model.rope_type();
        ModelInfo {
            n_params: self.model.n_params(),
            n_embd: self.model.n_embd(),
            n_layer: self.model.n_layer(),
            n_head: self.model.n_head(),
            n_head_kv: self.model.n_head_kv(),
            n_ctx_train: self.model.n_ctx_train(),
            n_vocab: self.model.n_vocab(),
            rope_type: format!("{:?}", rope),
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
    pub fn decode(&self, batch: &LlamaBatch) -> Result<()> {
        let mut ctx = self.context.borrow_mut();
        ctx.decode(batch)?;
        Ok(())
    }

    /// Get logits for the entire batch.
    pub fn get_logits(&self) -> Vec<f32> {
        self.context.borrow().get_logits().to_vec()
    }

    /// Get logits for a specific sequence.
    pub fn get_logits_ith(&self, i: i32) -> Result<Vec<f32>> {
        let logits = self.context.borrow().get_logits_ith(i);
        Ok(logits.to_vec())
    }

    /// Run a full generation loop: encode prompt, prefill, then decode token by token.
    pub fn generate(&self, prompt: &str, config: &SamplingConfig) -> Result<GenerationResult> {
        // Build sampler
        let mut sampler = self.build_sampler(config);

        // Encode prompt
        let prompt_tokens = self.encode(prompt, true)?;
        let prompt_len = prompt_tokens.len();

        info!("Prompt: {} tokens", prompt_len);

        // Create batch for prompt
        let mut batch = LlamaBatch::new(prompt_len as i32, 0, 1);
        for (i, tok) in prompt_tokens.iter().enumerate() {
            batch.add(*tok, i as i32, &[0], true)?;
        }

        // Prefill
        let t_start = Instant::now();
        self.decode(&batch)?;
        let prompt_time = t_start.elapsed().as_secs_f64() * 1000.0;

        // Sample first token
        let mut tokens: Vec<LlamaToken> = vec![];
        let mut ctx = self.context.borrow_mut();
        let logits = ctx.get_logits_ith((prompt_len - 1) as i32);
        let token = sampler.sample(&ctx, (prompt_len - 1) as i32);
        tokens.push(token);

        info!("First token sampled: {:?}", token);

        // Decode loop
        let t_gen_start = Instant::now();
        let mut gen_count = 0;

        for pos in prompt_len..(prompt_len + config.max_tokens as usize) {
            // Create new batch for single token
            let mut new_batch = LlamaBatch::new(1, pos as i32, 1);
            new_batch.add(token, pos as i32, &[0], false)?;

            // Decode
            self.decode(&new_batch)?;

            // Sample next token
            let logits = self.get_logits_ith(pos as i32)?;
            let ctx = self.context.borrow();
            let next_token = sampler.sample(&ctx, pos as i32);
            drop(ctx);

            token = next_token;
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
        let mut ctx = self.context.borrow_mut();
        let timings = ctx.timings();
        drop(ctx);

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
        let mut batch = LlamaBatch::new(tokens.len() as i32, 0, 1);

        for (i, tok) in tokens.iter().enumerate() {
            batch.add(*tok, i as i32, &[0], false)?;
        }

        self.decode(&batch)?;

        // Get embeddings from the last token of the sequence
        let emb = self.context.borrow().embeddings_ith(tokens.len() - 1)?;
        Ok(emb)
    }

    /// Clear the KV cache.
    pub fn clear_kv_cache(&self) -> Result<()> {
        let mut ctx = self.context.borrow_mut();
        ctx.clear_kv_cache()?;
        Ok(())
    }

    /// Reset the model's timings.
    pub fn reset_timings(&self) {
        let mut ctx = self.context.borrow_mut();
        ctx.reset_timings();
    }

    /// Print memory breakdown.
    pub fn print_memory_breakdown(&self) {
        self.context.borrow().print_memory_breakdown();
    }

    /// Get the context's KV cache size.
    pub fn n_ctx(&self) -> u32 {
        self.context.borrow().n_ctx()
    }

    /// Get the context's batch size.
    pub fn n_batch(&self) -> u32 {
        self.context.borrow().n_batch()
    }

    /// Get the context's ubatch size.
    pub fn n_ubatch(&self) -> u32 {
        self.context.borrow().n_ubatch()
    }

    /// Get the context's session manager.
    pub fn session_manager(&self) -> SessionManager {
        SessionManager::new(self.context.clone())
    }

    /// Convert a JSON schema to a grammar string for constrained decoding.
    pub fn schema_to_grammar(&self, schema: &str) -> Result<String> {
        let grammar = json_schema_to_grammar(&self.model, schema)?;
        Ok(grammar)
    }

    /// Build a sampler from the sampling config.
    fn build_sampler(&self, config: &SamplingConfig) -> LlamaSampler {
        let mut sampler = LlamaSampler::chain_simple([]);

        // Temperature
        if config.temperature > 0.0 {
            sampler = sampler.temp(config.temperature as f32);
        }

        // Top-k
        if config.top_k > 0 {
            sampler = sampler.top_k(config.top_k);
        }

        // Top-p
        if config.top_p > 0.0 {
            sampler = sampler.top_p(config.top_p as f32, -100.0);
        }

        // Min-p
        if config.min_p > 0.0 {
            sampler = sampler.min_p(config.min_p as f32, 256);
        }

        // TFS (tail free sampling)
        if config.tfs > 0.0 && config.tfs < 1.0 {
            sampler = sampler.top_n_sigma(config.tfs as f32);
        }

        // Typical p
        if config.typical_p > 0.0 && config.typical_p < 1.0 {
            sampler = sampler.typical(config.typical_p as f32);
        }

        // Repetition penalty
        if config.repetition_penalty != 1.0 {
            sampler = sampler.penalties(
                config.repetition_penalty as f32,
                config.repeat_last_n,
                0.0,
            );
        }

        sampler
    }
}

/// KV cache type.
#[derive(Debug, Clone, Copy, Default)]
pub enum KvCacheType {
    #[default]
    F32,
    F16,
}

impl From<KvCacheType> for llama_cpp_2::context::params::KvCacheType {
    fn from(value: KvCacheType) -> Self {
        match value {
            KvCacheType::F32 => Self::F32,
            KvCacheType::F16 => Self::F16,
        }
    }
}

/// Manages session save/load.
pub struct SessionManager {
    context: Rc<RefCell<LlamaContext<'static>>>,
}

impl SessionManager {
    pub fn new(context: Rc<RefCell<LlamaContext<'static>>>) -> Self {
        Self { context }
    }

    /// Save the current context state to a file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.context.borrow().save_session_file(path)?;
        Ok(())
    }

    /// Load context state from a file.
    pub fn load(&self, path: &str) -> Result<(), String> {
        self.context.borrow().load_session_file(path)?;
        Ok(())
    }

    /// Check if a session file exists.
    pub fn session_exists(path: &str) -> bool {
        std::fs::metadata(path).is_ok()
    }

    /// Save a specific sequence's state to a file.
    pub fn save_seq_state(&self, seq_id: i32, path: &str) -> Result<(), String> {
        self.context.borrow().state_seq_save_file(seq_id, path)?;
        Ok(())
    }

    /// Load a specific sequence's state from a file.
    pub fn load_seq_state(&self, seq_id: i32, path: &str) -> Result<(), String> {
        self.context.borrow().state_seq_load_file(seq_id, path)?;
        Ok(())
    }

    /// Get the size needed to save the full state.
    pub fn state_size(&self) -> Result<usize, String> {
        let size = self.context.borrow().get_state_size();
        Ok(size)
    }

    /// Get the size needed to save a specific sequence's state.
    pub fn state_seq_size(&self, seq_id: i32) -> Result<usize, String> {
        let ctx = self.context.borrow();
        let size = ctx.state_seq_get_size_ext(&ctx, seq_id);
        Ok(size)
    }
}

/// Inspect a GGUF file using the GGUF reader.
///
/// Note: GgufReader::from_path was removed in llama-cpp-2 0.1.x.
/// Use llama_cpp_sys_2::gguf_init_from_file directly, or load the model
/// and query metadata via LlamaModel methods.
pub fn inspect_gguf(path: &str) -> Result<String> {
    // Use llama_cpp_sys_2 directly for GGUF inspection
    let mut n_tensors: u64 = 0;
    let mut n_kv: u64 = 0;
    let gguf = unsafe {
        llama_cpp_sys_2::gguf_init_from_file(
            std::ffi::CString::new(path).unwrap().as_ptr(),
            false,
            &mut n_tensors,
            &mut n_kv,
        )
    };

    if gguf.is_null() {
        return Err(anyhow::anyhow!("Failed to init GGUF reader for: {}", path));
    }

    let mut info = String::new();
    info.push_str(&format!("GGUF file: {}\n", path));
    info.push_str(&format!("  n_tensors: {}\n", n_tensors));
    info.push_str(&format!("  n_kv: {}\n", n_kv));

    // Read key KV metadata
    let keys = unsafe { llama_cpp_sys_2::gguf_kv_get_all(gguf) };
    if !keys.is_null() {
        let n_keys = unsafe { llama_cpp_sys_2::gguf_n_keys(gguf) };
        for i in 0..n_keys {
            let key_ptr = unsafe { llama_cpp_sys_2::gguf_key_get_name(keys, i as i32) };
            let key = unsafe { std::ffi::CStr::from_ptr(key_ptr) }.to_string_lossy().to_string();
            let val_type = unsafe { llama_cpp_sys_2::gguf_kv_get_type(gguf, key_ptr) };
            info.push_str(&format!("  KV[{}]: {} (type={})\n", i, key, val_type));
        }
    }

    unsafe { llama_cpp_sys_2::gguf_free(gguf) };

    Ok(info)
}
