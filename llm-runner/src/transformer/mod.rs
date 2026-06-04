//! Transformer module: Llama-style model implementation for CPU inference.
//!
//! ## Components
//!
//! - `model` — `LlamaModel` loads GGUF weights and wires transformer layers
//! - `layer` — `TransformerLayer` with attention, FFN, RMSNorm
//! - `linear` — `Linear` layer for matrix multiplication
//! - `rms_norm` — RMS normalization
//! - `rope` — Rotary positional embeddings
//! - `sampling` — Token sampling (temperature, top-p, top-k)
//! - `tokenizer` — GGUF tokenizer integration
//!
//! ## Inference Flow
//!
//! ```text
//! GGUF file → LlamaModel::load_gguf() → forward(token, pos) → logits → sample() → next_token
//! ```

pub mod layer;
pub mod linear;
pub mod model;
pub mod rope;
pub mod rms_norm;
pub mod sampling;
pub mod tokenizer;

pub use model::LlamaModel;
pub use sampling::{sample, argmax, SamplingConfig};
pub use tokenizer::{load_tokenizer_from_gguf, GgufTokenizerConfig};
