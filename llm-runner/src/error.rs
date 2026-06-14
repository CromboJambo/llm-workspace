use crate::cuda_runtime::CudaError;
use crate::kernel::{AttentionError, GemmError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("GEMM error ({arch}, {m}x{n}x{k}): {detail}")]
    Gemm {
        arch: String,
        m: usize,
        n: usize,
        k: usize,
        #[source]
        detail: GemmError,
    },

    #[error("attention error (heads={num_heads}, dim={head_dim}, seq={seq}): {detail}")]
    Attention {
        num_heads: usize,
        head_dim: usize,
        seq: usize,
        #[source]
        detail: AttentionError,
    },

    #[error("tensor computation error: {0}")]
    Tensor(String),

    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    #[error("model loading error: {0}")]
    ModelLoad(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("device backend error: {0}")]
    Device(String),

    #[error("plug-in protocol error: {0}")]
    Protocol(#[from] pesti_plug_in::PlugInError),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("GGUF parse error: {0}")]
    Gguf(#[from] pesti_gguf::GgufError),

    #[error("asset loading error: {0}")]
    Asset(String),

    #[error("unspecified internal error: {0}")]
    Internal(String),

    #[error("dequantization error for tensor '{0}': {1}")]
    Dequant(String, String),

    #[error("GGUF header missing required field: {0}")]
    MissingHeaderField(String),
}

pub type Result<T> = std::result::Result<T, RunnerError>;
