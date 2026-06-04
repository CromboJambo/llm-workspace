use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("tensor computation error: {0}")]
    Tensor(String),

    #[error("model loading error: {0}")]
    ModelLoad(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("device backend error: {0}")]
    Device(String),

    #[error("plug-in protocol error: {0}")]
    Protocol(#[from] crabjar_llm_plug_in::PlugInError),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("GGUF parse error: {0}")]
    Gguf(#[from] crabjar_gguf::GgufError),

    #[error("asset loading error: {0}")]
    Asset(String),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, RunnerError>;
