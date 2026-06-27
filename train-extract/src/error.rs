use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrainExtractError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors error: {0}")]
    Safetensors(#[from] crabjar_safetensors::SafetensorsError),

    #[error("schema error: {0}")]
    Schema(#[from] crabjar_safetensors::SafetensorsSchemaError),

    #[error("empty dataset: no entries matched filters")]
    EmptyDataset,

    #[error("export failed: {0}")]
    Export(String),

    #[error("invalid path: {0}")]
    InvalidPath(String),
}

pub type TrainExtractResult<T> = Result<T, TrainExtractError>;
