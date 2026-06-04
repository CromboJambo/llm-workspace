use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(String),

    #[error("invalid magic number: {0}")]
    InvalidMagic(String),

    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("invalid value type: {0}")]
    InvalidValueType(u32),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("binary read error: {0}")]
    Binary(#[from] std::io::Error),

    #[error("invalid tensor data: {0}")]
    InvalidTensor(String),

    #[error("quantization not supported: {0}")]
    QuantizationNotSupported(String),
}
