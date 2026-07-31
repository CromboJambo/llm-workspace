pub mod error;
pub mod types;
pub mod parser;

#[cfg(test)]
mod tests {
    mod defensive_tests;
    mod gguf_v3_conformance;
}

pub use error::GgufError;
pub use types::*;
pub use parser::{compute_data_section_start, extract_tensor_bytes, extract_tensor_bytes_from, extract_tensor_bytes_from_path, parse_gguf, parse_gguf_reader, tensor_bytes_for_dtype};


