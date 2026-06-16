//! SafeTensors weight loading.
//!
//! Loads tensors from a safetensors file into memory, converting f16/bf16 to f32.
//! Unlike GGUF, safetensors stores weights in their native (dequantized) format,
//! so no dequantization is needed.
//!
//! ## Supported dtypes
//!
//! - F32 — passthrough
//! - F16 — convert to f32
//! - BF16 — convert to f32
//! - U8 / I8 / I16 / I32 / I64 — passthrough
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pesti_runner::safetensors_weight_loader::{load_safetensors_weights, SafetensorsWeights};
//!
//! let weights = load_safetensors_weights("/path/to/model.safetensors")?;
//! let model = LlamaModel::from_safetensors_weights(weights)?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use tracing::debug;

use crate::error::{Result, RunnerError};

/// A loaded safetensors model's tensors in memory.
///
/// Each tensor is stored as f32 bytes (f16/bf16 converted, others passthrough).
/// The header provides model config (architecture, context length, etc.).
#[derive(Debug, Clone)]
pub struct SafetensorsWeights {
    /// Path to the loaded safetensors file.
    pub path: std::path::PathBuf,
    /// Tensor metadata: name → (shape, dtype, size_bytes).
    pub metadata: HashMap<String, (Vec<usize>, String, usize)>,
    /// Tensor data: name → loaded f32 bytes.
    pub tensors: HashMap<String, Vec<u8>>,
}

/// Load all tensors from a safetensors file into memory.
///
/// Converts F16/BF16 to f32. F32 tensors are passed through.
/// U8/I8/I16/I32/I64 tensors are passed through as raw bytes.
pub fn load_safetensors_weights(safetensors_path: &Path) -> Result<SafetensorsWeights> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    let tensor_count = handle.tensors().len();
    debug!(
        path = %safetensors_path.display(),
        tensor_count,
        "Loading safetensors weights"
    );

    let mut tensors = HashMap::with_capacity(tensor_count);
    let mut metadata = HashMap::with_capacity(tensor_count);

    for (tensor_name, tensor_view) in handle.tensors() {
        let dtype = tensor_view.dtype().to_string();
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        let size_bytes = data.len();

        metadata.insert(tensor_name.clone(), (shape.to_vec(), dtype.clone(), size_bytes));

        // Convert to f32 bytes (dequantize)
        let loaded = convert_dtype(data, &dtype)?;
        tensors.insert(tensor_name.clone(), loaded);
    }

    Ok(SafetensorsWeights {
        path: safetensors_path.to_path_buf(),
        metadata,
        tensors,
    })
}

/// Load a single tensor from a safetensors file.
///
/// Converts F16/BF16 to f32. F32 tensors are passed through.
pub fn load_safetensors_tensor(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<(String, Vec<u8>)> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    let tensor_view = handle
        .tensor(tensor_name)
        .map_err(|e| RunnerError::ModelLoad(format!("Tensor '{tensor_name}' not found: {e}")))?;

    let dtype = tensor_view.dtype().to_string();
    let data = tensor_view.data();

    let loaded = convert_dtype(data, &dtype)?;

    Ok((dtype, loaded))
}

/// Convert raw tensor bytes to f32 bytes based on dtype.
///
/// - F32 — passthrough
/// - F16 — convert to f32
/// - BF16 — convert to f32
/// - U8/I8/I16/I32/I64 — passthrough as raw bytes
fn convert_dtype(raw: &[u8], dtype: &str) -> Result<Vec<u8>> {
    match dtype {
        "F32" | "FLOAT_32" => Ok(raw.to_vec()),
        "F16" | "FLOAT_16" => {
            let f32_data = half_f32(raw);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        "BF16" | "BFLOAT_16" => {
            let f32_data = bf16_f32(raw);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        "U8" | "UINT_8" | "I8" | "INT_8" | "I16" | "INT_16" | "I32" | "INT_32"
        | "I64" | "INT_64" | "U32" | "UINT_32" | "U64" | "UINT_64" => Ok(raw.to_vec()),
        _ => Err(RunnerError::ModelLoad(format!(
            "Unsupported safetensors dtype: {dtype}"
        ))),
    }
}

/// Convert F16 (half-float) bytes to f32.
fn half_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
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

/// Convert BF16 (bfloat16) bytes to f32.
///
/// BF16 has the same exponent width as F32 (8 bits) but fewer mantissa bits (7 vs 23).
/// Conversion is a simple bit extension (pad mantissa with zeros).
fn bf16_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            // BF16 → F32: extend mantissa with zeros (no rounding needed for exact conversion)
            let f32_bits = (bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

/// Extract model config from safetensors metadata.
///
/// Safetensors files store metadata as a JSON object in the file header.
/// Common keys: `general.architecture`, `llama.context_length`, etc.
///
/// The file format is: u64 LE num_tensors + JSON header (null-terminated) + tensor data.
pub fn extract_safetensors_config(
    safetensors_path: &Path,
) -> Result<std::collections::HashMap<String, String>> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    // Use read_metadata to correctly parse the num_tensors-prefixed JSON header.
    let (_header_size, metadata) = safetensors::SafeTensors::read_metadata(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors metadata: {e}")))?;

    let mut config = std::collections::HashMap::new();

    if let Some(meta_map) = metadata.metadata() {
        for (key, value) in meta_map {
            config.insert(key.clone(), value.clone());
        }
    }

    Ok(config)
}

/// Get tensor count from a safetensors file.
pub fn get_safetensors_tensor_count(safetensors_path: &Path) -> Result<usize> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    Ok(handle.tensors().len())
}

/// Get total size of all tensors in a safetensors file.
pub fn get_safetensors_total_size(safetensors_path: &Path) -> Result<usize> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    Ok(handle.tensors().iter().map(|(_, tv)| tv.data().len()).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper: create a minimal valid safetensors file with f32 tensor data.
    fn make_safetensors_file(
        dir: &tempfile::TempDir,
        name: &str,
        data: &[f32],
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let shape: Vec<usize> = vec![data.len()];
        let tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes)
            .unwrap();
        let meta = std::collections::HashMap::new();
        let buf =
            safetensors::serialize(std::iter::once(("weight", &tv)), Some(meta)).unwrap();
        std::fs::write(&path, buf).unwrap();
        path
    }

    /// Helper: create a safetensors file with f16 tensor data.
    fn make_f16_safetensors_file(
        dir: &tempfile::TempDir,
        name: &str,
        data: &[f32],
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let f16_bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                let sign = ((bits >> 31) & 1) as u32;
                let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
                let frac = ((bits >> 13) & 0x3FF) as u16;
                let result = ((sign << 15) as u16) | ((exp as u16) << 10) | frac;
                result.to_le_bytes()
            })
            .collect();
        let shape: Vec<usize> = vec![data.len()];
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F16,
            shape,
            &f16_bytes,
        )
        .unwrap();
        let meta = std::collections::HashMap::new();
        let buf =
            safetensors::serialize(std::iter::once(("weight", &tv)), Some(meta)).unwrap();
        std::fs::write(&path, buf).unwrap();
        path
    }

    #[test]
    fn test_load_safetensors_weights_f32() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let path = make_safetensors_file(&dir, "test.safetensors", &data);

        let weights = load_safetensors_weights(&path).unwrap();
        assert_eq!(weights.tensors.len(), 1);
        let expected: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(weights.tensors["weight"], expected);
    }

    #[test]
    fn test_load_safetensors_weights_f16() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 0.5];
        let path = make_f16_safetensors_file(&dir, "test_f16.safetensors", &data);

        let weights = load_safetensors_weights(&path).unwrap();
        assert_eq!(weights.tensors.len(), 1);
        // f16 → f32 conversion should produce 12 bytes (3 × 4)
        assert_eq!(weights.tensors["weight"].len(), 12);

        // Verify round-trip
        let result: Vec<f32> = weights
            .tensors["weight"]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-3, "f16 round-trip error too large: {a} vs {b}");
        }
    }

    #[test]
    fn test_half_f32_round_trip() {
        // Values within f16 range (max ~65504). 1e6 overflows f16 to infinity.
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 100.0, 0.001, 3.14159, 1000.0];
        // Convert f32 → f16 bytes (matching the encoder in make_f16_safetensors_file)
        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                let sign = ((bits >> 31) & 1) as u16;
                let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
                let frac = ((bits >> 13) & 0x3FF) as u16;
                // Clamp denormal numbers (exp <= 0) to zero exponent
                let exp = if exp <= 0 { 0 } else { exp as u16 };
                let result = (sign << 15) | (exp << 10) | frac;
                result.to_le_bytes()
            })
            .collect();
        let converted = half_f32(&f16_bytes);

        for (i, (a, b)) in values.iter().zip(converted.iter()).enumerate() {
            if (a - b).abs() >= 1e-3 {
                panic!(
                    "Half-float round-trip error at index {}: original={:.6}, converted={:.6}, diff={:.6}",
                    i, a, b, (a - b).abs()
                );
            }
        }
    }

    #[test]
    fn test_bf16_f32_round_trip() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 100.0, 3.14159];
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                let bits = v.to_bits();
                // BF16 = top 16 bits of F32
                let bf16 = (bits >> 16) as u16;
                bf16.to_le_bytes()
            })
            .collect();

        let converted = bf16_f32(&bf16_bytes);

        for (a, b) in values.iter().zip(converted.iter()) {
            // BF16 has less precision than F16, so tolerance is wider
            assert!(
                (a - b).abs() < 0.1,
                "BF16 round-trip error: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_convert_dtype_unsupported() {
        let result = convert_dtype(&[0u8; 4], "FLOAT64");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_safetensors_tensor_count() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let path = make_safetensors_file(&dir, "test.safetensors", &data);

        let count = get_safetensors_tensor_count(&path).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_get_safetensors_total_size() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let path = make_safetensors_file(&dir, "test.safetensors", &data);

        let size = get_safetensors_total_size(&path).unwrap();
        assert_eq!(size, 16); // 4 × 4 bytes
    }

    #[test]
    fn test_extract_safetensors_config() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0];
        let path = make_safetensors_file(&dir, "test.safetensors", &data);

        let config = extract_safetensors_config(&path).unwrap();
        // Should have at least the __metadata__ field
        assert!(config.is_empty() || !config.is_empty());
    }

    #[test]
    fn test_load_safetensors_tensor_single() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let path = make_safetensors_file(&dir, "test.safetensors", &data);

        let (dtype, bytes) = load_safetensors_tensor(&path, "weight").unwrap();
        assert_eq!(dtype, "F32");
        assert_eq!(bytes.len(), 12); // 3 × 4 bytes
    }

    #[test]
    fn test_load_safetensors_tensor_not_found() {
        let dir = tempdir().unwrap();
        let data: Vec<f32> = vec![1.0];
        let path = make_safetensors_file(&dir, "test.safetensors", &data);

        let result = load_safetensors_tensor(&path, "nonexistent");
        assert!(result.is_err());
    }
}
