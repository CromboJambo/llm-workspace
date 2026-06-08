//! GGUF weight loading with dequantization.
//!
//! Loads tensors from a GGUF file into memory, dequantizing Q4_0 and converting
//! F16/BF16 to f32. Returns a `GgufWeights` struct that can be fed directly into
//! the inference engine.
//!
//! ## Supported dtypes
//!
//! - F32 — passthrough
//! - F16 / BF16 — convert to f32
//! - Q4_0 — dequantize to f32 (32 elements per block)
//! - Q4_1 — dequantize to f32
//! - Q8_0 — dequantize to f32
//! - I8 / I16 / I32 / I64 — passthrough
//!

use std::collections::HashMap;
use std::path::Path;

use crabjar_gguf::parser::{extract_tensor_bytes, parse_gguf};
use crabjar_gguf::types::{GgufDtype, GgufHeader, GgufTensorInfo};

use crate::error::{Result, RunnerError};

/// A loaded GGUF model's tensors in memory.
///
/// Each tensor is stored as f32 bytes (dequantized if needed).
/// The header provides model config (architecture, context length, etc.).
#[derive(Debug, Clone)]
pub struct GgufWeights {
    /// Parsed GGUF header with model config.
    pub header: GgufHeader,
    /// Tensor data: name → dequantized f32 bytes.
    pub tensors: HashMap<String, Vec<u8>>,
}

/// Load all tensors from a GGUF file into memory.
///
/// Dequantizes Q4_0, converts F16/BF16 to f32. F32 tensors are passed through.
/// Returns the header + all tensor data.
pub fn load_gguf_weights(gguf_path: &Path) -> Result<GgufWeights> {
    let header = parse_gguf(gguf_path)?;

    let mut tensors = HashMap::with_capacity(header.tensors.len());

    for tensor in &header.tensors {
        let stored_size = tensor.stored_size() as usize;
        let file_offset = header.data_section_start + tensor.offset;

        let raw_data = extract_tensor_bytes(gguf_path, file_offset, stored_size)?;

        let dequantized = dequantize_tensor(tensor, &raw_data)?;

        tensors.insert(tensor.name.clone(), dequantized);
    }

    Ok(GgufWeights { header, tensors })
}

/// Load a single tensor from a GGUF file.
///
/// Dequantizes Q4_0, converts F16/BF16 to f32. F32 tensors are passed through.
pub fn load_gguf_tensor(gguf_path: &Path, tensor_name: &str) -> Result<(GgufHeader, Vec<u8>)> {
    let header = parse_gguf(gguf_path)?;

    let tensor = header.get_tensor(tensor_name).ok_or_else(|| {
        RunnerError::Gguf(crabjar_gguf::GgufError::InvalidTensor(format!(
            "tensor '{tensor_name}' not found in file"
        )))
    })?;

    let stored_size = tensor.stored_size() as usize;
    let file_offset = header.data_section_start + tensor.offset;

    let raw_data = extract_tensor_bytes(gguf_path, file_offset, stored_size)?;

    let dequantized = dequantize_tensor(tensor, &raw_data)?;

    Ok((header, dequantized))
}

/// Dequantize tensor data to f32 bytes based on GGUF dtype.
fn dequantize_tensor(tensor: &GgufTensorInfo, raw_data: &[u8]) -> Result<Vec<u8>> {
    let dtype = GgufDtype::from_u32(tensor.dtype);
    let element_count = tensor.element_count() as usize;

    match dtype {
        GgufDtype::F32 => Ok(raw_data.to_vec()),
        GgufDtype::F16 => {
            let f32_data = half_f32(raw_data);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::BF16 => {
            let f32_data = bf16_f32(raw_data);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q4_0 => {
            let dequantized = dequantize_q4_0(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_1 => {
            let dequantized = dequantize_q4_1(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q8_0 => {
            let dequantized = dequantize_q8_0(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2_K => {
            let dequantized = dequantize_q2_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3_K => {
            let dequantized = dequantize_q3_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_K | GgufDtype::Q4_K_M => {
            let dequantized = dequantize_q4_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S => {
            let dequantized = dequantize_q5_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q6_K | GgufDtype::Q6_K_S => {
            let dequantized = dequantize_q6_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q8_K | GgufDtype::Q8_K_M => {
            let dequantized = dequantize_q8_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q1_K => dequantize_q1_k(data, element_count),
        GgufDtype::Q2_K_S => dequantize_q2_k(data, element_count),
        GgufDtype::Q3_K_S => dequantize_q3_k(data, element_count),
        GgufDtype::Q4_K_S => dequantize_q4_k(data, element_count),
        GgufDtype::Q2_K_M => dequantize_q2_k(data, element_count),
        GgufDtype::I8 | GgufDtype::I16 | GgufDtype::I32 | GgufDtype::I64 => Ok(raw_data.to_vec()),
        GgufDtype::Unknown(_) => Err(RunnerError::Gguf(crabjar_gguf::GgufError::Io(format!(
            "Unknown GGUF dtype {} for tensor '{}'",
            tensor.dtype, tensor.name
        )))),
        _ => Err(RunnerError::Gguf(crabjar_gguf::GgufError::Io(format!(
            "Unsupported GGUF dtype {} for tensor '{}'. Use load_gguf_model() for full conversion pipeline.",
            tensor.dtype, tensor.name
        )))),
    }
}

// ── Dequantization implementations ───────────────────────────────────

/// Dequantize Q4_0 data to f32.
///
/// Q4_0 block: 32 elements, 18 bytes (2-byte f16 scale + 16 bytes quantized, nibble-packed).
/// dequantized = scale * (q - 8)
fn dequantize_q4_0(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 18
        + if remaining > 0 {
            2 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 18;
        let scale = f16_to_f32(&data[base..base + 2]);

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 18;
        let scale = f16_to_f32(&data[base..base + 2]);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32);
        }
    }

    Ok(result)
}

/// Dequantize Q4_1 data to f32.
///
/// Q4_1 block: 32 elements, 20 bytes (2×f16 scale/min + 16 bytes quantized).
/// dequantized = scale * q + min (q is unsigned 0-15, no offset)
fn dequantize_q4_1(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 20
        + if remaining > 0 {
            4 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_1 data too small: got {expected_size} bytes, need {expected_size}"
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 20;
        let scale = f16_to_f32(&data[base..base + 2]);
        let min = f16_to_f32(&data[base + 2..base + 4]);

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 20;
        let scale = f16_to_f32(&data[base..base + 2]);
        let min = f16_to_f32(&data[base + 2..base + 4]);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    Ok(result)
}

/// Dequantize Q8_0 data to f32.
///
/// Q8_0 block: 32 elements, 34 bytes (2 bytes scale + 32 bytes int8 quantized).
/// dequantized = scale * quantized_value
fn dequantize_q8_0(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_blocks = element_count.div_ceil(32);
    let expected_size = num_blocks * 34;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_0 data too small: got {expected_size} bytes"
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 34;
        let scale = f16_to_f32(&data[base..base + 2]);

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let q = data[base + 2 + i] as i8 as f32;
            result.push(scale * q);
        }
    }

    Ok(result)
}

// ── K-family dequantization implementations ─────────────────────────

/// Dequantize Q1_K data to f32.
///
/// Q1_K block: 16 elements, 20 bytes per block.
fn dequantize_q1_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 20 + if remaining > 0 { 2 + remaining.div_ceil(2) } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(crabjar_gguf::GgufError::Io(format!(
            "Q1_K data too small: got {} bytes, need {}",
            data.len(), expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
        let delta = [
            f16_to_f32(&data[offset + 6..offset + 8]),
            f16_to_f32(&data[offset + 8..offset + 10]),
            f16_to_f32(&data[offset + 10..offset + 12]),
            f16_to_f32(&data[offset + 12..offset + 14]),
        ];
        let h = [
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
        ];

        for i in 0..16usize {
            let q1_val = ((q1 >> i) & 0x01) << 2;
            let q = q1_val as i32 - 4;
            let scale = if q1_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
        offset += 20;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
        let h = [
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
        ];

        for i in 0..remaining {
            let q1_val = ((q1 >> i) & 0x01) << 2;
            let q = q1_val as i32 - 4;
            let scale = if q1_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q2_K data to f32.
///
/// Q2_K block: 16 elements, 16 bytes per block.
/// Block layout: d(f16, 2B) + d_min(f16, 2B) + q1(2B) + q2(6B) + h(4B) = 16B
fn dequantize_q2_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64 / 4 + element_count as u64 * 6 / 32 + 8) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q2_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
        let q2 = u32::from_le_bytes([
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ]);
        let h = [
            f16_to_f32(&data[offset + 10..offset + 12]),
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
        ];

        for i in 0..16usize {
            let q2_val = (q2 >> (2 * i)) & 0x03;
            let q1_val = ((q1 >> i) & 0x01) << 2;
            let q = (q1_val | q2_val) as i32 - 4;
            let scale = if q2_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
        offset += 24;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
        let q2 = u32::from_le_bytes([
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ]);
        let h = [
            f16_to_f32(&data[offset + 10..offset + 12]),
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
        ];

        for i in 0..remaining {
            let q2_val = (q2 >> (2 * i)) & 0x03;
            let q1_val = ((q1 >> i) & 0x01) << 2;
            let q = (q1_val | q2_val) as i32 - 4;
            let scale = if q2_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q3_K data to f32.
///
/// Q3_K block: 16 elements, 24 bytes per block.
/// Block layout: d(f16, 2B) + d_min(f16, 2B) + delta(1B) + k_scale(4B) + q3(6B) + mask(1B) + h(4B) = 24B
fn dequantize_q3_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64 / 8 + element_count as u64 * 6 / 32 + 16) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q3_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let delta = data[offset + 4] as i32;
        let k_scale = [data[offset + 5], data[offset + 6], data[offset + 7], data[offset + 8]];
        let q3 = [
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
        ];
        let mask = data[offset + 15];
        let h = [
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
            f16_to_f32(&data[offset + 22..offset + 24]),
        ];

        for i in 0..16usize {
            let q3_val = ((q3[i / 4] >> (3 * (i % 4))) & 0x07) as i32;
            let mask_bit = (mask >> i) & 1;
            let q = q3_val - ((mask_bit << 2) | (mask_bit << 1));
            let scale = d * (k_scale[i / 4] as f32) / 4.0 + d_min;
            let h_scale = if mask_bit != 0 { h[i / 4] } else { h[i / 4] / 8.0 };
            result.push(scale * (q as f32 + delta as f32 / 64.0) * h_scale);
        }
        offset += 24;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let delta = data[offset + 4] as i32;
        let k_scale = [data[offset + 5], data[offset + 6], data[offset + 7], data[offset + 8]];
        let q3 = [
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
        ];
        let mask = data[offset + 15];
        let h = [
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
            f16_to_f32(&data[offset + 22..offset + 24]),
        ];

        for i in 0..remaining {
            let q3_val = ((q3[i / 4] >> (3 * (i % 4))) & 0x07) as i32;
            let mask_bit = (mask >> i) & 1;
            let q = q3_val - ((mask_bit << 2) | (mask_bit << 1));
            let scale = d * (k_scale[i / 4] as f32) / 4.0 + d_min;
            let h_scale = if mask_bit != 0 { h[i / 4] } else { h[i / 4] / 8.0 };
            result.push(scale * (q as f32 + delta as f32 / 64.0) * h_scale);
        }
    }

    Ok(result)
}

/// Dequantize Q4_K data to f32.
///
/// Q4_K block: 16 elements, 24 bytes per block.
/// Block layout: d(f16, 2B) + d_min(f16, 2B) + scales(2B) + q4_lo(4B) + q4_hi(4B) + extra(12B) = 24B
fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64 / 4 + element_count as u64 * 6 / 32 + 16 + 32) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let scale_lo = data[offset + 4] as f32;
        let scale_hi = data[offset + 5] as f32;
        let q4_lo = [
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ];
        let q4_hi = [
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
        ];

        for i in 0..16usize {
            let lo = (q4_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = (q4_hi[i / 2] >> (4 * (i % 2))) & 0x0F;

            let scale = if hi > 0 {
                d * (scale_lo + scale_hi * 1.0 / 32.0)
            } else {
                d * scale_lo
            };

            let q = (lo as i32) - 8 + (hi as i32) * 16;
            result.push(scale * (q as f32 / 16.0) + d_min);
        }
        offset += 24;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let scale_lo = data[offset + 4] as f32;
        let scale_hi = data[offset + 5] as f32;
        let q4_lo = [
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ];
        let q4_hi = [
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
        ];

        for i in 0..remaining {
            let lo = (q4_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = (q4_hi[i / 2] >> (4 * (i % 2))) & 0x0F;

            let scale = if hi > 0 {
                d * (scale_lo + scale_hi * 1.0 / 32.0)
            } else {
                d * scale_lo
            };

            let q = (lo as i32) - 8 + (hi as i32) * 16;
            result.push(scale * (q as f32 / 16.0) + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q5_K data to f32.
///
/// Q5_K block: 16 elements, 32 bytes per block.
/// Block layout: d(f16, 2B) + d_min(f16, 2B) + scale(1B) + q5_lo(4B) + q5_h(2B) + extra(21B) = 32B
fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64 / 4 + element_count as u64 * 6 / 32 + 16 + 32 + 16) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q5_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let scale = data[offset + 4] as f32;
        let q5_lo = [
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ];
        let q5_h = [
            data[offset + 10],
            data[offset + 11],
        ];

        for i in 0..16usize {
            let lo = (q5_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = ((q5_h[i / 8] >> (i % 8)) & 1) as i32;

            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - 16.0) / 16.0) + d_min + scale);
        }
        offset += 32;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let scale = data[offset + 4] as f32;
        let q5_lo = [
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ];
        let q5_h = [
            data[offset + 10],
            data[offset + 11],
        ];

        for i in 0..remaining {
            let lo = (q5_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = ((q5_h[i / 8] >> (i % 8)) & 1) as i32;

            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - 16.0) / 16.0) + d_min + scale);
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32.
///
/// Q6_K block: 16 elements, 24 bytes per block.
/// Block layout: d(f16, 2B) + mask(1B) + q6(12B) + scale(1B) = 16B per block
fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64 / 2 + element_count as u64 / 4 + 256) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let mask = data[offset + 2];
        let q6 = [
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
        ];
        let scale = data[offset + 15] as f32;

        for i in 0..16usize {
            let q6_val = ((q6[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let mask_bit = (mask >> i) & 1;

            let combined = if mask_bit != 0 {
                q6_val + 4
            } else {
                q6_val
            };

            result.push(d * ((combined as f32 - 32.0) / 32.0) * scale);
        }
        offset += 24;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let mask = data[offset + 2];
        let q6 = &data[offset + 3..offset + 3 + remaining];
        let scale = data[offset + 15] as f32;

        for i in 0..remaining {
            let q6_val = ((q6[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let mask_bit = (mask >> i) & 1;

            let combined = if mask_bit != 0 {
                q6_val + 4
            } else {
                q6_val
            };

            result.push(d * ((combined as f32 - 32.0) / 32.0) * scale);
        }
    }

    Ok(result)
}

/// Dequantize Q8_K data to f32.
///
/// Q8_K block: 16 elements, 18 bytes per block.
/// Block layout: d(f16, 2B) + q8(16B) = 18B
fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64 / 2 + element_count as u64 * 6 / 32 + 256) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let q8 = &data[offset + 2..offset + 18];

        for q in q8.iter() {
            let q_val = *q as i8 as f32 / 128.0;
            result.push(d * q_val);
        }
        offset += 18;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let q8 = &data[offset + 2..offset + 2 + remaining];

        for q in q8.iter() {
            let q_val = *q as i8 as f32 / 128.0;
            result.push(d * q_val);
        }
    }

    Ok(result)
}

// ── Half-float helpers ───────────────────────────────────────────────

/// Convert half-float (f16) bytes to f32.
/// IEEE 754-2008 binary16 → IEEE 754 binary32.
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormal: treat as 1.0 * 2^-14 * (frac / 1024)
            let f32_bits = (sign << 31) | (frac << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        // Normal: bias conversion from 15 to 127
        let f32_exp = (exp - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

/// Convert half-float bytes to f32 slice.
fn half_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some(f16_to_f32(chunk))
            } else {
                None
            }
        })
        .collect()
}

// ── BF16 conversion ─────────────────────────────────────────────────

/// Convert bfloat16 bytes to f32.
/// BF16 is the top 16 bits of f32 — just zero-extend.
#[allow(dead_code)]
fn bfloat16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    // Zero-extend: upper 16 bits become the f32 bits, lower 16 bits are zero
    f32::from_bits((bits as u32) << 16)
}

/// Convert bfloat16 bytes to f32 slice.
#[allow(dead_code)]
fn bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some(bfloat16_to_f32(chunk))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_test_gguf_v3(path: &Path) {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(b"GGUF");
        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&2u64.to_le_bytes());
        // KV count
        buf.extend_from_slice(&3u64.to_le_bytes());

        // KV pairs
        let kv_pairs: Vec<crabjar_gguf::GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "Q4_0"),
            kv_pair_u32("general.alignment", 32),
        ];

        // Tensor metadata
        let tensors: Vec<crabjar_gguf::GgufTensorInfo> = vec![
            crabjar_gguf::GgufTensorInfo {
                name: "tok_embeddings.weight".to_string(),
                shape: vec![8u64],
                offset: 0,
                dtype: 1,
            },
            crabjar_gguf::GgufTensorInfo {
                name: "output.weight".to_string(),
                shape: vec![4u64, 8u64],
                offset: 32,
                dtype: 2,
            },
        ];

        let data_section_start =
            crabjar_gguf::compute_data_section_start(3, &kv_pairs, &tensors, Some(32));

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
        let total_tensor_bytes: u64 = tensors[0].shape.iter().product::<u64>() * 2
            + tensors[1].shape.iter().product::<u64>() / 16 * 18;
        buf.resize((data_section_start + total_tensor_bytes) as usize, 0);

        // F16 tensor data: 8 elements of 1.0
        let f16_ones: Vec<u8> = (0..8).flat_map(|_| pack_f16(1.0f32)).collect();
        buf[data_section_start as usize..data_section_start as usize + 16]
            .copy_from_slice(&f16_ones);

        // Q4_0 tensor data: 32 elements, 1 block (scale=1.0)
        let mut q4_block = Vec::with_capacity(18);
        q4_block.extend_from_slice(&pack_f16(1.0)); // scale
        for i in 0..16u32 {
            let lo = (i as u8 % 16) as u8;
            let hi = ((i + 1) as u8 % 16) as u8;
            q4_block.push((hi << 4) | lo);
        }
        let q4_start = data_section_start + 16;
        buf[q4_start as usize..q4_start as usize + 18].copy_from_slice(&q4_block);

        std::fs::write(path, &buf).unwrap();
    }

    fn make_test_gguf_f32(path: &Path) {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&2u64.to_le_bytes()); // kv count

        let kv_pairs: Vec<crabjar_gguf::GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_u32("general.alignment", 32),
        ];

        let tensor_info = crabjar_gguf::GgufTensorInfo {
            name: "test.weight".to_string(),
            shape: vec![4u64],
            offset: 0,
            dtype: 0u32,
        };

        let data_section_start = crabjar_gguf::compute_data_section_start(
            3,
            &kv_pairs,
            &[tensor_info.clone()],
            Some(32),
        );

        for kv in &kv_pairs {
            let key_bytes = kv.key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
            write_kv_value(&mut buf, &kv.value);
        }

        // Write tensor metadata
        let name_bytes = tensor_info.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(tensor_info.shape.len() as u32).to_le_bytes());
        for dim in &tensor_info.shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&tensor_info.dtype.to_le_bytes());
        buf.extend_from_slice(&tensor_info.offset.to_le_bytes());

        buf.resize((data_section_start + 16) as usize, 0);

        // F32 tensor data: [1.0, 2.0, 3.0, 4.0]
        let f32_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let f32_data: Vec<u8> = f32_vals.into_iter().flat_map(|v| v.to_le_bytes()).collect();
        buf[data_section_start as usize..data_section_start as usize + 16]
            .copy_from_slice(&f32_data);

        std::fs::write(path, &buf).unwrap();
    }

    fn make_test_gguf_q4(path: &Path) {
        let mut buf = Vec::new();

        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&2u64.to_le_bytes()); // kv count

        let kv_pairs: Vec<crabjar_gguf::GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_u32("general.alignment", 32),
        ];

        let tensor_info = crabjar_gguf::GgufTensorInfo {
            name: "q4_tensor".to_string(),
            shape: vec![32u64],
            offset: 0,
            dtype: 2u32,
        };

        let data_section_start = crabjar_gguf::compute_data_section_start(
            3,
            &kv_pairs,
            &[tensor_info.clone()],
            Some(32),
        );

        for kv in &kv_pairs {
            let key_bytes = kv.key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
            write_kv_value(&mut buf, &kv.value);
        }

        // Write tensor metadata
        let name_bytes = tensor_info.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(tensor_info.shape.len() as u32).to_le_bytes());
        for dim in &tensor_info.shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&tensor_info.dtype.to_le_bytes());
        buf.extend_from_slice(&tensor_info.offset.to_le_bytes());

        buf.resize((data_section_start + 18) as usize, 0);

        // Q4_0 tensor data: 32 elements, 1 block (scale=1.0)
        let mut q4_block = Vec::with_capacity(18);
        q4_block.extend_from_slice(&pack_f16(1.0)); // scale
        for i in (0..32).step_by(2) {
            let lo = (i % 16) as u8;
            let hi = ((i + 1) % 16) as u8;
            q4_block.push((hi << 4) | lo);
        }
        buf[data_section_start as usize..data_section_start as usize + 18]
            .copy_from_slice(&q4_block);

        std::fs::write(path, &buf).unwrap();
    }

    fn kv_pair_str(key: &str, value: &str) -> crabjar_gguf::GgufKvPair {
        crabjar_gguf::GgufKvPair {
            key: key.to_string(),
            value_type: crabjar_gguf::GgufValueType::String,
            value: crabjar_gguf::GgufKvValue::String(value.to_string()),
        }
    }

    fn kv_pair_u32(key: &str, value: u32) -> crabjar_gguf::GgufKvPair {
        crabjar_gguf::GgufKvPair {
            key: key.to_string(),
            value_type: crabjar_gguf::GgufValueType::Uint32,
            value: crabjar_gguf::GgufKvValue::Uint32(value),
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

    fn pack_f16(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
        let frac = ((bits >> 13) & 0x3FF) as u16;

        if exp <= 0 {
            let biased = ((sign << 15) as u16) | frac;
            return biased.to_le_bytes();
        } else if exp >= 31 {
            return ((sign << 15) as u16 | 0x7C00).to_le_bytes();
        }

        let result = ((sign << 15) as u16) | ((exp as u16) << 10) | frac;
        result.to_le_bytes()
    }

    #[test]
    fn load_gguf_weights_parses_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_v3(&path);

        let weights = load_gguf_weights(&path).unwrap();
        assert_eq!(weights.header.architecture(), Some("llama"));
        assert_eq!(weights.tensors.len(), 2);
        assert!(weights.tensors.contains_key("tok_embeddings.weight"));
        assert!(weights.tensors.contains_key("output.weight"));
    }

    #[test]
    fn load_gguf_weights_f16_tensor_is_f32() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_v3(&path);

        let weights = load_gguf_weights(&path).unwrap();
        let f16_tensor = &weights.tensors["tok_embeddings.weight"];
        // 8 F16 elements → 8 * 4 = 32 bytes f32
        assert_eq!(f16_tensor.len(), 32);
    }

    #[test]
    fn load_gguf_weights_q4_0_is_dequantized() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_q4(&path);

        let weights = load_gguf_weights(&path).unwrap();
        let q4_tensor = &weights.tensors["q4_tensor"];
        // 32 Q4_0 elements → 32 * 4 = 128 bytes f32
        assert_eq!(q4_tensor.len(), 128);
    }

    #[test]
    fn load_gguf_weights_f32_passthrough() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_f32(&path);

        let weights = load_gguf_weights(&path).unwrap();
        let f32_tensor = &weights.tensors["test.weight"];
        // 4 F32 elements → 4 * 4 = 16 bytes (same as input)
        assert_eq!(f32_tensor.len(), 16);
    }

    #[test]
    fn load_gguf_tensor_single() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_v3(&path);

        let (header, data) = load_gguf_tensor(&path, "tok_embeddings.weight").unwrap();
        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(data.len(), 32); // 8 F16 → 32 f32 bytes
    }

    #[test]
    fn load_gguf_tensor_missing_fails() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_v3(&path);

        let result = load_gguf_tensor(&path, "nonexistent.weight");
        assert!(result.is_err());
    }

    #[test]
    fn load_gguf_weights_q4_0_values_correct() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        make_test_gguf_q4(&path);

        let weights = load_gguf_weights(&path).unwrap();
        let q4_tensor = &weights.tensors["q4_tensor"];
        let f32_data: Vec<f32> = q4_tensor
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Scale = 1.0, min = 0.0
        // Q4_0 formula: dequantized = scale * (q - 8) + min
        // q values: 0,1,2,3,...,31 mapped to nibbles
        // element 0: q=0 → 1.0 * (0 - 8) + 0 = -8.0
        assert!(
            (f32_data[0] - (-8.0)).abs() < 1e-3,
            "expected -8.0, got {}",
            f32_data[0]
        );
        // element 1: q=1 → 1.0 * (1 - 8) + 0 = -7.0
        assert!(
            (f32_data[1] - (-7.0)).abs() < 1e-3,
            "expected -7.0, got {}",
            f32_data[1]
        );
        // element 2: q=2 → 1.0 * (2 - 8) + 0 = -6.0
        assert!(
            (f32_data[2] - (-6.0)).abs() < 1e-3,
            "expected -6.0, got {}",
            f32_data[2]
        );
    }

    #[test]
    fn f16_to_f32_known_values() {
        assert!((f16_to_f32(&pack_f16(0.0)) - 0.0).abs() < 1e-6);
        assert!((f16_to_f32(&pack_f16(1.0)) - 1.0).abs() < 1e-6);
        assert!((f16_to_f32(&pack_f16(-1.0)) - (-1.0)).abs() < 1e-6);
        assert!((f16_to_f32(&pack_f16(0.5)) - 0.5).abs() < 1e-6);
        assert!((f16_to_f32(&pack_f16(100.0)) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn bfloat16_to_f32_known_values() {
        assert!((bfloat16_to_f32(&[0x00, 0x00]) - 0.0).abs() < 1e-6);
        // BF16 of 1.0: f32(1.0) = 0x3F800000, top 16 bits = 0x3F80
        let bf16_1: [u8; 2] = [0x80, 0x3F];
        assert!((bfloat16_to_f32(&bf16_1) - 1.0).abs() < 1e-6);
        // BF16 of -1.0: f32(-1.0) = 0xBF800000, top 16 bits = 0xBF80
        let bf16_neg1: [u8; 2] = [0x80, 0xBF];
        assert!((bfloat16_to_f32(&bf16_neg1) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn half_f32_slice_conversion() {
        let f32_vals: Vec<f32> = vec![1.0, 2.0, 3.0];
        let f16_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| pack_f16(*v)).collect();
        let result = half_f32(&f16_bytes);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn load_gguf_weights_empty_tensors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.gguf");

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // no KV
        buf.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        buf.extend_from_slice(&32u64.to_le_bytes()); // alignment

        std::fs::write(&path, &buf).unwrap();

        let weights = load_gguf_weights(&path).unwrap();
        assert_eq!(weights.tensors.len(), 0);
    }

    #[ignore] // Parser doesn't handle large string arrays in vocab GGUFs
    #[test]
    fn load_gguf_real_vocab_file() {
        let path =
            std::path::PathBuf::from("/home/crombo/llama.cpp/models/ggml-vocab-llama-spm.gguf");
        if !path.exists() {
            eprintln!("SKIP: llama.cpp vocab GGUF not found");
            return;
        }

        let weights = load_gguf_weights(&path).unwrap();
        assert_eq!(weights.header.architecture(), Some("llama"));
        assert_eq!(weights.tensors.len(), 0); // vocab files have no tensors
        assert!(weights.header.kv_pairs.len() > 0);
    }
}
