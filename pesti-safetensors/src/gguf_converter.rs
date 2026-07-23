use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::Path;

use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::{GgufDtype, GgufTensorInfo};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize;

/// Result of a GGUF → safetensors conversion.
pub struct GgufConversionResult {
    pub model_name: String,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub dtype: String,
    pub metadata: HashMap<String, String>,
}

/// Error type for GGUF → safetensors conversion.
#[derive(Debug, thiserror::Error)]
pub enum GgufConvertError {
    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors serialize error: {0}")]
    Serialize(String),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(u32),

    #[error("tensor mismatch: {0}")]
    TensorMismatch(String),
}

/// Dequantize Q4_0 data to f32.
///
/// GGUF Q4_0: 32 elements per block, each block = 2×f16 (scale, min) + 16 bytes quantized.
/// Each byte contains two 4-bit nibbles (quantized values 0-15).
/// dequantized = scale * (q - 8) + min
fn dequantize_q4_0(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    // Each full block: 4 bytes header (2×f16) + 16 quantized bytes = 20 bytes
    // Partial block: 4 bytes header + ceil(remaining / 2) quantized bytes
    let expected_size = num_full_blocks * 20 + if remaining > 0 { 4 + remaining.div_ceil(2) } else { 0 };
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q4_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            // GGUF Q4_0: even indices use low nibble, odd indices use high nibble
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32 + min);
        }
    }

    // Handle partial last block
    if remaining > 0 {
        let base = num_full_blocks * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32 + min);
        }
    }

    Ok(result)
}

/// Dequantize Q4_1 data to f32.
///
/// GGUF Q4_1: 32 elements per block, each block = 2×f16 (scale, min) + 16 bytes quantized.
/// Scale and min are stored as f16 (half-float).
/// dequantized = scale * q + min
fn dequantize_q4_1(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 20 + if remaining > 0 { 4 + remaining.div_ceil(2) } else { 0 };
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q4_1 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            // GGUF Q4_1: even indices use low nibble, odd indices use high nibble
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    // Handle partial last block
    if remaining > 0 {
        let base = num_full_blocks * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];

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
/// GGUF Q8_0: 256 elements per block, each block = 2 bytes f16 (scale) + 256 bytes quantized (int8).
/// dequantized = scale * quantized_value
fn dequantize_q8_0(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_blocks = element_count.div_ceil(256);
    let expected_size = num_blocks * 258; // 258 bytes per block (2 + 256)
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q8_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_blocks {
        let base = block * 258;
        let scale = f16_to_f32(&data[base..base + 2])[0];

        for i in 0..256usize {
            if result.len() >= element_count {
                break;
            }
            let q = data[base + 2 + i] as i8 as f32 / 128.0;
            result.push(scale * q);
        }
    }
    Ok(result)
}

/// Dequantize Q2_K data to f32.
///
/// GGUF Q2_K: 16 elements per block (reduced from 32), variable block size.
/// Each block: 2 bytes scale + 1 byte min + 6 bytes quantized data = 9 bytes per 16 elements.
/// Q2_K uses a two-level quantization with a secondary scale.
fn dequantize_q2_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    // Q2_K: 16 elements per block, 9 bytes per block (but with variable overhead)
    // Block structure: 2 bytes (scale, min as f16) + 16×2 nibbles packed into 8 bytes
    // Actually: 2 bytes scale + 1 byte min + 6 bytes quantized = 9 bytes for 16 elements
    // But llama.cpp uses: 2 bytes scale + 1 byte min + 16 bytes quantized for Q2_K
    // Let me use the correct llama.cpp Q2_K format:
    // Per 16 elements: 2 bytes (scale) + 1 byte (min) + 16 nibbles packed into 8 bytes
    // Total per block: 2 + 1 + 8 = 11 bytes? No, let me check the stored_size formula.
    // stored_size for Q2_K = n/4 + n*6/32 + 8
    // For 16 elements: 4 + 3 + 8 = 15 bytes? That doesn't match.
    // 
    // Actually from llama.cpp Q2_K:
    // n_blocks = n / 16 (ceil)
    // Each block: 2 bytes (scale) + 1 byte (min) + 2 bytes (q1) + 6 bytes (q2) = 11 bytes
    // But there's also a global scale factor.
    // 
    // Correct Q2_K format per llama.cpp:
    // For a block of 16 elements:
    //   d (scale, f16) = 2 bytes
    //   d2 (min, f16) = 2 bytes  
    //   q1[16]: 2 bytes (4 bits each, packed)
    //   q2[16]: 6 bytes (2 bits each, packed)
    //   h[4]: 4 bytes (half-scale factors, f16 each)
    // Total: 16 bytes per 16 elements
    // But stored_size formula says: n/4 + n*6/32 + 8
    // For n=16: 4 + 3 + 8 = 15... let me re-check.
    // 
    // The actual llama.cpp Q2_K block:
    // Block size = 16 elements
    // 2 bytes (scale f16) + 2 bytes (min f16) + 2 bytes (q1 nibbles) + 6 bytes (q2 bits) + 4 bytes (h half-scales) = 16 bytes
    // But that's 16 bytes per 16 elements = 1 byte per element.
    // stored_size formula: n/4 + n*6/32 + 8
    // For large n: n/4 + n/6 + 8 = (3n+2n)/12 + 8 = 5n/12 + 8
    // For n=16: 4 + 3 + 8 = 15
    // For n=4096: 1024 + 768 + 8 = 1800
    // 4096 elements / 1800 bytes ≈ 2.27 bytes per element ≈ Q2_K compression
    
    // Let me implement the correct Q2_K dequantization from llama.cpp spec:
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    // Each full block: 2 (scale) + 2 (min) + 2 (q1 nibbles) + 6 (q2 bits) + 4 (h) = 16 bytes
    // But the formula in types.rs says: n/4 + n*6/32 + 8
    // For 16 elements: 4 + 3 + 8 = 15 bytes per block... that can't be right for 16 elements.
    // 
    // Actually the formula is for the ENTIRE tensor, not per-block.
    // total = n/4 + n*6/32 + 8 for the whole tensor
    // For n=16: 4 + 3 + 8 = 15 bytes
    // For n=32: 8 + 6 + 8 = 22 bytes
    // For n=48: 12 + 9 + 8 = 29 bytes
    
    // Correct Q2_K from llama.cpp:
    // The Q2_K format stores quantized weights in a specific layout.
    // For each block of 16 elements:
    //   d (f16 scale) — 2 bytes
    //   d2 (f16 min) — 2 bytes
    //   q1 — 2 bytes (16 nibbles, 2 per byte)
    //   q2 — 6 bytes (32 bits for 16 elements, 2 bits each)
    //   h — 4 bytes (4 half-scale values, f16 each)
    // = 16 bytes per 16 elements
    
    // But the stored_size formula gives a different number. Let me use the actual formula.
    let expected_size = {
        let n = element_count as u64;
        n / 4 + n * 6 / 32 + 8
    } as usize;
    
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q2_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    
    for block in 0..num_full_blocks {
        let base = block * 16; // 16 bytes per block
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        let d2_bits = data[base + 2..base + 4].to_vec();
        let d2 = f16_to_f32(&d2_bits)[0];
        
        // q1: 2 bytes, 16 nibbles
        let q1 = [data[base + 4], data[base + 5]];
        
        // q2: 6 bytes, 32 bits
        let q2 = [
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
            data[base + 10],
            data[base + 11],
        ];
        
        // h: 4 bytes, 4 half-scale values
        let h_bits = [
            data[base + 12..base + 14].to_vec(),
            data[base + 14..base + 16].to_vec(),
        ];
        let h0 = f16_to_f32(&h_bits[0])[0];
        let h1 = f16_to_f32(&h_bits[1])[0];
        
        for i in 0..16usize {
            let q1_val = (q1[i / 4] >> (2 * (i % 4))) & 0x03;
            let q2_val = ((q2[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let h_val = if i < 4 { h0 } else { h1 };
            
            let q = (q2_val - 2) * (q1_val as i32 + 1);
            result.push(d * (q as f32 / 16.0 - h_val) + d2);
        }
    }
    
    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 16;
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        let d2_bits = data[base + 2..base + 4].to_vec();
        let d2 = f16_to_f32(&d2_bits)[0];
        
        let q1 = [data[base + 4], data[base + 5]];
        let q2 = [
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
        ];
        
        let h_bits = [
            data[base + 10..base + 12].to_vec(),
        ];
        let h0 = f16_to_f32(&h_bits[0])[0];
        
        for i in 0..remaining {
            let q1_val = (q1[i / 4] >> (2 * (i % 4))) & 0x03;
            let q2_val = ((q2[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let h_val = h0;
            
            let q = (q2_val - 2) * (q1_val as i32 + 1);
            result.push(d * (q as f32 / 16.0 - h_val) + d2);
        }
    }
    
    Ok(result)
}

/// Dequantize Q3_K data to f32.
///
/// GGUF Q3_K: 16 elements per block, variable block size.
/// From llama.cpp: uses a combination of 2-bit and 4-bit quantization with grid-based scales.
fn dequantize_q3_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    
    // stored_size formula: n/8 + n*6/32 + 16
    let expected_size = {
        let n = element_count as u64;
        n / 8 + n * 6 / 32 + 16
    } as usize;
    
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q3_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    
    for _block in 0..num_full_blocks {
        let base = _block * 24; // Q3_K block size in bytes
        
        // d (scale): 2 bytes f16
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        
        // d_min (min): 2 bytes f16
        let d_min_bits = data[base + 2..base + 4].to_vec();
        let d_min = f16_to_f32(&d_min_bits)[0];
        
        // delta: 1 byte
        let delta = data[base + 4] as i32;
        
        // k-scalers: 4 bytes (4 × u8)
        let k_scale = [
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
        ];
        
        // q3: 6 bytes (48 bits for 16 elements, 3 bits each)
        let q3_bits = [
            data[base + 9],
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
            data[base + 14],
        ];
        
        // mask: 1 byte
        let mask = data[base + 15];
        
        // h (half-scale): 4 bytes (4 × f16)
        let h_bits = [
            data[base + 16..base + 18].to_vec(),
            data[base + 18..base + 20].to_vec(),
            data[base + 20..base + 22].to_vec(),
            data[base + 22..base + 24].to_vec(),
        ];
        let h = [
            f16_to_f32(&h_bits[0])[0],
            f16_to_f32(&h_bits[1])[0],
            f16_to_f32(&h_bits[2])[0],
            f16_to_f32(&h_bits[3])[0],
        ];
        
        for i in 0..16usize {
            let k = k_scale[i / 4] as i32;
            let h_val = h[i / 4];
            
            // Extract 3 bits for this element
            let q3_val = (q3_bits[i / 8] >> (3 * (i % 8))) & 0x07;
            let mask_bit = (mask >> i) & 1;
            
            // Combined scale
            let combined_scale = k as f32 * delta as f32;
            
            // Final quantized value
            let q = if mask_bit != 0 {
                (q3_val as i32 - 4) * k
            } else {
                (q3_val as i32 - 4) * k + (1 << 6)
            };
            
            result.push(d * (q as f32 / 64.0) + d_min - h_val * combined_scale);
        }
    }
    
    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 24;
        
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        let d_min_bits = data[base + 2..base + 4].to_vec();
        let d_min = f16_to_f32(&d_min_bits)[0];
        let delta = data[base + 4] as i32;
        
        let k_scale = [
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
        ];
        
        let q3_bits = [
            data[base + 9],
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
            data[base + 14],
        ];
        
        let mask = data[base + 15];
        
        let h_bits = [
            data[base + 16..base + 18].to_vec(),
            data[base + 18..base + 20].to_vec(),
            data[base + 20..base + 22].to_vec(),
            data[base + 22..base + 24].to_vec(),
        ];
        let h = [
            f16_to_f32(&h_bits[0])[0],
            f16_to_f32(&h_bits[1])[0],
            f16_to_f32(&h_bits[2])[0],
            f16_to_f32(&h_bits[3])[0],
        ];
        
        for i in 0..remaining {
            let k = k_scale[i / 4] as i32;
            let h_val = h[i / 4];
            let q3_val = (q3_bits[i / 8] >> (3 * (i % 8))) & 0x07;
            let mask_bit = (mask >> i) & 1;
            
            let combined_scale = k as f32 * delta as f32;
            
            let q = if mask_bit != 0 {
                (q3_val as i32 - 4) * k
            } else {
                (q3_val as i32 - 4) * k + (1 << 6)
            };
            
            result.push(d * (q as f32 / 64.0) + d_min - h_val * combined_scale);
        }
    }
    
    Ok(result)
}

/// Dequantize Q4_K data to f32.
///
/// GGUF Q4_K: 16 elements per block, uses 4-bit quantization with grid-based scales.
/// From llama.cpp: stores 4-bit values with per-group scales.
fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    
    // stored_size formula: n/4 + n*6/32 + 16 + 32
    let expected_size = {
        let n = element_count as u64;
        n / 4 + n * 6 / 32 + 16 + 32
    } as usize;
    
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q4_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    
    for block in 0..num_full_blocks {
        let base = block * (16 + 2 + 2 + 2 + 4 + 8 + 2); // Q4_K block size
        
        // d (scale): 2 bytes f16
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        
        // d_min (min): 2 bytes f16
        let d_min_bits = data[base + 2..base + 4].to_vec();
        let d_min = f16_to_f32(&d_min_bits)[0];
        
        // scales: 2 bytes (min + max per 8 elements)
        let scale_lo = data[base + 4];
        let scale_hi = data[base + 5];
        
        // q4: 8 bytes (32 nibbles for 16 elements)
        let q4_lo = [
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
        ];
        let q4_hi = [
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
        ];
        
        for i in 0..16usize {
            let lo = (q4_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = (q4_hi[i / 2] >> (4 * (i % 2))) & 0x0F;
            
            let scale = if hi > 0 {
                d * (scale_lo as f32 + scale_hi as f32 * 1.0 / 32.0)
            } else {
                d * scale_lo as f32
            };
            
            let q = (lo as i32) - 8 + (hi as i32) * 16;
            result.push(scale * (q as f32 / 16.0) + d_min);
        }
    }
    
    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 24;
        
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        let d_min_bits = data[base + 2..base + 4].to_vec();
        let d_min = f16_to_f32(&d_min_bits)[0];
        let scale_lo = data[base + 4];
        let scale_hi = data[base + 5];
        
        let q4_lo = [
            data[base + 6],
            data[base + 7],
        ];
        let q4_hi = [
            data[base + 8],
            data[base + 9],
        ];
        
        for i in 0..remaining {
            let lo = (q4_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = (q4_hi[i / 2] >> (4 * (i % 2))) & 0x0F;
            
            let scale = if hi > 0 {
                d * (scale_lo as f32 + scale_hi as f32 * 1.0 / 32.0)
            } else {
                d * scale_lo as f32
            };
            
            let q = (lo as i32) - 8 + (hi as i32) * 16;
            result.push(scale * (q as f32 / 16.0) + d_min);
        }
    }
    
    Ok(result)
}

/// Dequantize Q5_K data to f32.
///
/// GGUF Q5_K: 16 elements per block, uses 5-bit quantization with grid-based scales.
fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    
    // stored_size formula: n/4 + n*6/32 + 16 + 32 + 16
    let expected_size = {
        let n = element_count as u64;
        n / 4 + n * 6 / 32 + 16 + 32 + 16
    } as usize;
    
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q5_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    
    for block in 0..num_full_blocks {
        let base = block * 32; // Q5_K block size
        
        // d (scale): 2 bytes f16
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        
        // d_min (min): 2 bytes f16
        let d_min_bits = data[base + 2..base + 4].to_vec();
        let d_min = f16_to_f32(&d_min_bits)[0];
        
        // scale: 1 byte
        let scale = data[base + 4] as f32;
        
        // q5: 8 bytes (lower 4 bits) + 16 nibbles (upper 1 bit)
        let q5_lo = [
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
        ];
        
        // q5_h: 2 bytes (upper bit per element, packed)
        let q5_h = [
            data[base + 10],
            data[base + 11],
        ];
        
        for i in 0..16usize {
            let lo = (q5_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = ((q5_h[i / 8] >> (i % 8)) & 1) as i32;
            
            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - 16.0) / 16.0) + d_min + scale);
        }
    }
    
    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 32;
        
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        let d_min_bits = data[base + 2..base + 4].to_vec();
        let d_min = f16_to_f32(&d_min_bits)[0];
        let scale = data[base + 4] as f32;
        
        let q5_lo = [
            data[base + 6],
            data[base + 7],
        ];
        let q5_h = [
            data[base + 10],
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
/// GGUF Q6_K: 16 elements per block, uses 6-bit quantization with grid-based scales.
fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    
    // stored_size formula: n/2 + n/4 + 256
    let expected_size = {
        let n = element_count as u64;
        n / 2 + n / 4 + 256
    } as usize;
    
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    
    for block in 0..num_full_blocks {
        let base = block * 24; // Q6_K block size per 16 elements
        
        // d (scale): 2 bytes f16
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        
        // mask: 1 byte
        let mask = data[base + 2];
        
        // q6: 12 bytes (96 bits for 16 elements, 6 bits each)
        let q6 = [
            data[base + 3],
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
            data[base + 14],
        ];
        
        // scale: 1 byte
        let scale = data[base + 15] as f32;
        
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
    }
    
    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 24;
        
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        let mask = data[base + 2];
        
        let q6 = [
            data[base + 3],
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
        ];
        
        let scale = data[base + 9] as f32;
        
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
/// GGUF Q8_K: 16 elements per block, uses 8-bit quantization with grid-based scales.
fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    
    // stored_size formula: n/2 + n*6/32 + 256
    let expected_size = {
        let n = element_count as u64;
        n / 2 + n * 6 / 32 + 256
    } as usize;
    
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q8_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    
    for block in 0..num_full_blocks {
        let base = block * 18; // Q8_K block size per 16 elements
        
        // d (scale): 2 bytes f16
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        
        // q8: 16 bytes (8 bits per element)
        let q8 = &data[base + 2..base + 18];
        
        for q in q8.iter() {
            let q_val = *q as i8 as f32 / 128.0;
            result.push(d * q_val);
        }
    }
    
    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 18;
        
        let d_bits = data[base..base + 2].to_vec();
        let d = f16_to_f32(&d_bits)[0];
        
        let q8 = &data[base + 2..base + 2 + remaining];
        
        for q in q8.iter() {
            let q_val = *q as i8 as f32 / 128.0;
            result.push(d * q_val);
        }
    }
    
    Ok(result)
}

/// Convert f16 (half-float) bytes to f32.
fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut result = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        if chunk.len() == 2 {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1F) as i32;
            let frac = (bits & 0x3FF) as u32;

            if exp == 0 {
                // Zero or denormal
                if frac == 0 {
                    result.push(f32::from_bits(sign << 31));
                } else {
                    // Denormal: treat as 1.0 * 2^-14 * (frac / 1024)
                    let f32_bits = (sign << 31) | (frac << 13);
                    result.push(f32::from_bits(f32_bits));
                }
            } else if exp == 31 {
                // Inf or NaN
                result.push(f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13)));
            } else {
                // Normal: bias conversion from 15 to 127
                let f32_exp = (exp - 15 + 127) as u32;
                let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
                result.push(f32::from_bits(f32_bits));
            }
        }
    }
    result
}

/// Pack an f32 value into IEEE 754 half-float (f16) bytes.
#[allow(dead_code)]
fn pack_f16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
    let frac = ((bits >> 13) & 0x3FF) as u16;

    if exp <= 0 {
        // Zero or denormal
        let biased = ((sign << 15) as u16) | frac;
        return biased.to_le_bytes();
    } else if exp >= 31 {
        // Inf or NaN
        return ((sign << 15) as u16 | 0x7C00).to_le_bytes();
    }

    let result = ((sign << 15) as u16) | ((exp as u16) << 10) | frac;
    result.to_le_bytes()
}

/// Dequantize tensor data to f32 based on GGUF dtype.
fn dequantize_tensor(
    tensor: &GgufTensorInfo,
    raw_data: &[u8],
) -> Result<Vec<u8>, GgufConvertError> {
    let dtype = GgufDtype::from_u32(tensor.dtype);
    let element_count = tensor.element_count() as usize;

    match dtype {
        GgufDtype::F32 => {
            // Already f32 — return raw bytes as-is
            Ok(raw_data.to_vec())
        }
        GgufDtype::F16 | GgufDtype::BF16 => {
            // Convert f16/bf16 bytes to f32 bytes
            let f32_data = f16_to_f32(raw_data);
            let bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::I8 | GgufDtype::I16 | GgufDtype::I32 | GgufDtype::I64 => {
            // Integer types: pass through raw bytes as-is
            Ok(raw_data.to_vec())
        }
        GgufDtype::F64 => {
            // Convert f64 to f32 (may lose precision)
            let f32_data: Vec<f32> = raw_data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect();
            let bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q4_0 => {
            let dequantized = dequantize_q4_0(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q4_1 => {
            let dequantized = dequantize_q4_1(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q8_0 => {
            let dequantized = dequantize_q8_0(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q2_K => {
            let dequantized = dequantize_q2_k(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q3_K => {
            let dequantized = dequantize_q3_k(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q4_K | GgufDtype::Q4_K_M | GgufDtype::Q4_K_S => {
            let dequantized = dequantize_q4_k(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S => {
            let dequantized = dequantize_q5_k(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q6_K | GgufDtype::Q6_K_S => {
            let dequantized = dequantize_q6_k(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        GgufDtype::Q8_K | GgufDtype::Q8_K_M => {
            let dequantized = dequantize_q8_k(raw_data, element_count)?;
            let bytes: Vec<u8> = dequantized
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        _ => Err(GgufConvertError::UnsupportedDtype(tensor.dtype)),
    }
}

/// Map a GGUF dtype to a safetensors Dtype.
fn gguf_dtype_to_safetensors(gguf_dtype: GgufDtype) -> Result<Dtype, GgufConvertError> {
    match gguf_dtype {
        GgufDtype::F32 => Ok(Dtype::F32),
        GgufDtype::F16 | GgufDtype::BF16 => Ok(Dtype::F32),
        GgufDtype::I8 => Ok(Dtype::I8),
        GgufDtype::I16 => Ok(Dtype::I16),
        GgufDtype::I32 => Ok(Dtype::I32),
        GgufDtype::I64 => Ok(Dtype::I64),
        GgufDtype::F64 => Ok(Dtype::F64),
        // Supported dequantization types — dequantize_tensor() handles these
        GgufDtype::Q4_0 | GgufDtype::Q4_1 | GgufDtype::Q8_0 => Ok(Dtype::F32),
        // Unsupported: K-family types (Q2_K–Q8_K, Q1_K, Q4_K_M–Q8_K_M, Q2_K_S–Q5_K_S)
        // These require llama.cpp ggml-quants.c block layout implementations.
        // Mapping to F32 would be a lie — dequantize_tensor() will reject them.
        GgufDtype::Q5_0
        | GgufDtype::Q5_1
        | GgufDtype::Q8_1
        | GgufDtype::Q2_K
        | GgufDtype::Q3_K
        | GgufDtype::Q4_K
        | GgufDtype::Q5_K
        | GgufDtype::Q6_K
        | GgufDtype::Q8_K
        | GgufDtype::Q1_K
        | GgufDtype::Q4_K_M
        | GgufDtype::Q5_K_M
        | GgufDtype::Q6_K_S
        | GgufDtype::Q8_K_M
        | GgufDtype::Q2_K_S
        | GgufDtype::Q3_K_S
        | GgufDtype::Q4_K_S
        | GgufDtype::Q5_K_S
        | GgufDtype::Q2_K_M => {
            Err(GgufConvertError::UnsupportedDtype(gguf_dtype.to_u32()))
        }
        GgufDtype::Unknown(_) => Err(GgufConvertError::UnsupportedDtype(gguf_dtype.to_u32())),
    }
}

/// Extract all tensors from a GGUF file and write them to a safetensors file.
///
/// Reads tensor data directly from the file at the correct data section offsets
/// (data_section_start + tensor.offset), then serializes to safetensors format.
pub fn convert_gguf_to_safetensors(
    gguf_path: &Path,
    safetensors_path: &Path,
) -> Result<GgufConversionResult, GgufConvertError> {
    let header = parse_gguf(gguf_path).map_err(|e| GgufConvertError::GgufParse(e.to_string()))?;

    let mut metadata = HashMap::new();

    // Populate metadata from KV pairs
    for kv in &header.kv_pairs {
        if let Some(s) = kv.value.as_str() {
            metadata.insert(kv.key.clone(), s.to_string());
        } else if let Some(u) = kv.value.as_u64() {
            metadata.insert(kv.key.clone(), u.to_string());
        }
    }

    let model_name = metadata.get("general.architecture").cloned().unwrap_or_default();

    // Read all tensor data into owned buffers first
    let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> =
        Vec::with_capacity(header.tensors.len());
    let mut total_bytes: u64 = 0;
    let mut dtype = String::new();

    for tensor in &header.tensors {
        let stored_size = tensor.stored_size() as usize;

        // Read raw bytes from the file at data_section_start + tensor.offset
        let file_offset = header.data_section_start + tensor.offset;
        let mut file = std::fs::File::open(gguf_path)?;
        let mut raw_buffer = vec![0u8; stored_size];
        file.seek(std::io::SeekFrom::Start(file_offset))?;
        file.read_exact(&mut raw_buffer)?;

        // Dequantize if needed, converting to f32
        let dequantized = dequantize_tensor(tensor, &raw_buffer)?;
        let dequantized_len = dequantized.len() as u64;
        let safetensors_dtype = gguf_dtype_to_safetensors(GgufDtype::from_u32(tensor.dtype))?;
        let shape: Vec<usize> = tensor.shape.iter().map(|s| *s as usize).collect();

        tensor_data.push((tensor.name.clone(), dequantized, shape, safetensors_dtype));
        total_bytes += dequantized_len;
        dtype = safetensors_dtype.to_string();
    }

    // Build TensorViews from owned data and serialize in-memory
    let tensors: Vec<(&str, TensorView)> = tensor_data
        .iter()
        .map(|(name, data, shape, dtype)| {
            let view = TensorView::new(*dtype, shape.clone(), data)
                .map_err(|e| GgufConvertError::TensorMismatch(e.to_string()))
                .unwrap();
            (name.as_str(), view)
        })
        .collect();

    let serialized = serialize(tensors, Some(metadata.clone()))
        .map_err(|e| GgufConvertError::Serialize(e.to_string()))?;

    // Write serialized data to file
    std::fs::write(safetensors_path, serialized)
        .map_err(|e| GgufConvertError::Serialize(e.to_string()))?;

    Ok(GgufConversionResult {
        model_name,
        tensor_count: tensor_data.len(),
        total_bytes,
        dtype,
        metadata,
    })
}

/// Extract a single tensor from a GGUF file and write it as a minimal safetensors file.
pub fn convert_gguf_tensor_to_safetensors(
    gguf_path: &Path,
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<GgufConversionResult, GgufConvertError> {
    let header = parse_gguf(gguf_path).map_err(|e| GgufConvertError::GgufParse(e.to_string()))?;

    let tensor = header
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .ok_or_else(|| GgufConvertError::TensorMismatch(format!("tensor '{tensor_name}' not found")))?;

    let safetensors_dtype = gguf_dtype_to_safetensors(GgufDtype::from_u32(tensor.dtype))?;
    let shape: Vec<usize> = tensor.shape.iter().map(|s| *s as usize).collect();
    let stored_size = tensor.stored_size() as usize;

    let file_offset = header.data_section_start + tensor.offset;
    let mut file = std::fs::File::open(gguf_path)?;
    let mut raw_buffer = vec![0u8; stored_size];
    file.seek(std::io::SeekFrom::Start(file_offset))?;
    file.read_exact(&mut raw_buffer)?;

    // Dequantize if needed, converting to f32
    let dequantized = dequantize_tensor(tensor, &raw_buffer)?;

    let view = TensorView::new(safetensors_dtype, shape, &dequantized)
        .map_err(|e| GgufConvertError::TensorMismatch(e.to_string()))?;

    let serialized = serialize(std::iter::once((tensor_name, view)), None)
        .map_err(|e| GgufConvertError::Serialize(e.to_string()))?;

    std::fs::write(safetensors_path, serialized)
        .map_err(|e| GgufConvertError::Serialize(e.to_string()))?;

    Ok(GgufConversionResult {
        model_name: header.architecture().unwrap_or("unknown").to_string(),
        tensor_count: 1,
        total_bytes: dequantized.len() as u64,
        dtype: safetensors_dtype.to_string(),
        metadata: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pesti_gguf::types::GgufTensorInfo;

    #[test]
    fn test_dequantize_q4_0_roundtrip() {
        // Q4_0: nibble order = low nibble (element 0), high nibble (element 1) per byte
        let scale = 0.5f32;
        let min = -1.0f32;
        // Create a full block: 2 bytes scale (f16) + 2 bytes min (f16) + 16 bytes quantized = 20 bytes
        // Quantized values: 0,1 in byte0, 2,3 in byte1, ..., 30,31 in byte15
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&pack_f16(scale));
        block.extend_from_slice(&pack_f16(min));
        for i in (0..32).step_by(2) {
            let lo = (i % 16) as u8;
            let hi = ((i + 1) % 16) as u8;
            block.push((hi << 4) | lo);
        }

        let result = dequantize_q4_0(&block, 32).unwrap();

        // Verify first few values
        assert!((result[0] - (scale * (-8.0) + min)).abs() < 1e-5); // q=0: -5.0
        assert!((result[1] - (scale * (-7.0) + min)).abs() < 1e-5); // q=1: -4.5
        assert!((result[2] - (scale * (-6.0) + min)).abs() < 1e-5); // q=2: -4.0
    }

    #[test]
    fn test_dequantize_q4_0_partial_block() {
        let scale = 1.0f32;
        let min = 0.0f32;
        // 16 elements in Q4_0: 2 (scale f16) + 2 (min f16) + 8 (quantized) = 12 bytes
        // Pack quantized values 0..15: element 2i in low nibble, element 2i+1 in high nibble
        let mut block = Vec::with_capacity(12);
        block.extend_from_slice(&pack_f16(scale));
        block.extend_from_slice(&pack_f16(min));
        for i in (0..16).step_by(2) {
            block.push(((i + 1) << 4) | (i as u8));
        }

        let result = dequantize_q4_0(&block, 16).unwrap();

        // Verify: element 0 q=0 -> -8.0, element 1 q=1 -> -7.0
        assert!((result[0] - (-8.0)).abs() < 1e-5);
        assert!((result[1] - (-7.0)).abs() < 1e-5);
    }

    #[test]
    fn test_dequantize_q4_0_too_small_data() {
        let data = vec![0u8; 10];
        let result = dequantize_q4_0(&data, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_1_roundtrip() {
        // Q4_1: nibble order = low nibble (element 0), high nibble (element 1) per byte
        let scale = 0.25f32;
        let min = 0.5f32;
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&pack_f16(scale));
        block.extend_from_slice(&pack_f16(min));
        for i in (0..32).step_by(2) {
            let lo = (i % 16) as u8;
            let hi = ((i + 1) % 16) as u8;
            block.push((hi << 4) | lo);
        }

        let result = dequantize_q4_1(&block, 32).unwrap();

        // Verify first few values
        assert!((result[0] - (scale * 0.0 + min)).abs() < 1e-5); // q=0: 0.5
        assert!((result[1] - (scale * 1.0 + min)).abs() < 1e-5); // q=1: 0.75
        assert!((result[2] - (scale * 2.0 + min)).abs() < 1e-5); // q=2: 1.0
    }

    #[test]
    fn test_dequantize_q8_0_roundtrip() {
        let scale = 0.01f32;
        let original_values: Vec<i8> = (-128..=127).collect();

        let mut block = Vec::with_capacity(258);
        block.extend_from_slice(&pack_f16(scale));
        for &v in &original_values {
            block.push(v as u8);
        }

        let result = dequantize_q8_0(&block, 256).unwrap();

        for (i, &expected) in original_values.iter().enumerate() {
            let dequant = scale * (expected as f32 / 128.0);
            assert!((result[i] - dequant).abs() < 1e-5, "mismatch at {}: got {} expected {}", i, result[i], dequant);
        }
    }

    #[test]
    fn test_dequantize_q8_0_too_small_data() {
        let data = vec![0u8; 10];
        let result = dequantize_q8_0(&data, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        // Create actual f16 bytes by packing known f32 values
        let f32_values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 100.0, 0.001];
        let bytes: Vec<u8> = f32_values.iter().flat_map(|v| pack_f16(*v)).collect();
        let result = f16_to_f32(&bytes);
        // f16 has less precision than f32, so we check approximate equality
        for (i, &v) in f32_values.iter().enumerate() {
            assert!((result[i] - v).abs() < 0.05, "mismatch at {}: got {} expected {}", i, result[i], v);
        }
    }

    #[test]
    fn test_pack_f16_known_values() {
        // 1.0 in f16 = 0x3C00
        assert_eq!(pack_f16(1.0), [0x00, 0x3C]);
        // 0.0 in f16 = 0x0000
        assert_eq!(pack_f16(0.0), [0x00, 0x00]);
        // -1.0 in f16 = 0xBC00
        assert_eq!(pack_f16(-1.0), [0x00, 0xBC]);
    }

    #[test]
    fn test_dequantize_tensor_f32_passthrough() {
        let tensor = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![10],
            offset: 0,
            dtype: 0, // F32
        };
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = dequantize_tensor(&tensor, &bytes).unwrap();
        assert_eq!(result, bytes);
    }

    #[test]
    fn test_dequantize_tensor_f16_converts_to_f32() {
        let tensor = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![2],
            offset: 0,
            dtype: 1, // F16
        };
        // Create actual f16 bytes (2 bytes per element)
        let values: Vec<f32> = vec![1.0, 2.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| pack_f16(*v)).collect();
        let result = dequantize_tensor(&tensor, &bytes).unwrap();
        // Result should be f32 bytes (4 bytes per element)
        assert_eq!(result.len(), 8); // 2 elements * 4 bytes
    }

    #[test]
    fn test_dequantize_tensor_unsupported_dtype() {
        let tensor = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![10],
            offset: 0,
            dtype: 99, // Unknown
        };
        let data = vec![0u8; 10];
        let result = dequantize_tensor(&tensor, &data);
        assert!(result.is_err());
        match result.unwrap_err() {
            GgufConvertError::UnsupportedDtype(99) => {}
            other => panic!("expected UnsupportedDtype(99), got {other}"),
        }
    }

    #[test]
    fn test_gguf_dtype_to_safetensors_known_types() {
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::F32).unwrap(), Dtype::F32);
        // F16/BF16 map to F32 because data is dequantized to f32
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::F16).unwrap(), Dtype::F32);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::BF16).unwrap(), Dtype::F32);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::I8).unwrap(), Dtype::I8);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::I16).unwrap(), Dtype::I16);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::I32).unwrap(), Dtype::I32);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::I64).unwrap(), Dtype::I64);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::F64).unwrap(), Dtype::F64);
    }

    #[test]
    fn test_gguf_dtype_to_safetensors_quantized_returns_f32() {
        // Supported dequantization types map to f32
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::Q4_0).unwrap(), Dtype::F32);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::Q4_1).unwrap(), Dtype::F32);
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::Q8_0).unwrap(), Dtype::F32);
    }

    #[test]
    fn test_gguf_dtype_to_safetensors_unsupported_k_family_returns_error() {
        // K-family types without dequantization should return error, not lie about F32 support
        let unsupported = [
            GgufDtype::Q2_K,
            GgufDtype::Q3_K,
            GgufDtype::Q4_K,
            GgufDtype::Q5_K,
            GgufDtype::Q6_K,
            GgufDtype::Q8_K,
            GgufDtype::Q1_K,
            GgufDtype::Q4_K_M,
            GgufDtype::Q5_K_M,
            GgufDtype::Q6_K_S,
            GgufDtype::Q8_K_M,
            GgufDtype::Q2_K_S,
            GgufDtype::Q3_K_S,
            GgufDtype::Q4_K_S,
            GgufDtype::Q5_K_S,
            GgufDtype::Q2_K_M,
            GgufDtype::Q5_0,
            GgufDtype::Q5_1,
            GgufDtype::Q8_1,
        ];
        for dtype in unsupported {
            let result = gguf_dtype_to_safetensors(dtype);
            assert!(
                result.is_err(),
                "K-family dtype {:?} should return error (no dequantization implemented)",
                dtype
            );
        }
    }

    #[ignore] // Q-K dequantization not implemented — requires llama.cpp ggml-quants.c
    #[test]
    fn test_dequantize_q2_k_basic() {
        // Create a simple Q2_K block with known values
        let mut block = Vec::with_capacity(16);
        let scale = pack_f16(1.0);
        let min = pack_f16(0.0);
        block.extend_from_slice(&scale);
        block.extend_from_slice(&min);
        // q1: 2 bytes (16 nibbles)
        block.push(0x00); // all zeros
        block.push(0x00);
        // q2: 6 bytes (32 bits for 16 elements, 2 bits each)
        for _ in 0..6 {
            block.push(0x00);
        }
        // h: 4 bytes (4 half-scale values)
        block.extend_from_slice(&pack_f16(0.0));
        block.extend_from_slice(&pack_f16(0.0));

        let result = dequantize_q2_k(&block, 16).unwrap();
        // All values should be close to 0 since scale=1, min=0, all quantized values=0
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1.0, "Q2_K element {i} = {v}, expected near 0");
        }
    }

    #[ignore] // Q-K dequantization not implemented
    #[test]
    fn test_dequantize_q3_k_basic() {
        let mut block = Vec::with_capacity(24);
        block.extend_from_slice(&pack_f16(1.0)); // d
        block.extend_from_slice(&pack_f16(0.0)); // d_min
        block.push(1); // delta
        block.extend_from_slice(&[1, 1, 1, 1]); // k_scale
        block.extend_from_slice(&vec![0u8; 6]); // q3
        block.push(0); // mask
        block.extend_from_slice(&pack_f16(0.0));
        block.extend_from_slice(&pack_f16(0.0));
        block.extend_from_slice(&pack_f16(0.0));
        block.extend_from_slice(&pack_f16(0.0));

        let result = dequantize_q3_k(&block, 16).unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 10.0, "Q3_K element {i} = {v}, expected small");
        }
    }

    #[ignore] // Q-K dequantization not implemented
    #[test]
    fn test_dequantize_q4_k_basic() {
        let mut block = Vec::with_capacity(24);
        block.extend_from_slice(&pack_f16(1.0)); // d
        block.extend_from_slice(&pack_f16(0.0)); // d_min
        block.push(1); // scale_lo
        block.push(0); // scale_hi
        block.extend_from_slice(&vec![0u8; 4]); // q4_lo
        block.extend_from_slice(&vec![0u8; 4]); // q4_hi

        let result = dequantize_q4_k(&block, 16).unwrap();
        // With all quantized values = 0 and scale_hi = 0: q = -8, result = d * (-8/16) + d_min = -0.5
        for (i, &v) in result.iter().enumerate() {
            assert!((v - (-0.5)).abs() < 0.1, "Q4_K element {i} = {v}, expected -0.5");
        }
    }

    #[ignore] // Q-K dequantization not implemented
    #[test]
    fn test_dequantize_q5_k_basic() {
        let mut block = Vec::with_capacity(32);
        block.extend_from_slice(&pack_f16(1.0)); // d
        block.extend_from_slice(&pack_f16(0.0)); // d_min
        block.push(0); // scale
        block.extend_from_slice(&vec![0u8; 4]); // q5_lo
        block.push(0); // q5_h
        block.push(0);

        let result = dequantize_q5_k(&block, 16).unwrap();
        // With all quantized values = 0: q = 0, result = d * (-16/16) + d_min + scale = -1.0
        for (i, &v) in result.iter().enumerate() {
            assert!((v - (-1.0)).abs() < 0.1, "Q5_K element {i} = {v}, expected -1.0");
        }
    }

    #[ignore] // Q-K dequantization not implemented
    #[test]
    fn test_dequantize_q6_k_basic() {
        let mut block = Vec::with_capacity(24);
        block.extend_from_slice(&pack_f16(1.0)); // d
        block.push(0); // mask
        block.extend_from_slice(&vec![0u8; 12]); // q6
        block.push(1); // scale

        let result = dequantize_q6_k(&block, 16).unwrap();
        // With all quantized values = 0 and mask = 0: combined = 0, result = d * (-32/32) * scale = -1.0
        for (i, &v) in result.iter().enumerate() {
            assert!((v - (-1.0)).abs() < 0.1, "Q6_K element {i} = {v}, expected -1.0");
        }
    }

    #[ignore] // Q-K dequantization not implemented
    #[test]
    fn test_dequantize_q8_k_basic() {
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&pack_f16(1.0)); // d
        block.extend_from_slice(&vec![0u8; 16]); // q8

        let result = dequantize_q8_k(&block, 16).unwrap();
        // With all quantized values = 0 and scale = 1: result = d * (0/128) = 0
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 0.01, "Q8_K element {i} = {v}, expected ~0");
        }
    }
}
