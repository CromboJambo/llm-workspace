use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufDtype, GgufHeader, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION_1: u32 = 1;
const GGUF_VERSION_2: u32 = 2;

pub fn parse_gguf(path: &Path) -> Result<GgufHeader, GgufError> {
    let file = std::fs::File::open(path).map_err(|e| GgufError::Io(format!("open {}: {}", path.display(), e)))?;
    let reader = &mut (file as std::fs::File);
    parse_gguf_reader(reader)
}

pub fn parse_gguf_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    // Ensure we're at the start of the file
    let pos_before = reader.seek(std::io::SeekFrom::Start(0))?;
    eprintln!("DEBUG: seek() returned position {}", pos_before);

    let magic = read_bytes(reader, 4)?;
    eprintln!("DEBUG: Magic bytes = {:?}", String::from_utf8_lossy(&magic));
    if magic.as_slice() != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(String::from_utf8_lossy(&magic).to_string()));
    }

    let version = reader.read_u32::<LittleEndian>()?;
    eprintln!("DEBUG: Version u32={}", version);

    match version {
        GGUF_VERSION_1 => parse_v1(reader),
        GGUF_VERSION_2 => parse_v2(reader),
        GGUF_VERSION_3 => parse_v3(reader),
        _ => Err(GgufError::UnsupportedVersion(version)),
    }
}

fn parse_v1<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let kv_count = reader.read_u64::<LittleEndian>()?;
    
    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader)?);
    }
    
    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(1, &kv_pairs, &[], alignment);

    Ok(GgufHeader { version: 1, kv_pairs, tensors: vec![], data_alignment: alignment, data_section_start })
}

fn parse_v2<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(2, &kv_pairs, &tensors, alignment);

    Ok(GgufHeader { version: 2, kv_pairs, tensors, data_alignment: alignment, data_section_start })
}

fn parse_v3<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    // v3 practical format: 8-byte alignment padding after counts (zeros) before KV pairs
    for _ in 0..8 {
        let _ = reader.read_u8()?; // skip alignment padding byte
    }

    eprintln!("parse_v3: tensor_count={}, kv_count={}", tensor_count, kv_count);

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair_v3(reader)?);
    }
    eprintln!("parse_v3: read {} KV pairs", kv_pairs.len());

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info_v3(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, alignment);

    eprintln!("parse_v3: computed data_section_start={}, alignment={:?}", data_section_start, alignment);

    Ok(GgufHeader { version: 3, kv_pairs, tensors, data_alignment: alignment, data_section_start })
}

fn read_kv_pair<R: Read + Seek>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    let key = read_string(reader)?;
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

    let value = read_kv_value(reader, value_type)?;

    Ok(GgufKvPair { key, value_type, value })
}

fn read_string<R: Read + Seek>(reader: &mut R) -> Result<String, GgufError> {
    let len = reader.read_u32::<LittleEndian>()? as u64; // v1/v2 practical: u32 key length, u64 string value (per spec)
    if len > 1024 * 1024u64 { return Err(GgufError::Io(format!("string length {} exceeds max", len))); }
    let bytes = read_bytes(reader, len as usize)?;
    String::from_utf8(bytes).map_err(GgufError::Utf8)
}

fn read_tensor_info<R: Read + Seek>(reader: &mut R) -> Result<GgufTensorInfo, GgufError> {
    let name = read_string(reader)?; // v1/v2 uses u32 for tensor names per spec
    
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }
    
    let dtype = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
}

fn read_alignment_from_kv(kv_pairs: &[GgufKvPair]) -> Option<u64> {
    kv_pairs.iter().find(|p| p.key == "general.alignment").and_then(|p| p.value.as_u64())
}

fn read_bytes<R>(reader: &mut R, len: usize) -> Result<Vec<u8>, GgufError> where R: Read {
    let mut buf = vec![0u8; len]; // Fixed: pre-sized buffer instead of with_capacity
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_kv_pair_v3<R>(reader: &mut R) -> Result<GgufKvPair, GgufError> where R: Read + std::io::Seek {
    use std::io::Read as _;

    // v3 format: [key_bytes][value_type(u32)][string_length(u32) or element_count]
    // After each KV pair's value data, there are 8 bytes of zero-padding (alignment).
    // We need to seek back to the end of the previous value and re-read from the start.

    let mut key_buf = Vec::new();
    let mut in_value_mode = false;
    let mut value_type_bytes = [0u8; 4];
    let mut length_bytes = [0u8; 4];
    let mut element_or_string_len: u32 = 0;

    // Phase 1: Read key bytes until we hit a value_type byte (< 32) or buffer the rest
    loop {
        let b = reader.read_u8()?;

        eprintln!("DEBUG read_kv_pair_v3: byte={:#04x} (value={})", b, b);

        // Value_type detected! Read u32 and then length directly (no alignment between them)
        if !in_value_mode && (b < 32 || b <= 15) {
            eprintln!("DEBUG: Triggered value_type detection at byte 0x{:02x}", b);

            // The trigger byte IS the first byte of value_type u32, so prepend it to buffer
            let mut vtype_bytes = [b, 0u8, 0u8, 0u8];
            reader.read_exact(&mut vtype_bytes[1..4])?;

            let value_type_raw = u32::from_le_bytes(vtype_bytes);
            eprintln!("DEBUG: value_type_raw = {} (expected STRING=8 or TENSOR_COUNT=12)", value_type_raw);

            // Read string length u32
            reader.read_exact(&mut length_bytes)?;
            element_or_string_len = u32::from_le_bytes(length_bytes);
            eprintln!("DEBUG: string_length = {}", element_or_string_len);

            let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

            // Read value based on type and length/count
            let value = match value_type {
                GgufValueType::String => {
                    if element_or_string_len > 0 {
                        let bytes = read_bytes(reader, element_or_string_len as usize)?;
                        eprintln!("DEBUG: read {} bytes for string value (key='{}')", bytes.len(), String::from_utf8_lossy(&key_buf));
                        GgufKvValue::String(String::from_utf8(bytes).map_err(GgufError::Utf8)?)
                    } else {
                        eprintln!("DEBUG: empty string value for key '{}'", String::from_utf8(key_buf.clone()).unwrap_or_default());
                        GgufKvValue::String(String::default())
                    }
                }
                _ => read_kv_value(reader, value_type)?,
            };

            let key = String::from_utf8(key_buf).map_err(GgufError::Utf8)?;
            eprintln!("KV v3: key='{}' type={} len={}", key, value_type_raw, element_or_string_len);
            return Ok(GgufKvPair { key, value_type, value });
        } else if !in_value_mode {
            // Value mode started - read the next 3 bytes of value_type
            in_value_mode = true;
            let vtype_start = b;
            reader.read_exact(&mut value_type_bytes[1..4])?;

            let value_type_raw = u32::from_le_bytes([vtype_start, value_type_bytes[1], value_type_bytes[2], value_type_bytes[3]]);
            eprintln!("DEBUG: value_type_raw (alt) = {} (expected STRING=8 or TENSOR_COUNT=12)", value_type_raw);

            // Read string length u32
            reader.read_exact(&mut length_bytes)?;
            element_or_string_len = u32::from_le_bytes(length_bytes);
            eprintln!("DEBUG: string_length (alt) = {}", element_or_string_len);

            let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

            // Read value based on type and length/count
            let value = match value_type {
                GgufValueType::String => {
                    if element_or_string_len > 0 {
                        let bytes = read_bytes(reader, element_or_string_len as usize)?;
                        eprintln!("DEBUG: read {} bytes for string value (key='{}')", bytes.len(), String::from_utf8_lossy(&key_buf));
                        GgufKvValue::String(String::from_utf8(bytes).map_err(GgufError::Utf8)?)
                    } else {
                        eprintln!("DEBUG: empty string value for key '{}'", String::from_utf8(key_buf.clone()).unwrap_or_default());
                        GgufKvValue::String(String::default())
                    }
                }
                _ => read_kv_value(reader, value_type)?,
            };

            let key = String::from_utf8(key_buf).map_err(GgufError::Utf8)?;
            eprintln!("KV v3 (alt): key='{}' type={} len={}", key, value_type_raw, element_or_string_len);
            return Ok(GgufKvPair { key, value_type, value });
        } else {
            // Printable char - part of the key name
            key_buf.push(b);
        }
    }

    // Should never reach here if format is correct (value_type byte always < 32)
    let key = String::from_utf8(key_buf).map_err(GgufError::Utf8)?;
    Ok(GgufKvPair { key, value_type: GgufValueType::String, value: GgufKvValue::String(String::default()) }) // placeholder fallback
}

fn read_kv_value<R>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> where R: Read {
    match value_type {
        GgufValueType::Uint8 => Ok(GgufKvValue::Uint8(reader.read_u8()?)),
        GgufValueType::Int8 => Ok(GgufKvValue::Int8(reader.read_i8()?)),
        GgufValueType::Float32 => {
            let mut bytes = [0u8; 4]; reader.read_exact(&mut bytes)?;
            Ok(GgufKvValue::Float32(f32::from_le_bytes(bytes)))
        }
        _ => Err(GgufError::InvalidValueType(value_type as u32)), // placeholder for other types
    }
}

fn read_tensor_info_v3<R>(reader: &mut R) -> Result<GgufTensorInfo, GgufError> where R: Read + std::io::Seek {
    let mut name_buf = Vec::new();
    let mut found_name = false;
    let mut name = String::default();

    loop {
        let b = reader.read_u8()?;
        if b < 32 || b <= 15 {
            reader.seek(std::io::SeekFrom::Current(-1))?;
            let mut vtype_bytes = [0u8; 4]; reader.read_exact(&mut vtype_bytes)?;
            let _value_type_raw = u32::from_le_bytes(vtype_bytes);

            let mut length_bytes = [0u8; 4]; reader.read_exact(&mut length_bytes)?;
            let name_len: u64 = u32::from_le_bytes(length_bytes) as u64;

            if name_len > 1024 * 1024u64 { return Err(GgufError::Io(format!("tensor name length {} exceeds max", name_len))); }
            
            let mut tmp_buf = vec![0u8; name_len as usize]; reader.read_exact(&mut tmp_buf)?;
            name = String::from_utf8(tmp_buf).map_err(GgufError::Utf8)?;
            found_name = true;
            break;
        } else {
            name_buf.push(b);
        }
    }

    if !found_name {
        let mut vtype_bytes = [0u8; 4]; reader.read_exact(&mut vtype_bytes)?;
        let _value_type_raw = u32::from_le_bytes(vtype_bytes);
        
        let mut length_bytes = [0u8; 4]; reader.read_exact(&mut length_bytes)?;
        let name_len: u64 = u32::from_le_bytes(length_bytes) as u64;
        
        if name_len > 1024 * 1024u64 { return Err(GgufError::Io(format!("tensor name length {} exceeds max", name_len))); }

        let mut tmp_buf = vec![0u8; name_len as usize]; reader.read_exact(&mut tmp_buf)?;
        name = String::from_utf8(tmp_buf).map_err(GgufError::Utf8)?;
    }

    let n_dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }
    let dtype = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
}

pub fn compute_data_section_start(version: u32, kv_pairs: &[GgufKvPair], tensors: &[GgufTensorInfo], data_alignment: Option<u64>) -> u64 {
    let header_base: u64 = 4 + 4 + 8 + 8; // magic + version + tensor_count + kv_count
    let kv_size: u64 = match version {
        3 => kv_pairs.iter().map(|p| p.raw_byte_size_v3() as u64).sum(),
        _ => kv_pairs.iter().map(|p| p.raw_byte_size() as u64).sum(),
    };
    let tensor_size: u64 = tensors.iter().map(|t| t.raw_byte_size() as u64).sum();
    let mut data_section = header_base.checked_add(kv_size).and_then(|v| v.checked_add(tensor_size)).unwrap_or(u64::MAX);

    if version == 3 {
        if let Some(alignment) = data_alignment {
            if alignment > 0 {
                let remainder = data_section % alignment;
                if remainder != 0 { data_section += alignment - remainder; }
            }
        }
    }

    data_section
}

pub fn extract_tensor_bytes<R>(reader: &mut R, dtype: u32, element_count: u64, offset: u64, data_section_start: u64) -> Result<Vec<u8>, GgufError> where R: Read {
    let _ = (dtype, element_count); // placeholder - will be implemented later
    Ok(vec![])
}

pub fn extract_tensor_bytes_from<R>(reader: &mut R, dtype: u32, element_count: u64, offset: u64, data_section_start: u64) -> Result<Vec<u8>, GgufError> where R: Read {
    let _ = (dtype, element_count); // placeholder
    Ok(vec![])
}

pub fn tensor_bytes_for_dtype(dtype: u32, element_count: u64) -> usize {
    let _ = dtype; // placeholder
    element_count as usize * 4 // rough guess for now
}

#[cfg(test)]
mod tests_real_file {
    use super::*;
    
    #[test]
    fn test_parse_conformance_corpus_qwen2_5() {
        let path = Path::new("/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");

        // Should parse without error
        let header = parse_gguf(path).expect("Failed to parse real GGUF file");

        eprintln!("Header version: {}", header.version);
        assert_eq!(header.version, 3);
        
        // Should have KV pairs
        assert!(header.kv_pairs.len() > 0, "Should have KV pairs");

        // Check a specific key exists
        let has_architecture = header.kv_pairs.iter().any(|p| p.key == "general.architecture");
        assert!(has_architecture, "Should have general.architecture KV pair");
        
        eprintln!("SUCCESS: Real GGUF file parsed correctly!");
    }
}
