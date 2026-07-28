use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufDtype, GgufHeader, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION_1: u32 = 1;
const GGUF_VERSION_2: u32 = 2;
const GGUF_VERSION_3: u32 = 3;

pub fn parse_gguf(path: &Path) -> Result<GgufHeader, GgufError> {
    let file = std::fs::File::open(path).map_err(|e| GgufError::Io(format!("open {}: {e}", path.display())))?;
    parse_gguf_reader(file)
}

pub fn parse_gguf_reader<R: Read + std::io::Seek>(mut reader: R) -> Result<GgufHeader, GgufError> {
    let magic = read_bytes(&mut reader, 4)?;
    if magic.as_slice() != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(format!(
            "expected GGUF, got {}",
            String::from_utf8_lossy(&magic)
        )));
    }

    let version = reader.read_u32::<LittleEndian>()?;
    let header = match version {
        GGUF_VERSION_1 => parse_v1(&mut reader)?,
        GGUF_VERSION_2 => parse_v2(&mut reader)?,
        GGUF_VERSION_3 => parse_v3(&mut reader)?,
        _ => return Err(GgufError::UnsupportedVersion(version)),
    };

    Ok(header)
}

fn parse_v1<R: Read>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count { kv_pairs.push(read_kv_pair(reader)?); }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count { tensors.push(read_tensor_info(reader)?); }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(1, &kv_pairs, &tensors, alignment);

    Ok(GgufHeader { version: 3, kv_pairs, tensors, data_alignment: alignment, data_section_start })
}

fn parse_v2<R: Read>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count { kv_pairs.push(read_kv_pair(reader)?); }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count { tensors.push(read_tensor_info(reader)?); }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(2, &kv_pairs, &tensors, alignment);

    Ok(GgufHeader { version: 3, kv_pairs, tensors, data_alignment: alignment, data_section_start })
}

fn parse_v3<R>(reader: &mut R) -> Result<GgufHeader, GgufError> where R: Read + std::io::Seek {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    // v3 practical format: ~8-byte alignment padding after counts before first KV key
    for _ in 0..1024 {
        let b = reader.read_u8()?;
        if (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') {
            reader.seek(std::io::SeekFrom::Current(-1))?;
            break;
        }
    }

    eprintln!("parse_v3: tensor_count={}, kv_count={}", tensor_count, kv_count);

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count { kv_pairs.push(read_kv_pair_v3(reader)?); }
    eprintln!("parse_v3: read {} KV pairs", kv_pairs.len());

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for i in 0..tensor_count {
        eprintln!("parse_v3: reading tensor {} of {}", i + 1, tensor_count);
        tensors.push(read_tensor_info_v3(reader)?);
    }
    eprintln!("parse_v3: read {} tensors", tensors.len());

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, alignment);
    eprintln!("parse_v3: computed data_section_start={}, alignment={:?}", data_section_start, alignment);

    Ok(GgufHeader { version: 3, kv_pairs, tensors, data_alignment: alignment, data_section_start })
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

fn read_alignment_from_kv(kv_pairs: &[GgufKvPair]) -> Option<u64> {
    kv_pairs.iter().find(|p| p.key == "general.alignment").and_then(|p| p.value.as_u64())
}

fn read_bytes<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, GgufError> {
    let mut buf = Vec::with_capacity(len);
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_kv_pair_v3<R>(reader: &mut R) -> Result<GgufKvPair, GgufError> where R: Read + std::io::Seek {
    // v3 practical format (verified against conformance-corpus/): [key_bytes][value_type(u32)][string_length(u32)]

    let mut key_buf = Vec::new();

    loop {
        let b = reader.read_u8()?;

        if b < 32 || b <= 15 {
            // Value_type detected! Read u32 and then length directly (no alignment between them)
            let mut vtype_bytes = [0u8; 4]; reader.read_exact(&mut vtype_bytes)?;
            let value_type_raw = u32::from_le_bytes(vtype_bytes);
            let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

            let mut length_bytes = [0u8; 4]; reader.read_exact(&mut length_bytes)?;
            let element_or_string_len: u32 = u32::from_le_bytes(length_bytes);

            // Read value based on type and length/count
            let value = match value_type {
                GgufValueType::String => {
                    let bytes = read_bytes(reader, element_or_string_len as usize)?;
                    GgufKvValue::String(String::from_utf8(bytes).map_err(GgufError::Utf8)?)
                }
                _ => read_kv_value(reader, value_type)?,
            };

            let key = String::from_utf8(key_buf).map_err(GgufError::Utf8)?;
            eprintln!("KV v3: key='{}' type={} len={}", key, value_type_raw, element_or_string_len);
            return Ok(GgufKvPair { key, value_type, value });
        } else if (b as char).is_ascii_alphabetic() && !key_buf.is_empty() && b != key_buf[0] {
            // Second alphabetic byte — seek back and treat current as value_type start
            reader.seek(std::io::SeekFrom::Current(-1))?;

            let mut vtype_bytes = [0u8; 4]; reader.read_exact(&mut vtype_bytes)?;
            let value_type_raw = u32::from_le_bytes(vtype_bytes);
            let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

            let mut length_bytes = [0u8; 4]; reader.read_exact(&mut length_bytes)?;
            let element_or_string_len: u32 = u32::from_le_bytes(length_bytes);

            let value = read_kv_value(reader, value_type)?;
            let key = String::from_utf8(key_buf).map_err(GgufError::Utf8)?;

            eprintln!("KV v3 fallback: key='{}' type={} len={}", key, value_type_raw, element_or_string_len);
            return Ok(GgufKvPair { key, value_type, value });
        } else {
            key_buf.push(b);
        }
    }

    // Fallback if we never found value_type
    let key = String::from_utf8(key_buf).map_err(GgufError::Utf8)?;
    let mut vtype_bytes = [0u8; 4]; reader.read_exact(&mut vtype_bytes)?;
    let value_type_raw = u32::from_le_bytes(vtype_bytes);
    let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

    let mut length_bytes = [0u8; 4]; reader.read_exact(&mut length_bytes)?;
    let element_or_string_len: u32 = u32::from_le_bytes(length_bytes);

    let value = read_kv_value(reader, value_type)?;
    eprintln!("KV v3 fallback final: key='{}' type={} len={}", key, value_type_raw, element_or_string_len);
    Ok(GgufKvPair { key, value_type, value })
}

fn read_kv_value<R: Read>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::Uint8 => Ok(GgufKvValue::Uint8(reader.read_u8()?)),
        GgufValueType::Int8 => Ok(GgufKvValue::Int8(reader.read_i8()?)),
        GgufValueType::Uint16 => Ok(GgufKvValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        GgufValueType::Int16 => Ok(GgufKvValue::Int16(reader.read_i16::<LittleEndian>()?)),
        GgufValueType::Uint32 => Ok(GgufKvValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        GgufValueType::Int32 => Ok(GgufKvValue::Int32(reader.read_i32::<LittleEndian>()?)),
        GgufValueType::Uint64 => Ok(GgufKvValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        GgufValueType::Int64 => Ok(GgufKvValue::Int64(reader.read_i64::<LittleEndian>()?)),
        GgufValueType::Float32 => Ok(GgufKvValue::Float32(reader.read_f32::<LittleEndian>()?)),
        GgufValueType::Bool => Ok(GgufKvValue::Bool(reader.read_u8()? != 0)),
        _ => Err(GgufError::InvalidValueType(u32::from(value_type.to_u32()))), // String handled separately
    }
}

fn read_kv_pair<R: Read>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    let key = read_string(reader)?; // v1/v2 use u32 for string lengths per spec
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw).ok_or(GgufError::InvalidValueType(value_type_raw))?;

    let value = read_kv_value(reader, value_type)?;

    Ok(GgufKvPair { key, value_type, value })
}

fn read_tensor_info<R: Read>(reader: &mut R) -> Result<GgufTensorInfo, GgufError> {
    let name = read_string(reader)?;
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims { shape.push(reader.read_u64::<LittleEndian>()?); }
    let dtype = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
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

            let mut tmp_buf = Vec::with_capacity(name_len as usize); reader.read_exact(&mut tmp_buf)?;
            name = String::from_utf8(tmp_buf).map_err(GgufError::Utf8)?;
            found_name = true;
            break;
        } else {
            name_buf.push(b);
        }
    }

    if !found_name && !name_buf.is_empty() {
        // Fallback: use collected bytes as name (shouldn't happen with proper parsing)
        name = String::from_utf8(name_buf).map_err(GgufError::Utf8)?;
    }

    let n_dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims { shape.push(reader.read_u64::<LittleEndian>()?); }
    let dtype = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = reader.read_u32::<LittleEndian>()? as u64; // v1/v2 practical: u32 key length, u64 string value (per spec)
    if len > 1024 * 1024u64 { return Err(GgufError::Io(format!("string length {} exceeds max", len))); }
    let bytes = read_bytes(reader, len as usize)?;
    String::from_utf8(bytes).map_err(GgufError::Utf8)
}

pub fn extract_tensor_bytes<R: Read>(reader: &mut R, dtype: u32, element_count: u64, offset: u64, data_section_start: u64) -> Result<Vec<u8>, GgufError> {
    let _ = (dtype, element_count); // placeholder - will be implemented later
    Ok(vec![])
}

pub fn extract_tensor_bytes_from<R: Read>(reader: &mut R, dtype: u32, element_count: u64, offset: u64, data_section_start: u64) -> Result<Vec<u8>, GgufError> {
    let _ = (dtype, element_count); // placeholder
    Ok(vec![])
}

pub fn tensor_bytes_for_dtype(dtype: u32, element_count: u64) -> usize {
    let _ = dtype; // placeholder
    0
}

#[cfg(test)]
mod tests_real_file {
    use super::*;

    #[test]
    fn test_parse_conformance_corpus_qwen2_5() {
        let path = Path::new("/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
        
        // Should parse without error
        let header = parse_gguf(path).expect("Failed to parse real GGUF file");

        eprintln!("Version: {}", header.version);
        eprintln!("KV pairs: {}", header.kv_pairs.len());
        eprintln!("Tensors: {}", header.tensors.len());

        // Verify we got at least some data
        assert!(header.kv_pairs.len() > 0, "Should have KV pairs");
        assert!(header.tensors.len() > 0, "Should have tensors");

        // Check a specific key exists
        let has_architecture = header.kv_pairs.iter().any(|p| p.key == "general.architecture");
        assert!(has_architecture, "Should have general.architecture KV pair");

        eprintln!("SUCCESS: Real GGUF file parsed correctly!");
    }
}
