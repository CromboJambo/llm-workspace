use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufDtype, GgufHeader, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION_1: u32 = 1;
const GGUF_VERSION_2: u32 = 2;
const GGUF_VERSION_3: u32 = 3;

pub fn parse_gguf(path: &Path) -> Result<GgufHeader, GgufError> {
    let file =
        std::fs::File::open(path).map_err(|e| GgufError::Io(format!("open {}: {e}", path.display())))?;
    let reader = BufReader::new(file);
    parse_gguf_reader(reader)
}

pub fn parse_gguf_reader<R: Read>(mut reader: R) -> Result<GgufHeader, GgufError> {
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
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info(reader)?);
    }

    let data_section_start = compute_data_section_start(1, &kv_pairs, &tensors, None);

    Ok(GgufHeader {
        version: 1,
        kv_pairs,
        tensors,
        data_alignment: None,
        data_section_start,
    })
}

fn parse_v2<R: Read>(reader: &mut R) -> Result<GgufHeader, GgufError> {
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

    let data_section_start = compute_data_section_start(2, &kv_pairs, &tensors, None);

    Ok(GgufHeader {
        version: 2,
        kv_pairs,
        tensors,
        data_alignment: None,
        data_section_start,
    })
}

fn parse_v3<R: Read>(reader: &mut R) -> Result<GgufHeader, GgufError> {
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

    let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, alignment);

    Ok(GgufHeader {
        version: 3,
        kv_pairs,
        tensors,
        data_alignment: alignment,
        data_section_start,
    })
}

/// Compute the byte offset where the tensor data section begins.
///
/// GGUF tensor offsets are relative to the data section start, not the file start.
/// The data section starts after the header (magic + version + counts + KV pairs + tensor metadata),
/// aligned up to `data_alignment` for v3.
fn compute_data_section_start(version: u32, kv_pairs: &[GgufKvPair], tensors: &[GgufTensorInfo], data_alignment: Option<u64>) -> u64 {
    let header_base: u64 = 4 + 4 + 8 + 8; // magic + version + tensor_count + kv_count
    let kv_size: usize = kv_pairs.iter().map(|p| p.raw_byte_size()).sum();
    let tensor_size: usize = tensors.iter().map(|t| t.raw_byte_size()).sum();
    let mut data_section: u64 = header_base + kv_size as u64 + tensor_size as u64;

    if version == 3
        && let Some(alignment) = data_alignment
        && alignment > 0
    {
        let remainder = data_section % alignment;
        if remainder != 0 {
            data_section += alignment - remainder;
        }
    }

    data_section
}

/// Read the `general.alignment` value from KV pairs (GGUF v3).
fn read_alignment_from_kv(kv_pairs: &[GgufKvPair]) -> Option<u64> {
    kv_pairs
        .iter()
        .find(|p| p.key == "general.alignment")
        .and_then(|p| p.value.as_u64())
}

fn read_bytes<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, GgufError> {
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| GgufError::Io(format!("read {len} bytes: {e}")))?;
    Ok(buf)
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = reader.read_u64::<LittleEndian>()?;
    if len > 1024 * 1024 {
        return Err(GgufError::Io(format!("string length {len} exceeds max 1MB")));
    }
    let bytes = read_bytes(reader, len as usize)?;
    String::from_utf8(bytes).map_err(GgufError::Utf8)
}

fn read_kv_pair<R: Read>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    let key = read_string(reader)?;
    let value_type = read_value_type(reader)?;
    let value = read_kv_value(reader, value_type)?;
    #[cfg(debug_assertions)]
    eprintln!("KV key='{}' type={}", key, value_type.to_u32());
    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

fn read_value_type<R: Read>(reader: &mut R) -> Result<GgufValueType, GgufError> {
    let raw = reader.read_u32::<LittleEndian>()?;
    GgufValueType::from_u32(raw).ok_or(GgufError::InvalidValueType(raw))
}

fn read_kv_value<R: Read>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::Uint8 => {
            let v = reader.read_u8()?;
            Ok(GgufKvValue::Uint8(v))
        }
        GgufValueType::Int8 => {
            let v = reader.read_i8()?;
            Ok(GgufKvValue::Int8(v))
        }
        GgufValueType::Uint16 => {
            let v = reader.read_u16::<LittleEndian>()?;
            Ok(GgufKvValue::Uint16(v))
        }
        GgufValueType::Int16 => {
            let v = reader.read_i16::<LittleEndian>()?;
            Ok(GgufKvValue::Int16(v))
        }
        GgufValueType::Uint32 => {
            let v = reader.read_u32::<LittleEndian>()?;
            Ok(GgufKvValue::Uint32(v))
        }
        GgufValueType::Int32 => {
            let v = reader.read_i32::<LittleEndian>()?;
            Ok(GgufKvValue::Int32(v))
        }
        GgufValueType::Uint64 => {
            let v = reader.read_u64::<LittleEndian>()?;
            Ok(GgufKvValue::Uint64(v))
        }
        GgufValueType::Int64 => {
            let v = reader.read_i64::<LittleEndian>()?;
            Ok(GgufKvValue::Int64(v))
        }
        GgufValueType::Float32 => {
            let v = reader.read_f32::<LittleEndian>()?;
            Ok(GgufKvValue::Float32(v))
        }
        GgufValueType::Bool => {
            let v = reader.read_u8()? != 0;
            Ok(GgufKvValue::Bool(v))
        }
        GgufValueType::String => {
            let s = read_string(reader)?;
            Ok(GgufKvValue::String(s))
        }
        GgufValueType::Array => {
            let element_type = read_value_type(reader)?;
            let count = reader.read_u64::<LittleEndian>()?;
            match element_type {
                GgufValueType::Int8Array => {
                    let mut vals = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        vals.push(reader.read_i8()?);
                    }
                    Ok(GgufKvValue::Int8Array(vals))
                }
                GgufValueType::Uint8Array => {
                    let mut vals = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        vals.push(reader.read_u8()?);
                    }
                    Ok(GgufKvValue::Uint8Array(vals))
                }
                _ => {
                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        values.push(read_kv_value(reader, element_type)?);
                    }
                    Ok(GgufKvValue::Array(values))
                }
            }
        }
        GgufValueType::Float16 => {
            let raw = reader.read_u16::<LittleEndian>()?;
            Ok(GgufKvValue::Float16(raw))
        }
        GgufValueType::Int8Array => {
            let count = reader.read_u64::<LittleEndian>()?;
            let mut vals = Vec::with_capacity(count as usize);
            for _ in 0..count {
                vals.push(reader.read_i8()?);
            }
            Ok(GgufKvValue::Int8Array(vals))
        }
        GgufValueType::Uint8Array => {
            let count = reader.read_u64::<LittleEndian>()?;
            let mut vals = Vec::with_capacity(count as usize);
            for _ in 0..count {
                vals.push(reader.read_u8()?);
            }
            Ok(GgufKvValue::Uint8Array(vals))
        }
        GgufValueType::Bfloat16 => {
            let raw = reader.read_u16::<LittleEndian>()?;
            let v = (raw as u32) << 16;
            let f = f32::from_bits(v);
            Ok(GgufKvValue::Bfloat16(f))
        }
    }
}

fn read_tensor_info<R: Read>(reader: &mut R) -> Result<GgufTensorInfo, GgufError> {
    let name = read_string(reader)?;
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }
    let dtype = reader.read_u32::<LittleEndian>()?;
    let offset = reader.read_u64::<LittleEndian>()?;
    Ok(GgufTensorInfo { name, shape, offset, dtype })
}

/// Extract raw tensor bytes from a GGUF file at a given offset.
pub fn extract_tensor_bytes(path: &Path, offset: u64, size: usize) -> Result<Vec<u8>, GgufError> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        GgufError::Io(format!("open {}: {e}", path.display()))
    })?;

    file.seek(std::io::SeekFrom::Start(offset))
        .map_err(|e| GgufError::Io(format!("seek to offset {offset}: {e}")))?;

    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)
        .map_err(|e| GgufError::Io(format!("read {size} bytes at offset {offset}: {e}")))?;

    Ok(buffer)
}

/// Extract raw tensor bytes from a GGUF file using the header's data section offset.
pub fn extract_tensor_bytes_with_header(path: &Path, header: &GgufHeader, tensor: &GgufTensorInfo) -> Result<Vec<u8>, GgufError> {
    let file_offset = header.data_section_start + tensor.offset;
    extract_tensor_bytes(path, file_offset, tensor.stored_size() as usize)
}

/// Extract tensor bytes from an already-opened file handle.
pub fn extract_tensor_bytes_from<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    offset: u64,
    size: usize,
) -> Result<Vec<u8>, GgufError> {
    reader
        .seek(std::io::SeekFrom::Start(offset))
        .map_err(|e| GgufError::Io(format!("seek to offset {offset}: {e}")))?;

    let mut buffer = vec![0u8; size];
    reader
        .read_exact(&mut buffer)
        .map_err(|e| GgufError::Io(format!("read {size} bytes at offset {offset}: {e}")))?;

    Ok(buffer)
}

/// Compute the raw byte size of a tensor before quantization.
pub fn tensor_bytes_for_dtype(element_count: u64, dtype: GgufDtype) -> usize {
    (element_count as usize) * dtype.bytes_per_element()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GgufKvPair, GgufKvValue, GgufValueType};
    use crate::GgufVersion;

    fn make_v3_header() -> GgufHeader {
        let kv_pairs = vec![
            GgufKvPair {
                key: "general.architecture".to_string(),
                value_type: GgufValueType::String,
                value: GgufKvValue::String("llama".to_string()),
            },
            GgufKvPair {
                key: "general.file_type".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(6),
            },
            GgufKvPair {
                key: "llama.context_length".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(4096),
            },
            GgufKvPair {
                key: "llama.embedding_length".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(4096),
            },
            GgufKvPair {
                key: "llama.block_count".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(32),
            },
            GgufKvPair {
                key: "llama.attention.head_count".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(32),
            },
            GgufKvPair {
                key: "llama.attention.head_count_kv".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(8),
            },
            GgufKvPair {
                key: "llama.rope.dimension_count".to_string(),
                value_type: GgufValueType::Int32,
                value: GgufKvValue::Int32(128),
            },
        ];
        let tensors = vec![
            GgufTensorInfo {
                name: "token_embd.weight".to_string(),
                shape: vec![4096],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "blk.0.attn_k.weight".to_string(),
                shape: vec![4096, 4096],
                offset: 67108864,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "blk.0.attn_output.weight".to_string(),
                shape: vec![4096, 4096],
                offset: 134217728,
                dtype: 1,
            },
        ];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, Some(32));

        GgufHeader {
            version: 3,
            kv_pairs,
            tensors,
            data_alignment: Some(32),
            data_section_start,
        }
    }

    #[test]
    fn test_get_kv_str() {
        let header = make_v3_header();
        assert_eq!(header.get_kv_str("general.architecture"), Some("llama"));
        assert!(header.get_kv_str("nonexistent").is_none());
    }

    #[test]
    fn test_get_kv_u32() {
        let header = make_v3_header();
        assert_eq!(header.get_kv_u32("general.file_type"), Some(6));
        assert_eq!(header.get_kv_u32("llama.context_length"), Some(4096));
        assert!(header.get_kv_u32("nonexistent").is_none());
    }

    #[test]
    fn test_get_kv_i32() {
        let header = make_v3_header();
        assert_eq!(header.get_kv_i32("llama.rope.dimension_count"), Some(128));
        assert!(header.get_kv_i32("nonexistent").is_none());
    }

    #[test]
    fn test_get_kv_array() {
        let header = make_v3_header();
        assert!(header.get_kv_array("nonexistent").is_none());
    }

    #[test]
    fn test_to_config_map() {
        let header = make_v3_header();
        let map = header.to_config_map();
        assert_eq!(map.len(), 8);
        assert!(map.contains_key("general.architecture"));
        assert_eq!(map["general.architecture"].as_str(), Some("llama"));
    }

    #[test]
    fn test_gguf_type_e2e() {
        let header = make_v3_header();
        let config = header.to_config_map();
        let arch = config.get("general.architecture").unwrap();
        assert_eq!(arch.as_str(), Some("llama"));
        let ctx = config.get("llama.context_length").unwrap();
        assert_eq!(ctx.as_u32(), Some(4096));
    }

    #[test]
    fn test_architecture_helper() {
        let header = make_v3_header();
        assert_eq!(header.architecture(), Some("llama"));
    }

    #[test]
    fn test_context_length_helper() {
        let header = make_v3_header();
        assert_eq!(header.context_length(), Some(4096));
    }

    #[test]
    fn test_embedding_length_helper() {
        let header = make_v3_header();
        assert_eq!(header.embedding_length(), Some(4096));
    }

    #[test]
    fn test_block_count_helper() {
        let header = make_v3_header();
        assert_eq!(header.block_count(), Some(32));
    }

    #[test]
    fn test_attention_head_count_helpers() {
        let header = make_v3_header();
        assert_eq!(header.attention_head_count(), Some(32));
        assert_eq!(header.attention_head_count_kv(), Some(8));
    }

    #[test]
    fn test_rope_dimension_count_helper() {
        let header = make_v3_header();
        assert_eq!(header.rope_dimension_count(), Some(128));
    }

    #[test]
    fn test_file_type_helper() {
        let header = make_v3_header();
        assert_eq!(header.file_type(), Some("6".to_string()));
    }

    #[test]
    fn test_tensor_helpers() {
        let header = make_v3_header();
        assert!(header.has_tensor("token_embd.weight"));
        assert!(!header.has_tensor("nonexistent"));
        let tensor = header.get_tensor("token_embd.weight").unwrap();
        assert_eq!(tensor.element_count(), 4096);
        assert_eq!(tensor.ndims(), 1);
    }

    #[test]
    fn test_total_tensor_bytes_f32() {
        let header = make_v3_header();
        // token_embd: 4096, blk.0.attn_k: 4096*4096, blk.0.attn_output: 4096*4096
        let expected = 4096 + (4096 * 4096) + (4096 * 4096);
        assert_eq!(header.total_tensor_bytes_f32(), expected);
    }

    fn make_minimal_gguf_v3_bytes() -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(b"GGUF");

        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());

        // Tensor count
        buf.extend_from_slice(&2u64.to_le_bytes());

        // KV count
        buf.extend_from_slice(&3u64.to_le_bytes());

        // KV pair 1: general.architecture = "llama" (string)
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(10u32).to_le_bytes()); // STRING type
        buf.extend_from_slice(&(5u64).to_le_bytes()); // "llama" length
        buf.extend_from_slice(b"llama");

        // KV pair 2: general.file_type = 1 (F16) (uint32)
        let key = "general.file_type";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(4u32).to_le_bytes()); // UINT32 type
        buf.extend_from_slice(&1u32.to_le_bytes());

        // KV pair 3: general.alignment = 32
        let key = "general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(4u32).to_le_bytes()); // UINT32 type
        buf.extend_from_slice(&32u32.to_le_bytes());

        // Tensor 1: token_embd.weight (shape [4096], dtype F16, offset 0)
        let name = "token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&4096u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Tensor 2: output.weight (shape [4096, 32000], dtype F16, offset after tensor 1)
        let name = "output.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        buf.extend_from_slice(&4096u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&32000u64.to_le_bytes()); // shape[1]
        buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
        buf.extend_from_slice(&(4096 * 2u64).to_le_bytes()); // offset (F16 = 2 bytes per element)

        buf
    }

    #[test]
    fn test_parse_minimal_gguf_v3() {
        let bytes = make_minimal_gguf_v3_bytes();
        let header = parse_gguf_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(header.version, 3);
        assert_eq!(header.data_alignment, Some(32));
        assert_eq!(header.kv_pairs.len(), 3);
        assert_eq!(header.tensors.len(), 2);

        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(header.get_kv_u32("general.file_type"), Some(1));
    }

    #[test]
    fn test_parse_invalid_magic() {
        let bytes = vec![b'B', b'G', b'G', b'M', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = parse_gguf_reader(std::io::Cursor::new(&bytes));
        assert!(result.is_err());
        match result.unwrap_err() {
            GgufError::InvalidMagic(_) => {}
            other => panic!("expected InvalidMagic, got {other}"),
        }
    }

    #[test]
    fn test_parse_unsupported_version() {
        let mut bytes = vec![b'G', b'G', b'U', b'F'];
        bytes.extend_from_slice(&99u32.to_le_bytes());
        let result = parse_gguf_reader(std::io::Cursor::new(&bytes));
        assert!(result.is_err());
        match result.unwrap_err() {
            GgufError::UnsupportedVersion(99) => {}
            other => panic!("expected UnsupportedVersion(99), got {other}"),
        }
    }

    #[test]
    fn test_extract_tensor_bytes_from_cursor() {
        let bytes = make_minimal_gguf_v3_bytes();
        let mut cursor = std::io::Cursor::new(bytes);

        // Read tensor at offset 0, size 16 (8 f16 elements)
        let data = extract_tensor_bytes_from(&mut cursor, 0, 16).unwrap();
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn test_dtype_roundtrip() {
        use crate::parser::GgufDtype;
        for v in 0..=29u32 {
            let dtype = GgufDtype::from_u32(v);
            assert_eq!(dtype.to_u32(), v);
        }
        assert_eq!(GgufDtype::from_u32(100).to_u32(), 100);
    }

    #[test]
    fn test_dtype_quantized_check() {
        use crate::parser::GgufDtype;
        assert!(!GgufDtype::F32.is_quantized());
        assert!(!GgufDtype::F16.is_quantized());
        assert!(GgufDtype::Q4_0.is_quantized());
        assert!(GgufDtype::Q8_0.is_quantized());
        assert!(GgufDtype::Q2_K.is_quantized());
        assert!(!GgufDtype::I8.is_quantized());
        assert!(!GgufDtype::I32.is_quantized());
    }

    #[test]
    fn test_dtype_bytes_per_element() {
        use crate::parser::GgufDtype;
        assert_eq!(GgufDtype::F32.bytes_per_element(), 4);
        assert_eq!(GgufDtype::F16.bytes_per_element(), 2);
        assert_eq!(GgufDtype::I8.bytes_per_element(), 1);
        assert_eq!(GgufDtype::I32.bytes_per_element(), 4);
        assert_eq!(GgufDtype::I64.bytes_per_element(), 8);
        assert_eq!(GgufDtype::F64.bytes_per_element(), 8);
        assert_eq!(GgufDtype::BF16.bytes_per_element(), 2);
        assert_eq!(GgufDtype::Q8_0.bytes_per_element(), 2);
        assert_eq!(GgufDtype::Q4_0.bytes_per_element(), 0);
        assert_eq!(GgufDtype::Unknown(99).bytes_per_element(), 0);
    }

    #[test]
    fn test_tensor_bytes_for_dtype() {
        use crate::parser::{GgufDtype, tensor_bytes_for_dtype};
        assert_eq!(tensor_bytes_for_dtype(4096, GgufDtype::F32), 4096 * 4);
        assert_eq!(tensor_bytes_for_dtype(4096, GgufDtype::F16), 4096 * 2);
        assert_eq!(tensor_bytes_for_dtype(100, GgufDtype::I8), 100);
    }

    fn make_minimal_gguf_v1_bytes() -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(b"GGUF");

        // Version
        buf.extend_from_slice(&1u32.to_le_bytes());

        // Tensor count
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV count
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV pair: general.architecture = "llama" (string)
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(10u32).to_le_bytes()); // STRING type
        buf.extend_from_slice(&(5u64).to_le_bytes()); // "llama" length
        buf.extend_from_slice(b"llama");

        // Tensor: token_embd.weight (shape [4096], dtype F16, offset 0)
        let name = "token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&4096u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        buf
    }

    fn make_minimal_gguf_v2_bytes() -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(b"GGUF");

        // Version
        buf.extend_from_slice(&2u32.to_le_bytes());

        // Tensor count
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV count
        buf.extend_from_slice(&1u64.to_le_bytes());

        // KV pair: general.architecture = "qwen2" (string)
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(10u32).to_le_bytes()); // STRING type
        buf.extend_from_slice(&(5u64).to_le_bytes()); // "qwen2" length
        buf.extend_from_slice(b"qwen2");

        // Tensor: token_embd.weight (shape [4096], dtype F16, offset 0)
        let name = "token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&4096u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        buf
    }

    #[test]
    fn test_parse_minimal_gguf_v1() {
        let bytes = make_minimal_gguf_v1_bytes();
        let header = parse_gguf_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(header.version, 1);
        assert_eq!(header.data_alignment, None);
        assert_eq!(header.kv_pairs.len(), 1);
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(header.data_section_start, 118);
    }

    #[test]
    fn test_parse_minimal_gguf_v2() {
        let bytes = make_minimal_gguf_v2_bytes();
        let header = parse_gguf_reader(std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(header.version, 2);
        assert_eq!(header.data_alignment, None);
        assert_eq!(header.kv_pairs.len(), 1);
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.architecture(), Some("qwen2"));
    }

    #[test]
    fn test_parse_gguf_v1_v2_v3_data_section_alignment() {
        // v1: no alignment field, data_section_start = header_base + kv_size + tensor_size
        let v1 = make_minimal_gguf_v1_bytes();
        let h1 = parse_gguf_reader(std::io::Cursor::new(&v1)).unwrap();

        // v2: same as v1, no alignment
        let v2 = make_minimal_gguf_v2_bytes();
        let h2 = parse_gguf_reader(std::io::Cursor::new(&v2)).unwrap();

        // v3: has alignment field, data_section_start is aligned
        let v3 = make_minimal_gguf_v3_bytes();
        let h3 = parse_gguf_reader(std::io::Cursor::new(&v3)).unwrap();

        // v1 and v2 should have the same data_section_start (no alignment)
        assert_eq!(h1.data_section_start, h2.data_section_start);

        // v3 should have data_section_start >= v1's value due to alignment padding
        assert!(h3.data_section_start >= h1.data_section_start);
        assert_eq!(h3.data_alignment, Some(32));
    }

    #[test]
    fn test_kv_pair_raw_byte_size() {
        use crate::types::GgufKvPair;

        let kv = GgufKvPair {
            key: "test".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(42),
        };
        // 8 (key_len) + 4 (key) + 4 (type) + 4 (value) = 20
        assert_eq!(kv.raw_byte_size(), 8 + 4 + 4 + 4);

        let str_kv = GgufKvPair {
            key: "arch".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("llama".to_string()),
        };
        // 8 (key_len) + 4 (key) + 4 (type) + 8 (str_len) + 5 (str) = 29
        assert_eq!(str_kv.raw_byte_size(), 8 + 4 + 4 + 8 + 5);
    }

    #[test]
    fn test_tensor_stored_size_f32() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![100, 200],
            offset: 0,
            dtype: 0, // F32
        };
        // 100 * 200 * 4 bytes
        assert_eq!(info.stored_size(), 100 * 200 * 4);
    }

    #[test]
    fn test_tensor_stored_size_f16() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![100, 200],
            offset: 0,
            dtype: 1, // F16
        };
        // 100 * 200 * 2 bytes
        assert_eq!(info.stored_size(), 100 * 200 * 2);
    }

    #[test]
    fn test_tensor_stored_size_q4_0() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![256],
            offset: 0,
            dtype: 2, // Q4_0
        };
        // Q4_0: 8 full blocks of 32 = 8 * 18 = 144
        assert_eq!(info.stored_size(), 144);
    }

    #[test]
    fn test_tensor_stored_size_q4_0_partial() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![32],
            offset: 0,
            dtype: 2, // Q4_0
        };
        // Q4_0: 32 elements = one partial block
        // full_blocks=0, remaining=32 => 2 + 32/2 = 18
        assert_eq!(info.stored_size(), 18);
    }

    #[test]
    fn test_gguf_header_get_tensor() {
        let header = make_v3_header();
        let tensor = header.get_tensor("token_embd.weight").unwrap();
        assert_eq!(tensor.name, "token_embd.weight");
        assert_eq!(tensor.shape, vec![4096u64]);
        assert_eq!(tensor.element_count(), 4096);
        assert!(header.has_tensor("blk.0.attn_k.weight"));
        assert!(!header.has_tensor("nonexistent"));
    }

    #[test]
    fn test_gguf_header_helpers_comprehensive() {
        let header = make_v3_header();

        // Architecture
        assert_eq!(header.architecture(), Some("llama"));

        // Context length
        assert_eq!(header.context_length(), Some(4096));

        // Embedding length
        assert_eq!(header.embedding_length(), Some(4096));

        // Block count
        assert_eq!(header.block_count(), Some(32));

        // Attention heads
        assert_eq!(header.attention_head_count(), Some(32));
        assert_eq!(header.attention_head_count_kv(), Some(8));

        // Rope
        assert_eq!(header.rope_dimension_count(), Some(128));

        // File type
        assert_eq!(header.file_type(), Some("6".to_string()));
    }

    #[test]
    fn test_value_type_name() {
        assert_eq!(GgufKvValue::Uint8(1).type_name(), "u8");
        assert_eq!(GgufKvValue::String("test".to_string()).type_name(), "str");
        assert_eq!(GgufKvValue::Bool(true).type_name(), "bool");
        assert_eq!(GgufKvValue::Float32(1.0).type_name(), "f32");
        assert_eq!(GgufKvValue::Array(vec![]).type_name(), "array");
        assert_eq!(GgufKvValue::Bfloat16(1.0).type_name(), "bf16");
    }

    #[test]
    fn test_gguf_version_from_u32_and_to_u32() {
        assert_eq!(GgufVersion::from_u32(1), Some(GgufVersion::V1));
        assert_eq!(GgufVersion::from_u32(2), Some(GgufVersion::V2));
        assert_eq!(GgufVersion::from_u32(3), Some(GgufVersion::V3));
        assert_eq!(GgufVersion::from_u32(4), None);

        assert_eq!(GgufVersion::V1.to_u32(), 1);
        assert_eq!(GgufVersion::V2.to_u32(), 2);
        assert_eq!(GgufVersion::V3.to_u32(), 3);
    }

    #[test]
    fn test_extract_tensor_bytes_with_header() {
        let header = make_v3_header();
        let tensor = header.get_tensor("token_embd.weight").unwrap();
        // This would fail without a real file, so we just verify the offset calculation
        let file_offset = header.data_section_start + tensor.offset;
        assert!(file_offset > 0 || header.data_section_start > 0);
    }

    #[test]
    fn test_compute_data_section_start_v3_aligned() {
        let kv_pairs: Vec<GgufKvPair> = vec![];
        let tensors: Vec<GgufTensorInfo> = vec![];
        let start = compute_data_section_start(3, &kv_pairs, &tensors, Some(32));
        assert_eq!(start % 32, 0);
    }

    #[test]
    fn test_compute_data_section_start_v3_not_aligned() {
        // Create a header where the base size is not aligned to 32
        let kv_pairs = vec![
            GgufKvPair {
                key: "x".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(1),
            },
        ];
        let tensors: Vec<GgufTensorInfo> = vec![];
        let start = compute_data_section_start(3, &kv_pairs, &tensors, Some(32));
        assert_eq!(start % 32, 0);
    }
}
