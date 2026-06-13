use pesti_gguf::GgufDtype;
use pesti_gguf::GgufHeader;
use pesti_plug_in::manifest::WeightManifest;

use crate::error::RunnerError;
use pesti_safetensors::error::SafetensorsSchemaError;
use std::path::Path;
use tracing::debug;

/// Model loader that consumes WeightManifest from safetensors DB.
///
/// loads tensors for inference engine consumption.
pub struct ModelLoader {
    pub conn: rusqlite::Connection,
}

impl ModelLoader {
    pub fn new(conn: rusqlite::Connection) -> Self {
        Self { conn }
    }

    /// Load weight manifest for a model from safetensors DB.
    pub fn load_manifest(&self, model_name: &str) -> Result<WeightManifest, RunnerError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_name, repo_id, file_path, tensor_count, dtype, device, size_bytes, checksum, metadata, loaded_at, created_at, active FROM model_weights
             WHERE model_name = ?1 AND active = 1
             ORDER BY loaded_at DESC LIMIT 1",
        )?;

        let row = stmt
            .query_row(rusqlite::params![model_name], |row| {
                let metadata_str: String = row.get(9)?;
                let metadata: serde_json::Value =
                    serde_json::from_str(&metadata_str).unwrap_or_default();

                Ok(pesti_plug_in::manifest::ModelWeightRow {
                    id: row.get(0)?,
                    model_name: row.get(1)?,
                    repo_id: row.get(2)?,
                    file_path: row.get(3)?,
                    tensor_count: row.get(4)?,
                    dtype: row.get(5)?,
                    device: row.get(6)?,
                    size_bytes: row.get(7)?,
                    checksum: row.get(8)?,
                    metadata,
                    loaded_at: row.get(10)?,
                    created_at: row.get(11)?,
                    active: row.get(12)?,
                })
            })
            .map_err(RunnerError::Sqlite)?;

        let mut tensor_stmt = self.conn.prepare(
            "SELECT id, weight_id, tensor_name, shape, dtype, size_bytes, checksum FROM tensor_metadata
             WHERE weight_id = ?1",
        )?;

        let tensors: Vec<pesti_plug_in::manifest::TensorMetadataRow> = tensor_stmt
            .query_map(rusqlite::params![row.id], |row| {
                Ok(pesti_plug_in::manifest::TensorMetadataRow {
                    id: row.get(0)?,
                    weight_id: row.get(1)?,
                    tensor_name: row.get(2)?,
                    shape: row.get(3)?,
                    dtype: row.get(4)?,
                    size_bytes: row.get(5)?,
                    checksum: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let manifest = WeightManifest {
            weight_id: row.id,
            model_name: row.model_name,
            repo_id: row.repo_id,
            file_path: row.file_path,
            tensor_count: row.tensor_count,
            dtype: row.dtype,
            device: row.device,
            size_bytes: row.size_bytes,
            checksum: row.checksum,
            tensors,
            metadata: row.metadata,
            lazy_loading: true,
        };

        debug!(
            model_name = %model_name,
            tensor_count = manifest.tensor_count,
            "Model loader: manifest loaded from safetensors DB"
        );

        Ok(manifest)
    }

    /// Verify weight checksum integrity.
    pub fn verify_checksum(&self, weight_id: &str, expected: &str) -> Result<bool, RunnerError> {
        pesti_safetensors::schema::verify_weight_checksum(&self.conn, weight_id, expected)
            .map_err(|_: SafetensorsSchemaError| {
                RunnerError::Sqlite(rusqlite::Error::QueryReturnedNoRows)
            })
    }

    /// List active weights for model selection.
    pub fn list_active(
        &self,
        limit: usize,
    ) -> Result<Vec<pesti_safetensors::schema::ModelWeightRow>, RunnerError> {
        pesti_safetensors::schema::list_active_weights(&self.conn, limit).map_err(
            |e: SafetensorsSchemaError| {
                RunnerError::Sqlite(match e {
                    SafetensorsSchemaError::Sqlite(r) => r,
                    _ => rusqlite::Error::QueryReturnedNoRows,
                })
            },
        )
    }

    /// Parse a GGUF file and return its header (metadata only, no tensor data).
    pub fn load_gguf_header(path: &Path) -> Result<GgufHeader, RunnerError> {
        pesti_gguf::parser::parse_gguf(path).map_err(RunnerError::Gguf)
    }

    /// Detect the quantization type from a GGUF header.
    pub fn detect_quantization(header: &GgufHeader) -> Option<GgufDtype> {
        header
            .get_kv_str("general.file_type")
            .and_then(|s| s.parse::<u32>().ok())
            .map(GgufDtype::from_u32)
            .or_else(|| {
                header.get_kv_str("general.file_type").map(|s| match s {
                    "F16" => GgufDtype::F16,
                    "F32" => GgufDtype::F32,
                    "Q4_0" => GgufDtype::Q4_0,
                    "Q4_1" => GgufDtype::Q4_1,
                    "Q5_0" => GgufDtype::Q5_0,
                    "Q5_1" => GgufDtype::Q5_1,
                    "Q8_0" => GgufDtype::Q8_0,
                    "Q8_1" => GgufDtype::Q8_1,
                    "Q2_K" => GgufDtype::Q2_K,
                    "Q3_K" => GgufDtype::Q3_K,
                    "Q4_K" => GgufDtype::Q4_K,
                    "Q5_K" => GgufDtype::Q5_K,
                    "Q6_K" => GgufDtype::Q6_K,
                    "Q8_K" => GgufDtype::Q8_K,
                    "BF16" => GgufDtype::BF16,
                    _ => GgufDtype::Unknown(0),
                })
            })
    }

    /// Extract raw tensor bytes from a GGUF file by tensor name.
    pub fn extract_gguf_tensor(path: &Path, tensor_name: &str) -> Result<Vec<u8>, RunnerError> {
        let header = Self::load_gguf_header(path)?;
        let tensor = header.get_tensor(tensor_name).ok_or_else(|| {
            RunnerError::Gguf(pesti_gguf::GgufError::InvalidTensor(format!(
                "tensor '{tensor_name}' not found"
            )))
        })?;

        let size = tensor.element_count() as usize;
        pesti_gguf::parser::extract_tensor_bytes(path, tensor.offset, size)
            .map_err(RunnerError::Gguf)
    }

    /// Get architecture from a GGUF header.
    pub fn gguf_architecture(header: &GgufHeader) -> Option<&str> {
        header.architecture()
    }

    /// Get context length from a GGUF header.
    pub fn gguf_context_length(header: &GgufHeader) -> Option<u32> {
        header.context_length()
    }

    /// Get embedding dimension from a GGUF header.
    pub fn gguf_embedding_length(header: &GgufHeader) -> Option<u32> {
        header.embedding_length()
    }

    /// Get block count (number of layers) from a GGUF header.
    pub fn gguf_block_count(header: &GgufHeader) -> Option<u32> {
        header.block_count()
    }

    /// Get attention head count from a GGUF header.
    pub fn gguf_attention_head_count(header: &GgufHeader) -> Option<u32> {
        header.attention_head_count()
    }

    /// Get attention head count KV from a GGUF header.
    pub fn gguf_attention_head_count_kv(header: &GgufHeader) -> Option<u32> {
        header.attention_head_count_kv()
    }

    /// Get rope dimension count from a GGUF header.
    pub fn gguf_rope_dimension_count(header: &GgufHeader) -> Option<i32> {
        header.rope_dimension_count()
    }

    /// Get normalization epsilon from a GGUF header.
    pub fn gguf_normalization_epsilon(header: &GgufHeader) -> Option<f32> {
        header.normalization_epsilon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_weight_loader::load_gguf_weights;
    use pesti_gguf::compute_data_section_start;
    use pesti_gguf::{GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn make_test_gguf(path: &Path) {
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
            kv_pair_u32("llama.context_length", 4096),
            kv_pair_u32("llama.embedding_length", 64),
            kv_pair_u32("llama.block_count", 2),
            kv_pair_u32("llama.attention.head_count", 4),
            kv_pair_u32("llama.attention.head_count_kv", 2),
            kv_pair_u32("llama.feed_forward_length", 128),
            kv_pair_i32("llama.rope.dimension_count", 64),
            kv_pair_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
            kv_pair_u32("tokenizer.ggml.tokens", 32000),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![
            GgufTensorInfo {
                name: "tok_embeddings.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "output.weight".to_string(),
                shape: vec![32000u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wq.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wk.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wv.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention.wo.weight".to_string(),
                shape: vec![64u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.attention_norm.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.ffn_norm.weight".to_string(),
                shape: vec![64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.feed_forward.w1.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.feed_forward.w2.weight".to_string(),
                shape: vec![128u64, 64u64],
                offset: 0,
                dtype: 1,
            },
            GgufTensorInfo {
                name: "layers.0.feed_forward.w3.weight".to_string(),
                shape: vec![64u64, 128u64],
                offset: 0,
                dtype: 1,
            },
        ];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total_tensor_bytes: u64 = tensors
            .iter()
            .map(|t| {
                let elems: u64 = t.shape.iter().product();
                elems * 2
            })
            .sum();
        buf.resize((data_section_start + total_tensor_bytes) as usize, 0);
        for i in 0..total_tensor_bytes as usize {
            buf[data_section_start as usize + i] = if i % 2 == 0 { 0x00 } else { 0x3F };
        }
        std::fs::write(path, &buf).unwrap();
    }

    fn kv_pair_str(key: &str, value: &str) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String(value.to_string()),
        }
    }
    fn kv_pair_u32(key: &str, value: u32) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(value),
        }
    }
    fn kv_pair_f32(key: &str, value: f32) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: GgufValueType::Float32,
            value: GgufKvValue::Float32(value),
        }
    }
    fn kv_pair_i32(key: &str, value: i32) -> GgufKvPair {
        GgufKvPair {
            key: key.to_string(),
            value_type: GgufValueType::Int32,
            value: GgufKvValue::Int32(value),
        }
    }
    fn write_kv_value(buf: &mut Vec<u8>, value: &GgufKvValue) {
        match value {
            GgufKvValue::Uint8(v) => buf.push(*v),
            GgufKvValue::Int8(v) => buf.push(*v as u8),
            GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
            GgufKvValue::Int16(v) => buf.extend_from_slice(&(*v as i16).to_le_bytes()),
            GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            GgufKvValue::Int32(v) => buf.extend_from_slice(&(*v as i32).to_le_bytes()),
            GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            GgufKvValue::Int64(v) => buf.extend_from_slice(&(*v as i64).to_le_bytes()),
            GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            GgufKvValue::Bool(v) => buf.push(*v as u8),
            GgufKvValue::String(s) => {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            GgufKvValue::Int8Array(arr) => {
                let bytes: Vec<u8> = arr.iter().map(|b| *b as u8).collect();
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(&bytes);
            }
            GgufKvValue::Uint8Array(arr) => {
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                buf.extend_from_slice(arr);
            }
            GgufKvValue::Array(arr) => {
                buf.extend_from_slice(&9u32.to_le_bytes());
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                for elem in arr {
                    write_kv_value(buf, elem);
                }
            }
            GgufKvValue::Bfloat16(v) => {
                let raw = (*v as u32) << 16;
                buf.extend_from_slice(&((raw as u16) as u16).to_le_bytes());
            }
            GgufKvValue::Float16(v) => buf.extend_from_slice(&(*v as u16).to_le_bytes()),
        }
    }

    #[test]
    fn detect_quantization_f16() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 1,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        assert_eq!(
            ModelLoader::detect_quantization(&header),
            Some(GgufDtype::F16)
        );
    }

    #[test]
    fn detect_quantization_q4_0() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "Q4_0"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        assert_eq!(
            ModelLoader::detect_quantization(&header),
            Some(GgufDtype::Q4_0)
        );
    }

    #[test]
    fn detect_quantization_q8_0() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "Q8_0"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        assert_eq!(
            ModelLoader::detect_quantization(&header),
            Some(GgufDtype::Q8_0)
        );
    }

    #[test]
    fn detect_quantization_bf16() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "BF16"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        assert_eq!(
            ModelLoader::detect_quantization(&header),
            Some(GgufDtype::BF16)
        );
    }

    #[test]
    fn detect_quantization_unknown_returns_none() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "UNKNOWN_X"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        assert_eq!(
            ModelLoader::detect_quantization(&header),
            Some(GgufDtype::Unknown(0))
        );
    }

    #[test]
    fn detect_quantization_no_file_type_returns_none() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "llama")];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        // No file_type key → returns None
        assert_eq!(ModelLoader::detect_quantization(&header), None);
    }

    #[test]
    fn extract_gguf_tensor_returns_bytes() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        make_test_gguf(&path);
        let result = ModelLoader::extract_gguf_tensor(&path, "tok_embeddings.weight");
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(bytes.len() > 0);
    }

    #[test]
    fn extract_gguf_tensor_missing_tensor_fails() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        make_test_gguf(&path);
        let result = ModelLoader::extract_gguf_tensor(&path, "nonexistent.weight");
        assert!(result.is_err());
    }

    #[test]
    fn load_gguf_header_returns_header() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 1,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = ModelLoader::load_gguf_header(&path).unwrap();
        assert_eq!(header.architecture(), Some("llama"));
    }

    #[test]
    fn load_gguf_header_nonexistent_file_fails() {
        let path =
            PathBuf::from(tempdir().unwrap().path().to_str().unwrap()).join("nonexistent.gguf");
        let result = ModelLoader::load_gguf_header(&path);
        assert!(result.is_err());
    }

    #[test]
    fn gguf_architecture_returns_value() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "mistral")];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = ModelLoader::load_gguf_header(&path).unwrap();
        assert_eq!(ModelLoader::gguf_architecture(&header), Some("mistral"));
    }

    #[test]
    fn gguf_context_length_returns_value() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_u32("llama.context_length", 8192),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = ModelLoader::load_gguf_header(&path).unwrap();
        assert_eq!(ModelLoader::gguf_context_length(&header), Some(8192));
    }

    #[test]
    fn gguf_helpers_return_none_when_missing() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![kv_pair_str("general.architecture", "llama")];
        let tensors: Vec<GgufTensorInfo> = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 0,
        }];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = ModelLoader::load_gguf_header(&path).unwrap();
        assert!(ModelLoader::gguf_embedding_length(&header).is_none());
        assert!(ModelLoader::gguf_block_count(&header).is_none());
        assert!(ModelLoader::gguf_attention_head_count(&header).is_none());
        assert!(ModelLoader::gguf_attention_head_count_kv(&header).is_none());
        assert!(ModelLoader::gguf_rope_dimension_count(&header).is_none());
        assert!(ModelLoader::gguf_normalization_epsilon(&header).is_none());
    }

    #[test]
    fn load_gguf_weights_with_real_file() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        make_test_gguf(&path);
        let result = load_gguf_weights(&path);
        assert!(result.is_ok());
        let weights = result.unwrap();
        assert!(weights.tensors.contains_key("tok_embeddings.weight"));
        assert!(weights.tensors.contains_key("output.weight"));
        assert_eq!(weights.header.architecture(), Some("llama"));
    }

    #[test]
    fn load_gguf_weights_empty_tensors() {
        let dir = tempdir().unwrap();
        let path = PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_str("general.file_type", "F16"),
        ];
        let tensors: Vec<GgufTensorInfo> = vec![];
        let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, None);
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
        let total: u64 = tensors
            .iter()
            .map(|t| t.shape.iter().product::<u64>() * 2)
            .sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let result = load_gguf_weights(&path);
        assert!(result.is_ok());
        let weights = result.unwrap();
        assert!(weights.tensors.is_empty());
    }
}
