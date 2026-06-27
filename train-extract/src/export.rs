use crate::error::{TrainExtractError, TrainExtractResult};
use crate::format::Sample;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Dataset manifest stored alongside the exported data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub version: u32,
    pub created_at: String,
    pub entry_count: usize,
    pub total_weight: f64,
    pub unique_tags: Vec<String>,
    pub tag_weights: HashMap<String, f64>,
    pub source_provenance: Vec<String>,
    pub checksum: String,
}

/// Export format for the training dataset.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExportFormat {
    #[default]
    Jsonl,
    SafetensorsManifest,
}

/// Configuration for dataset export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub output_dir: String,
    pub dataset_name: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Jsonl,
            output_dir: "./train-data".to_string(),
            dataset_name: "personal-knowledge".to_string(),
        }
    }
}

/// Export samples as a JSONL file and return manifest metadata.
pub fn export_jsonl(samples: &[Sample], config: &ExportConfig) -> TrainExtractResult<DatasetManifest> {
    let output_path = format!("{}/{}.jsonl", config.output_dir, config.dataset_name);

    fs::create_dir_all(&config.output_dir).map_err(|e| {
        TrainExtractError::Export(format!("failed to create output dir: {e}"))
    })?;

    let mut file = fs::File::create(&output_path).map_err(|e| {
        TrainExtractError::Export(format!("failed to create file: {e}"))
    })?;

    // Write each sample as a JSON line
    for sample in samples {
        let json_line = serde_json::to_string(sample).map_err(|e| {
            TrainExtractError::Export(format!("failed to serialize sample: {e}"))
        })?;
        writeln!(file, "{json_line}").map_err(|e| {
            TrainExtractError::Export(format!("failed to write line: {e}"))
        })?;
    }

    // Compute checksum of the file content
    let file_content = fs::read(&output_path).map_err(|e| {
        TrainExtractError::Export(format!("failed to read output file: {e}"))
    })?;
    let checksum = hex::encode(Sha256::digest(&file_content));

    // Collect unique tags and tag weights
    let mut tag_weights = HashMap::new();
    let mut unique_tags = std::collections::HashSet::new();
    let mut total_weight = 0.0;

    for sample in samples {
        total_weight += sample.weight;
        for tag in &sample.tags {
            unique_tags.insert(tag.clone());
            *tag_weights.entry(tag.clone()).or_insert(0.0) += sample.weight;
        }
    }

    // Collect source provenance
    let mut source_provenance = Vec::new();
    for sample in samples {
        source_provenance.push(sample.provenance_id.clone());
    }

    let manifest = DatasetManifest {
        version: 1,
        created_at: chrono::Utc::now().to_rfc3339(),
        entry_count: samples.len(),
        total_weight,
        unique_tags: unique_tags.into_iter().collect(),
        tag_weights,
        source_provenance,
        checksum: checksum.clone(),
    };

    // Write manifest alongside the data file
    let manifest_path = format!("{}/{}.manifest.json", config.output_dir, config.dataset_name);
    let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
        TrainExtractError::Export(format!("failed to serialize manifest: {e}"))
    })?;
    fs::write(&manifest_path, manifest_json).map_err(|e| {
        TrainExtractError::Export(format!("failed to write manifest: {e}"))
    })?;

    tracing::debug!(
        output_path = %output_path,
        manifest_path = %manifest_path,
        entry_count = samples.len(),
        checksum = %checksum,
        "Exported JSONL dataset"
    );

    Ok(manifest)
}

/// Export a safetensors-compatible manifest for the training dataset.
/// Stores the manifest in the safetensors SQLite store as a "model weight" entry.
pub fn export_safetensors_manifest(
    samples: &[Sample],
    config: &ExportConfig,
    store: &crabjar_safetensors::SafetensorsStore,
) -> TrainExtractResult<DatasetManifest> {
    let output_path = format!("{}/{}.train.safetensors", config.output_dir, config.dataset_name);
    fs::create_dir_all(&config.output_dir).map_err(|e| {
        TrainExtractError::Export(format!("failed to create output dir: {e}"))
    })?;

    // Create an empty safetensors file (no tensors, just metadata)
    let tensors: Vec<(&str, safetensors::tensor::TensorView)> = Vec::new();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "dataset_type".to_string(),
        "instruction_tuning".to_string(),
    );
    metadata.insert(
        "dataset_name".to_string(),
        config.dataset_name.clone(),
    );
    metadata.insert(
        "created_at".to_string(),
        chrono::Utc::now().to_rfc3339(),
    );
    metadata.insert("entry_count".to_string(), samples.len().to_string());
    metadata.insert("source".to_string(), "train-extract".to_string());

    let serialized = safetensors::serialize(tensors, Some(metadata)).map_err(|e| {
        TrainExtractError::Export(format!("failed to serialize safetensors: {e}"))
    })?;

    fs::write(&output_path, serialized).map_err(|e| {
        TrainExtractError::Export(format!("failed to write safetensors file: {e}"))
    })?;

    // Parse and store in SQLite
    let (weight_id, _tensor_rows) = store.parse_weights(
        &output_path,
        &config.dataset_name,
        "train-extract",
    )?;

    // Compute checksum
    let file_content = fs::read(&output_path).map_err(|e| {
        TrainExtractError::Export(format!("failed to read output file: {e}"))
    })?;
    let checksum = hex::encode(Sha256::digest(&file_content));

    // Update weight with checksum
    let _ = store.verify_checksum(&weight_id, &checksum);

    // Collect manifest data
    let mut tag_weights = HashMap::new();
    let mut unique_tags = std::collections::HashSet::new();
    let mut total_weight = 0.0;
    let mut source_provenance = Vec::new();

    for sample in samples {
        total_weight += sample.weight;
        for tag in &sample.tags {
            unique_tags.insert(tag.clone());
            *tag_weights.entry(tag.clone()).or_insert(0.0) += sample.weight;
        }
        source_provenance.push(sample.provenance_id.clone());
    }

    let manifest = DatasetManifest {
        version: 1,
        created_at: chrono::Utc::now().to_rfc3339(),
        entry_count: samples.len(),
        total_weight,
        unique_tags: unique_tags.into_iter().collect(),
        tag_weights,
        source_provenance,
        checksum,
    };

    tracing::debug!(
        weight_id = %weight_id,
        output_path = %output_path,
        entry_count = samples.len(),
        "Exported safetensors manifest"
    );

    Ok(manifest)
}

/// Export the training dataset with the given config.
pub fn export(
    samples: &[Sample],
    config: &ExportConfig,
    store: Option<&crabjar_safetensors::SafetensorsStore>,
) -> TrainExtractResult<DatasetManifest> {
    match config.format {
        ExportFormat::Jsonl => export_jsonl(samples, config),
        ExportFormat::SafetensorsManifest => {
            let Some(store) = store else {
                return Err(TrainExtractError::Export(
                    "safetensors store required for this format".to_string(),
                ));
            };
            export_safetensors_manifest(samples, config, store)
        }
    }
}

/// Generate a JSONL dataset from extracted data.
pub fn export_dataset(
    samples: &[Sample],
    output_dir: impl AsRef<Path>,
) -> TrainExtractResult<DatasetManifest> {
    let dir = output_dir.as_ref().to_string_lossy().to_string();
    let config = ExportConfig {
        format: ExportFormat::Jsonl,
        output_dir: dir.clone(),
        dataset_name: "personal-knowledge".to_string(),
    };
    export_jsonl(samples, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Sample, SampleSource};
    use tempfile::tempdir;

    fn make_test_samples() -> Vec<Sample> {
        vec![
            Sample {
                instruction: "What is the rule?".to_string(),
                response: "Always use thiserror.".to_string(),
                source: SampleSource::KnowledgeEntry {
                    entry_id: 1,
                    kind: "Pattern".to_string(),
                },
                weight: 0.9,
                tags: vec!["rust".to_string(), "error-handling".to_string()],
                provenance_id: "knowledge:1".to_string(),
            },
            Sample {
                instruction: "What is the pattern?".to_string(),
                response: "Use snake_case.".to_string(),
                source: SampleSource::KnowledgeEntry {
                    entry_id: 2,
                    kind: "Pattern".to_string(),
                },
                weight: 0.7,
                tags: vec!["rust".to_string(), "naming".to_string()],
                provenance_id: "knowledge:2".to_string(),
            },
        ]
    }

    #[test]
    fn export_jsonl_creates_file() {
        let dir = tempdir().unwrap();
        let samples = make_test_samples();
        let config = ExportConfig {
            format: ExportFormat::Jsonl,
            output_dir: dir.path().to_string_lossy().to_string(),
            dataset_name: "test".to_string(),
        };
        let manifest = export_jsonl(&samples, &config).unwrap();
        assert!(manifest.entry_count == 2);

        let jsonl_path = dir.path().join("test.jsonl");
        assert!(jsonl_path.exists());

        let content = fs::read_to_string(&jsonl_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);

        for line in lines {
            let _sample: Sample = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn export_jsonl_creates_manifest() {
        let dir = tempdir().unwrap();
        let samples = make_test_samples();
        let config = ExportConfig {
            format: ExportFormat::Jsonl,
            output_dir: dir.path().to_string_lossy().to_string(),
            dataset_name: "test".to_string(),
        };
        let manifest = export_jsonl(&samples, &config).unwrap();

        let manifest_path = dir.path().join("test.manifest.json");
        assert!(manifest_path.exists());

        let manifest_str = fs::read_to_string(&manifest_path).unwrap();
        let parsed: DatasetManifest = serde_json::from_str(&manifest_str).unwrap();
        assert_eq!(parsed.entry_count, 2);
        assert!(!parsed.checksum.is_empty());
        assert_eq!(parsed.unique_tags.len(), 3); // rust, error-handling, naming
    }

    #[test]
    fn export_jsonl_checksum_is_deterministic() {
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();
        let samples = make_test_samples();

        let config1 = ExportConfig {
            format: ExportFormat::Jsonl,
            output_dir: dir1.path().to_string_lossy().to_string(),
            dataset_name: "test".to_string(),
        };
        let config2 = ExportConfig {
            format: ExportFormat::Jsonl,
            output_dir: dir2.path().to_string_lossy().to_string(),
            dataset_name: "test".to_string(),
        };

        let m1 = export_jsonl(&samples, &config1).unwrap();
        let m2 = export_jsonl(&samples, &config2).unwrap();

        assert_eq!(m1.checksum, m2.checksum);
    }

    #[test]
    fn export_safetensors_manifest_requires_store() {
        let dir = tempdir().unwrap();
        let samples = make_test_samples();
        let config = ExportConfig {
            format: ExportFormat::SafetensorsManifest,
            output_dir: dir.path().to_string_lossy().to_string(),
            dataset_name: "test".to_string(),
        };
        let result = export(&samples, &config, None);
        assert!(result.is_err());
    }

    #[test]
    fn export_dataset_creates_jsonl() {
        let dir = tempdir().unwrap();
        let samples = make_test_samples();
        let manifest = export_dataset(&samples, dir.path()).unwrap();
        assert_eq!(manifest.entry_count, 2);

        let jsonl_path = dir.path().join("personal-knowledge.jsonl");
        assert!(jsonl_path.exists());
    }
}
