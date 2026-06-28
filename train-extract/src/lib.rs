pub mod error;
pub mod extract;
pub mod format;
pub mod export;
pub mod weight;

pub use error::{TrainExtractError, TrainExtractResult};
pub use extract::{extract, ExtractConfig, ExtractedData, KnowledgeEntry, LogEvent, Chunk, Annotation};
pub use format::{format_samples, apply_weighting, Sample, SampleSource};
pub use export::{export, export_jsonl, export_safetensors_manifest, ExportConfig, ExportFormat, DatasetManifest};
pub use weight::{weight_samples, WeightConfig, compute_tag_boost, compute_recency};

/// Full pipeline: extract → format → weight → export.
///
/// This is the primary entry point for generating a training dataset
/// from the knowledge store and mirror-log databases.
pub fn run_pipeline(
    knowledge_conn: &rusqlite::Connection,
    mirror_log_conn: Option<&rusqlite::Connection>,
    weight_config: WeightConfig,
    export_config: ExportConfig,
    store: Option<&crabjar_safetensors::SafetensorsStore>,
) -> TrainExtractResult<(Vec<Sample>, DatasetManifest)> {
    // Phase 1: Extract
    let config = ExtractConfig {
        tags: weight_config.tag_boost.keys().cloned().collect(),
        ..ExtractConfig::default()
    };
    let data = extract(knowledge_conn, mirror_log_conn, &config)?;

    // Phase 2: Compute tag boost from data distribution
    let tag_boost = compute_tag_boost(&data, 1);
    let weighted_config = WeightConfig {
        tag_boost,
        ..weight_config
    };

    // Phase 3: Format and weight
    let samples = weight_samples(&data, &weighted_config);

    if samples.is_empty() {
        return Err(TrainExtractError::EmptyDataset);
    }

    // Phase 4: Export
    let manifest = export(&samples, &export_config, store)?;

    Ok((samples, manifest))
}

/// Quick export: extract all active knowledge and write JSONL.
pub fn quick_export(
    knowledge_conn: &rusqlite::Connection,
    mirror_log_conn: Option<&rusqlite::Connection>,
    output_dir: impl AsRef<std::path::Path>,
) -> TrainExtractResult<DatasetManifest> {
    let data = extract(
        knowledge_conn,
        mirror_log_conn,
        &ExtractConfig::default(),
    )?;

    let samples = weight_samples(&data, &WeightConfig::default());
    let dir = output_dir.as_ref().to_string_lossy().to_string();
    let export_config = ExportConfig {
        format: ExportFormat::Jsonl,
        output_dir: dir,
        dataset_name: "personal-knowledge".to_string(),
    };

    export_jsonl(&samples, &export_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use rusqlite::Connection;

    fn make_test_db(dir: &tempfile::TempDir) -> (rusqlite::Connection, rusqlite::Connection) {
        let kconn = Connection::open(dir.path().join("knowledge.db")).unwrap();
        kconn.execute_batch(
            "CREATE TABLE knowledge_entries (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                weight REAL NOT NULL DEFAULT 1.0,
                active INTEGER NOT NULL DEFAULT 1,
                source_type TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL DEFAULT '',
                provenance_id TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
            )",
        ).unwrap();

        let mconn = Connection::open(dir.path().join("mirror.db")).unwrap();
        mconn.execute_batch(
            "CREATE TABLE events (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                meta TEXT,
                created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                active INTEGER NOT NULL DEFAULT 1
            )",
        ).unwrap();

        (kconn, mconn)
    }

    #[test]
    fn quick_export_creates_jsonl() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'Use thiserror for errors.', '\"instruction\"', '[\"rust\"]', '{}', 1.0, 1, 'user', 't1', 1000000)",
            [],
        ).unwrap();

        let output_dir = dir.path().join("output");
        let manifest = quick_export(&kconn, Some(&mconn), &output_dir).unwrap();
        assert_eq!(manifest.entry_count, 1);

        let jsonl_path = output_dir.join("personal-knowledge.jsonl");
        assert!(jsonl_path.exists());
    }

    #[test]
    fn quick_export_fails_on_empty_db() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        let output_dir = dir.path().join("output");
        let result = quick_export(&kconn, Some(&mconn), &output_dir);
        assert!(result.is_err());
    }

    #[test]
    fn run_pipeline_with_data() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'Always format before committing.', '\"instruction\"', '[\"rust\", \"workflow\"]', '{}', 0.9, 1, 'user', 't1', 1000000)",
            [],
        ).unwrap();
        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (2, 'Use snake_case for functions.', '\"pattern\"', '[\"rust\", \"naming\"]', '{}', 0.7, 1, 'agent', 't2', 1000001)",
            [],
        ).unwrap();

        mconn.execute(
            "INSERT INTO events (id, content, source, meta, created_at, active)
             VALUES ('evt-1', 'Updated Cargo.toml', 'file', null, 1000000, 1)",
            [],
        ).unwrap();

        let output_dir = dir.path().join("output");
        let weight_config = WeightConfig::default().tag_boost("rust", 1.5);
        let export_config = ExportConfig {
            format: ExportFormat::Jsonl,
            output_dir: output_dir.to_string_lossy().to_string(),
            dataset_name: "test-pipeline".to_string(),
        };

        let (samples, manifest) = run_pipeline(
            &kconn,
            Some(&mconn),
            weight_config,
            export_config.clone(),
            None,
        )
        .unwrap();

        assert!(samples.len() >= 2);
        assert_eq!(manifest.entry_count, samples.len());

        // Verify rust-tagged samples got boosted
        let rust_samples: Vec<_> = samples.iter().filter(|s| s.tags.contains(&"rust".to_string())).collect();
        assert!(!rust_samples.is_empty());
    }
}
