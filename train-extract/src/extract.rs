use crate::error::{TrainExtractError, TrainExtractResult};
use agent_context::KnowledgeKind;
use rusqlite::Connection;
use std::collections::HashMap;

/// Extracted knowledge entry from the knowledge store.
#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    pub id: i64,
    pub content: String,
    pub kind: KnowledgeKind,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub weight: f64,
    pub source_type: String,
    pub source_id: String,
    pub provenance_id: String,
    pub active: bool,
    pub created_at: i64,
}

/// Extracted event from mirror-log.
#[derive(Debug, Clone)]
pub struct LogEvent {
    pub id: String,
    pub content: String,
    pub source: String,
    pub meta: Option<String>,
    pub timestamp: i64,
}

/// Extracted chunk from mirror-log.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub event_id: String,
    pub chunk_index: i64,
    pub content: String,
    pub timestamp: i64,
}

/// Extracted annotation from state-doc overlay.
#[derive(Debug, Clone)]
pub struct Annotation {
    pub id: String,
    pub doc_name: String,
    pub kind: String,
    pub message: String,
    pub status: String,
    pub resolution_reason: Option<String>,
    pub confidence: f64,
    pub created_at: u128,
}

/// Configuration for data extraction.
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// Only include entries with these tags (empty = all tags).
    pub tags: Vec<String>,
    /// Maximum number of entries to extract per source.
    pub max_entries: usize,
    /// Only include entries created after this unix timestamp (0 = all).
    pub since: i64,
    /// Include resolved annotations from state-docs.
    pub include_annotations: bool,
    /// Include mirror-log events.
    pub include_events: bool,
    /// Include chunks.
    pub include_chunks: bool,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            max_entries: 10000,
            since: 0,
            include_annotations: true,
            include_events: true,
            include_chunks: false,
        }
    }
}

/// Extract data from the knowledge store and mirror-log databases.
pub fn extract(
    knowledge_conn: &Connection,
    mirror_log_conn: Option<&Connection>,
    config: &ExtractConfig,
) -> TrainExtractResult<ExtractedData> {
    let mut data = ExtractedData::default();

    data.knowledge_entries = extract_knowledge_entries(knowledge_conn, config)?;
    data.sample_count = data.knowledge_entries.len();

    if config.include_events {
        data.events = extract_events(mirror_log_conn, config)?;
    }

    if config.include_annotations {
        data.annotations = extract_annotations(knowledge_conn, config)?;
    }

    if config.include_chunks {
        data.chunks = extract_chunks(mirror_log_conn, config)?;
    }

    if data.is_empty() {
        return Err(TrainExtractError::EmptyDataset);
    }

    Ok(data)
}

/// Extract knowledge entries from the knowledge store.
fn extract_knowledge_entries(
    conn: &Connection,
    config: &ExtractConfig,
) -> TrainExtractResult<Vec<KnowledgeEntry>> {
    let mut stmt = conn.prepare(
        "SELECT id, content, kind, tags, metadata, weight, active, source_type, source_id, provenance_id, created_at
         FROM knowledge_entries
         WHERE active = 1
           AND (?1 = 0 OR created_at >= ?1)
         ORDER BY created_at DESC
         LIMIT ?2",
    )?;

    let rows = stmt.query_map([config.since, config.max_entries as i64], |row| {
        let tags_str: String = row.get(3)?;
        let metadata_str: String = row.get(4)?;
        let kind_str: String = row.get(2)?;

        let kind = match kind_str.as_str() {
            "\"instruction\"" => KnowledgeKind::Instruction,
            "\"pattern\"" => KnowledgeKind::Pattern,
            "\"example\"" => KnowledgeKind::Example,
            "\"context\"" => KnowledgeKind::Context,
            _ => KnowledgeKind::Context,
        };

        Ok(KnowledgeEntry {
            id: row.get(0)?,
            content: row.get(1)?,
            kind,
            tags: serde_json::from_str(&tags_str).unwrap_or_default(),
            metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
            weight: row.get(5)?,
            active: row.get(6)?,
            source_type: row.get(7)?,
            source_id: row.get(8)?,
            provenance_id: row.get(9)?,
            created_at: row.get(10)?,
        })
    })?;

    let entries: Vec<KnowledgeEntry> = rows.collect::<Result<Vec<_>, _>>().map_err(|e| {
        TrainExtractError::Export(format!("failed to read knowledge entries: {e}"))
    })?;

    // Apply tag filter if specified
    let filtered = if config.tags.is_empty() {
        entries
    } else {
        entries
            .into_iter()
            .filter(|entry| {
                config.tags.iter().any(|tag| {
                    entry.tags.iter().any(|t| t == tag)
                })
            })
            .collect()
    };

    tracing::debug!(
        entries_extracted = filtered.len(),
        "Extracted knowledge entries"
    );

    Ok(filtered)
}

/// Extract events from mirror-log database.
fn extract_events(
    conn: Option<&Connection>,
    config: &ExtractConfig,
) -> TrainExtractResult<Vec<LogEvent>> {
    let Some(conn) = conn else {
        return Ok(Vec::new());
    };

    let mut stmt = conn.prepare(
        "SELECT id, content, source, meta, created_at
         FROM events
         WHERE active = 1
           AND (?1 = 0 OR created_at >= ?1)
         ORDER BY created_at DESC
         LIMIT ?2",
    )?;

    let rows = stmt.query_map([config.since, config.max_entries as i64], |row| {
        Ok(LogEvent {
            id: row.get(0)?,
            content: row.get(1)?,
            source: row.get(2)?,
            meta: row.get(3)?,
            timestamp: row.get(4)?,
        })
    })?;

    let events: Vec<LogEvent> = rows.collect::<Result<Vec<_>, _>>().map_err(|e| {
        TrainExtractError::Export(format!("failed to read events: {e}"))
    })?;

    tracing::debug!(events_extracted = events.len(), "Extracted mirror-log events");

    Ok(events)
}

/// Extract chunks from mirror-log database.
fn extract_chunks(
    conn: Option<&Connection>,
    config: &ExtractConfig,
) -> TrainExtractResult<Vec<Chunk>> {
    let Some(conn) = conn else {
        return Ok(Vec::new());
    };

    let mut stmt = conn.prepare(
        "SELECT id, event_id, chunk_index, content, timestamp
         FROM chunks
         WHERE timestamp >= ?1
         ORDER BY timestamp DESC
         LIMIT ?2",
    )?;

    let rows = stmt.query_map([config.since, config.max_entries as i64], |row| {
        Ok(Chunk {
            id: row.get(0)?,
            event_id: row.get(1)?,
            chunk_index: row.get(2)?,
            content: row.get(3)?,
            timestamp: row.get(4)?,
        })
    })?;

    let chunks: Vec<Chunk> = rows.collect::<Result<Vec<_>, _>>().map_err(|e| {
        TrainExtractError::Export(format!("failed to read chunks: {e}"))
    })?;

    tracing::debug!(chunks_extracted = chunks.len(), "Extracted chunks");

    Ok(chunks)
}

/// Extract resolved annotations from the knowledge store.
/// Annotations that have been promoted to knowledge entries are already
/// in knowledge_entries. This extracts resolved annotation metadata
/// for the training dataset.
fn extract_annotations(
    _conn: &Connection,
    _config: &ExtractConfig,
) -> TrainExtractResult<Vec<Annotation>> {
    // Annotations stored in overlays are not directly queryable from SQLite.
    // We return an empty list here; the caller should extract annotations
    // from state-doc overlays separately via KnowledgeBridge.
    // This is a placeholder for when annotation data is persisted.
    tracing::debug!("Annotations not persisted to knowledge store; use KnowledgeBridge for annotation extraction");
    Ok(Vec::new())
}

/// Combined extracted data ready for formatting.
#[derive(Debug, Default)]
pub struct ExtractedData {
    pub knowledge_entries: Vec<KnowledgeEntry>,
    pub events: Vec<LogEvent>,
    pub annotations: Vec<Annotation>,
    pub chunks: Vec<Chunk>,
    pub sample_count: usize,
}

impl ExtractedData {
    /// Returns true if no data was extracted.
    pub fn is_empty(&self) -> bool {
        self.knowledge_entries.is_empty()
            && self.events.is_empty()
            && self.annotations.is_empty()
            && self.chunks.is_empty()
    }

    /// Get a summary of the extracted data.
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "knowledge_entries": self.knowledge_entries.len(),
            "events": self.events.len(),
            "annotations": self.annotations.len(),
            "chunks": self.chunks.len(),
            "total_samples": self.knowledge_entries.len() + self.events.len() + self.annotations.len() + self.chunks.len(),
        })
    }

    /// Group knowledge entries by tag for weighted sampling.
    pub fn tag_distribution(&self) -> HashMap<String, usize> {
        let mut dist = HashMap::new();
        for entry in &self.knowledge_entries {
            for tag in &entry.tags {
                *dist.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_test_db(dir: &tempfile::TempDir) -> (Connection, Connection) {
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
    fn extract_returns_empty_when_no_entries() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);
        let config = ExtractConfig::default();
        let result = extract(&kconn, Some(&mconn), &config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TrainExtractError::EmptyDataset));
    }

    #[test]
    fn extract_knowledge_entries_with_data() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        // Insert a test entry
        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'test content', '\"pattern\"', '[\"test\", \"rust\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();

        let config = ExtractConfig::default();
        let data = extract(&kconn, Some(&mconn), &config).unwrap();
        assert_eq!(data.knowledge_entries.len(), 1);
        assert_eq!(data.knowledge_entries[0].content, "test content");
        assert_eq!(data.knowledge_entries[0].tags, vec!["test".to_string(), "rust".to_string()]);
    }

    #[test]
    fn extract_filters_by_tag() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'rust entry', '\"pattern\"', '[\"rust\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();
        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (2, 'python entry', '\"pattern\"', '[\"python\"]', '{}', 1.0, 1, 'user', 'test-2', 1000000)",
            [],
        ).unwrap();

        let config = ExtractConfig {
            tags: vec!["rust".to_string()],
            ..ExtractConfig::default()
        };
        let data = extract(&kconn, Some(&mconn), &config).unwrap();
        assert_eq!(data.knowledge_entries.len(), 1);
        assert_eq!(data.knowledge_entries[0].content, "rust entry");
    }

    #[test]
    fn extract_filters_by_since_timestamp() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'old entry', '\"pattern\"', '[\"test\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();
        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (2, 'new entry', '\"pattern\"', '[\"test\"]', '{}', 1.0, 1, 'user', 'test-2', 2000000)",
            [],
        ).unwrap();

        let config = ExtractConfig {
            since: 1500000,
            ..ExtractConfig::default()
        };
        let data = extract(&kconn, Some(&mconn), &config).unwrap();
        assert_eq!(data.knowledge_entries.len(), 1);
        assert_eq!(data.knowledge_entries[0].content, "new entry");
    }

    #[test]
    fn extract_includes_events() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'test entry', '\"pattern\"', '[\"test\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();

        mconn.execute(
            "INSERT INTO events (id, content, source, meta, created_at, active)
             VALUES ('evt-1', 'test event', 'file', null, 1000000, 1)",
            [],
        ).unwrap();

        let config = ExtractConfig::default();
        let data = extract(&kconn, Some(&mconn), &config).unwrap();
        assert_eq!(data.events.len(), 1);
        assert_eq!(data.events[0].content, "test event");
    }

    #[test]
    fn extract_without_mirror_log_returns_empty_events() {
        let dir = tempdir().unwrap();
        let (kconn, _) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'test entry', '\"pattern\"', '[\"test\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();

        let config = ExtractConfig {
            include_events: true,
            ..ExtractConfig::default()
        };
        let data = extract(&kconn, None, &config).unwrap();
        assert!(data.events.is_empty());
    }

    #[test]
    fn extract_summary_works() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'test', '\"pattern\"', '[\"tag1\", \"tag2\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();

        let config = ExtractConfig::default();
        let data = extract(&kconn, Some(&mconn), &config).unwrap();
        let summary = data.summary();
        assert_eq!(summary["knowledge_entries"], 1);
        assert_eq!(summary["total_samples"], 1);
    }

    #[test]
    fn extract_tag_distribution_works() {
        let dir = tempdir().unwrap();
        let (kconn, mconn) = make_test_db(&dir);

        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (1, 'a', '\"pattern\"', '[\"rust\", \"pattern\"]', '{}', 1.0, 1, 'user', 'test-1', 1000000)",
            [],
        ).unwrap();
        kconn.execute(
            "INSERT INTO knowledge_entries (id, content, kind, tags, metadata, weight, active, source_type, source_id, created_at)
             VALUES (2, 'b', '\"pattern\"', '[\"rust\", \"example\"]', '{}', 1.0, 1, 'user', 'test-2', 1000001)",
            [],
        ).unwrap();

        let config = ExtractConfig::default();
        let data = extract(&kconn, Some(&mconn), &config).unwrap();
        let dist = data.tag_distribution();
        assert_eq!(*dist.get("rust").unwrap(), 2);
        assert_eq!(*dist.get("pattern").unwrap(), 1);
        assert_eq!(*dist.get("example").unwrap(), 1);
    }
}
