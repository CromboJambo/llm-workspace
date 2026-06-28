use crate::extract::{Annotation, Chunk, ExtractedData, KnowledgeEntry, LogEvent};
use serde::{Deserialize, Serialize};

/// A single instruction-tuning sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub instruction: String,
    pub response: String,
    pub source: SampleSource,
    pub weight: f64,
    pub tags: Vec<String>,
    pub provenance_id: String,
}

/// Where a sample came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleSource {
    KnowledgeEntry {
        entry_id: i64,
        kind: String,
    },
    LogEvent {
        event_id: String,
        source: String,
    },
    Chunk {
        chunk_id: String,
        event_id: String,
    },
    Annotation {
        doc_name: String,
        annotation_id: String,
        annotation_kind: String,
    },
}

impl SampleSource {
    pub fn to_provenance(&self) -> String {
        match self {
            SampleSource::KnowledgeEntry { entry_id, .. } => format!("knowledge:{entry_id}"),
            SampleSource::LogEvent { event_id, .. } => format!("event:{event_id}"),
            SampleSource::Chunk { chunk_id, .. } => format!("chunk:{chunk_id}"),
            SampleSource::Annotation {
                doc_name,
                annotation_id,
                ..
            } => format!("annotation:{doc_name}:{annotation_id}"),
        }
    }
}

/// Format extracted data into instruction-tuning samples.
pub fn format_samples(data: &ExtractedData) -> Vec<Sample> {
    let mut samples = Vec::new();

    for entry in &data.knowledge_entries {
        samples.extend(format_knowledge_entry(entry));
    }

    for event in &data.events {
        samples.extend(format_log_event(event));
    }

    for chunk in &data.chunks {
        samples.extend(format_chunk(chunk));
    }

    for annotation in &data.annotations {
        samples.extend(format_annotation(annotation));
    }

    samples
}

/// Format a knowledge entry as instruction/response pair.
fn format_knowledge_entry(entry: &KnowledgeEntry) -> Vec<Sample> {
    let mut samples = Vec::new();

    let tags_str = if entry.tags.is_empty() {
        String::new()
    } else {
        format!(" ({})", entry.tags.join(", "))
    };

    let kind_label = match entry.kind {
        agent_context::KnowledgeKind::Instruction => "rule",
        agent_context::KnowledgeKind::Pattern => "pattern",
        agent_context::KnowledgeKind::Example => "example",
        agent_context::KnowledgeKind::Context => "context",
    };

    let instruction = format!(
        "What is the {kind_label}{tags_str}?\n\n{}",
        entry.content.lines().next().unwrap_or("")
    );

    let response = entry
        .content
        .lines()
        .skip_while(|l| l.is_empty())
        .take_while(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    let provenance = format!("knowledge:{}", entry.id);

    samples.push(Sample {
        instruction,
        response: if response.is_empty() {
            entry.content.clone()
        } else {
            response
        },
        source: SampleSource::KnowledgeEntry {
            entry_id: entry.id,
            kind: format!("{:?}", entry.kind),
        },
        weight: entry.weight,
        tags: entry.tags.clone(),
        provenance_id: provenance,
    });

    samples
}

/// Format a mirror-log event as instruction/response pair.
fn format_log_event(event: &LogEvent) -> Vec<Sample> {
    let content = event.content.trim();
    if content.is_empty() {
        return Vec::new();
    }

    let first_line = content.lines().next().unwrap_or(content);
    let instruction = format!("What happened in the {source} event?\n\n{first_line}", source = event.source);

    let response = content
        .lines()
        .skip_while(|l| l.is_empty())
        .take_while(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    vec![Sample {
        instruction,
        response: if response.is_empty() { content.to_string() } else { response },
        source: SampleSource::LogEvent {
            event_id: event.id.clone(),
            source: event.source.clone(),
        },
        weight: 0.5,
        tags: vec![event.source.clone()],
        provenance_id: format!("event:{}", event.id),
    }]
}

/// Format a chunk as instruction/response pair.
fn format_chunk(chunk: &Chunk) -> Vec<Sample> {
    let content = chunk.content.trim();
    if content.is_empty() {
        return Vec::new();
    }

    let instruction = format!("What is the content of chunk {idx}?\n\n{first}", idx = chunk.chunk_index + 1, first = content.lines().next().unwrap_or(content));

    let response = content
        .lines()
        .skip_while(|l| l.is_empty())
        .take_while(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    vec![Sample {
        instruction,
        response: if response.is_empty() { content.to_string() } else { response },
        source: SampleSource::Chunk {
            chunk_id: chunk.id.clone(),
            event_id: chunk.event_id.clone(),
        },
        weight: 0.3,
        tags: vec!["chunk".to_string()],
        provenance_id: format!("chunk:{}", chunk.id),
    }]
}

/// Format a resolved annotation as instruction/response pair.
fn format_annotation(annotation: &Annotation) -> Vec<Sample> {
    let mut samples = Vec::new();

    if annotation.status != "resolved" {
        return samples;
    }

    let reason = match &annotation.resolution_reason {
        Some(r) => r.clone(),
        None => return samples,
    };

    let instruction = format!(
        "What decision was made about the {doc} annotation ({kind})?\n\n{message}",
        doc = annotation.doc_name,
        kind = annotation.kind,
        message = annotation.message
    );

    samples.push(Sample {
        instruction,
        response: reason,
        source: SampleSource::Annotation {
            doc_name: annotation.doc_name.clone(),
            annotation_id: annotation.id.clone(),
            annotation_kind: annotation.kind.clone(),
        },
        weight: annotation.confidence,
        tags: vec!["annotation".to_string(), annotation.doc_name.clone()],
        provenance_id: format!("annotation:{}:{}", annotation.doc_name, annotation.id),
    });

    samples
}

/// Apply weighting to samples based on confidence decay and tag frequency.
pub fn apply_weighting(
    samples: Vec<Sample>,
    recency_seconds: f64,
    decay_half_life: f64,
    tag_boost: &std::collections::HashMap<String, f64>,
) -> Vec<Sample> {
    let decay = 2.0f64.powf(-(recency_seconds / decay_half_life));

    samples
        .into_iter()
        .map(|mut sample| {
            // Apply time decay
            let decayed = sample.weight * decay;

            // Apply tag boost
            let mut tag_multiplier = 1.0;
            for tag in &sample.tags {
                if let Some(boost) = tag_boost.get(tag) {
                    tag_multiplier *= boost;
                }
            }

            sample.weight = (decayed * tag_multiplier).clamp(0.01, 1.0);
            sample
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::KnowledgeEntry;
    use agent_context::KnowledgeKind;

    fn make_sample_data() -> ExtractedData {
        ExtractedData {
            knowledge_entries: vec![
                KnowledgeEntry {
                    id: 1,
                    content: "Always use thiserror for library errors.\n\nNever use unwrap in library code.".to_string(),
                    kind: KnowledgeKind::Instruction,
                    tags: vec!["rust".to_string(), "error-handling".to_string()],
                    metadata: serde_json::json!({}),
                    weight: 0.9,
                    source_type: "user".to_string(),
                    source_id: "test-1".to_string(),
                    provenance_id: "prov-1".to_string(),
                    active: true,
                    created_at: 1000000,
                },
                KnowledgeEntry {
                    id: 2,
                    content: "Use snake_case for function names.".to_string(),
                    kind: KnowledgeKind::Pattern,
                    tags: vec!["rust".to_string(), "naming".to_string()],
                    metadata: serde_json::json!({}),
                    weight: 0.7,
                    source_type: "agent".to_string(),
                    source_id: "test-2".to_string(),
                    provenance_id: "prov-2".to_string(),
                    active: true,
                    created_at: 1000000,
                },
            ],
            events: vec![
                LogEvent {
                    id: "evt-1".to_string(),
                    content: "Updated Cargo.toml".to_string(),
                    source: "file".to_string(),
                    meta: None,
                    timestamp: 1000000,
                },
            ],
            annotations: vec![],
            chunks: vec![],
            sample_count: 3,
        }
    }

    #[test]
    fn format_samples_returns_samples_for_knowledge_entries() {
        let data = make_sample_data();
        let samples = format_samples(&data);
        assert!(!samples.is_empty());
        assert!(samples.iter().any(|s| s.provenance_id.starts_with("knowledge:")));
    }

    #[test]
    fn format_samples_includes_events() {
        let data = make_sample_data();
        let samples = format_samples(&data);
        let event_samples: Vec<_> = samples.iter().filter(|s| {
            matches!(s.source, SampleSource::LogEvent { .. })
        }).collect();
        assert_eq!(event_samples.len(), 1);
    }

    #[test]
    fn format_empty_data_returns_empty() {
        let data = ExtractedData::default();
        let samples = format_samples(&data);
        assert!(samples.is_empty());
    }

    #[test]
    fn format_knowledge_entry_creates_instruction_response() {
        let entry = KnowledgeEntry {
            id: 1,
            content: "Test content here.".to_string(),
            kind: KnowledgeKind::Pattern,
            tags: vec!["test".to_string()],
            metadata: serde_json::json!({}),
            weight: 1.0,
            source_type: "user".to_string(),
            source_id: "test".to_string(),
            provenance_id: "prov-1".to_string(),
            active: true,
            created_at: 1000000,
        };
        let samples = format_knowledge_entry(&entry);
        assert_eq!(samples.len(), 1);
        assert!(samples[0].instruction.contains("pattern"));
        assert!(samples[0].response.contains("Test content"));
    }

    #[test]
    fn format_log_event_creates_sample() {
        let event = LogEvent {
            id: "evt-1".to_string(),
            content: "Something happened.".to_string(),
            source: "file".to_string(),
            meta: None,
            timestamp: 1000000,
        };
        let samples = format_log_event(&event);
        assert_eq!(samples.len(), 1);
        assert!(matches!(samples[0].source, SampleSource::LogEvent { .. }));
    }

    #[test]
    fn format_log_event_skips_empty_content() {
        let event = LogEvent {
            id: "evt-1".to_string(),
            content: "   ".to_string(),
            source: "file".to_string(),
            meta: None,
            timestamp: 1000000,
        };
        let samples = format_log_event(&event);
        assert!(samples.is_empty());
    }

    #[test]
    fn format_annotation_includes_resolved_only() {
        let resolved = Annotation {
            id: "ann-1".to_string(),
            doc_name: "test.md".to_string(),
            kind: "note".to_string(),
            message: "Test annotation".to_string(),
            status: "resolved".to_string(),
            resolution_reason: Some("Fixed the issue".to_string()),
            confidence: 0.85,
            created_at: 1000000,
        };
        let samples = format_annotation(&resolved);
        assert_eq!(samples.len(), 1);
        assert!(matches!(samples[0].source, SampleSource::Annotation { .. }));

        let open = Annotation {
            id: "ann-2".to_string(),
            doc_name: "test.md".to_string(),
            kind: "note".to_string(),
            message: "Open annotation".to_string(),
            status: "open".to_string(),
            resolution_reason: None,
            confidence: 0.5,
            created_at: 1000000,
        };
        let samples = format_annotation(&open);
        assert!(samples.is_empty());
    }

    #[test]
    fn apply_weighting_reduces_weight_with_decay() {
        let samples = vec![
            Sample {
                instruction: "test".to_string(),
                response: "test".to_string(),
                source: SampleSource::KnowledgeEntry {
                    entry_id: 1,
                    kind: "Pattern".to_string(),
                },
                weight: 1.0,
                tags: vec!["test".to_string()],
                provenance_id: "test".to_string(),
            },
        ];

        let decayed = apply_weighting(samples, 86400.0, 86400.0, &std::collections::HashMap::new());
        assert!(decayed[0].weight < 1.0);
        assert!(decayed[0].weight > 0.0);
    }

    #[test]
    fn apply_weighting_no_decay_when_recent() {
        let samples = vec![
            Sample {
                instruction: "test".to_string(),
                response: "test".to_string(),
                source: SampleSource::KnowledgeEntry {
                    entry_id: 1,
                    kind: "Pattern".to_string(),
                },
                weight: 0.5,
                tags: vec!["test".to_string()],
                provenance_id: "test".to_string(),
            },
        ];

        let result = apply_weighting(samples, 0.0, 86400.0, &std::collections::HashMap::new());
        assert_eq!(result[0].weight, 0.5);
    }

    #[test]
    fn apply_weighting_applies_tag_boost() {
        let mut boost = std::collections::HashMap::new();
        boost.insert("boosted".to_string(), 2.0);

        let samples = vec![
            Sample {
                instruction: "test".to_string(),
                response: "test".to_string(),
                source: SampleSource::KnowledgeEntry {
                    entry_id: 1,
                    kind: "Pattern".to_string(),
                },
                weight: 0.5,
                tags: vec!["boosted".to_string()],
                provenance_id: "test".to_string(),
            },
        ];

        let result = apply_weighting(samples, 0.0, 86400.0, &boost);
        assert_eq!(result[0].weight, 1.0); // 0.5 * 2.0 = 1.0, capped at 1.0
    }

    #[test]
    fn sample_source_provenance_formats_correctly() {
        let ke = SampleSource::KnowledgeEntry {
            entry_id: 42,
            kind: "Pattern".to_string(),
        };
        assert_eq!(ke.to_provenance(), "knowledge:42");

        let le = SampleSource::LogEvent {
            event_id: "evt-99".to_string(),
            source: "file".to_string(),
        };
        assert_eq!(le.to_provenance(), "event:evt-99");

        let ch = SampleSource::Chunk {
            chunk_id: "chunk-1".to_string(),
            event_id: "evt-1".to_string(),
        };
        assert_eq!(ch.to_provenance(), "chunk:chunk-1");

        let an = SampleSource::Annotation {
            doc_name: "test.md".to_string(),
            annotation_id: "ann-1".to_string(),
            annotation_kind: "note".to_string(),
        };
        assert_eq!(an.to_provenance(), "annotation:test.md:ann-1");
    }
}
