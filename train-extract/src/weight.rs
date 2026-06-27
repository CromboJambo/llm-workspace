use crate::extract::ExtractedData;
use crate::format::{apply_weighting, format_samples, Sample};
use std::collections::HashMap;

/// Configuration for the weighting phase.
#[derive(Debug, Clone)]
pub struct WeightConfig {
    /// Recency in seconds since the oldest entry.
    pub recency_seconds: f64,
    /// Half-life in seconds for confidence decay.
    pub decay_half_life: f64,
    /// Tag-based boost multipliers.
    pub tag_boost: HashMap<String, f64>,
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            recency_seconds: 86400.0 * 30.0, // 30 days
            decay_half_life: 86400.0 * 7.0,  // 7 day half-life
            tag_boost: HashMap::new(),
        }
    }
}

impl WeightConfig {
    /// Set the recency window in days.
    pub fn recency_days(mut self, days: f64) -> Self {
        self.recency_seconds = days * 86400.0;
        self
    }

    /// Set the decay half-life in days.
    pub fn half_life_days(mut self, days: f64) -> Self {
        self.decay_half_life = days * 86400.0;
        self
    }

    /// Add a tag boost multiplier.
    pub fn tag_boost(mut self, tag: &str, boost: f64) -> Self {
        self.tag_boost.insert(tag.to_string(), boost);
        self
    }
}

/// Weight samples from extracted data.
pub fn weight_samples(data: &ExtractedData, config: &WeightConfig) -> Vec<Sample> {
    let raw_samples = format_samples(data);
    apply_weighting(raw_samples, config.recency_seconds, config.decay_half_life, &config.tag_boost)
}

/// Compute tag frequency from extracted data for automatic boosting.
/// Tags appearing in more entries get a higher boost.
pub fn compute_tag_boost(data: &ExtractedData, min_entries: usize) -> HashMap<String, f64> {
    let mut boost = HashMap::new();
    let total = data.sample_count.max(1);

    for (tag, count) in data.tag_distribution() {
        if count >= min_entries {
            let ratio = count as f64 / total as f64;
            // Boost is proportional to frequency, capped at 3.0
            let b = 1.0 + (ratio * 2.0).min(2.0);
            boost.insert(tag.clone(), b);
        }
    }

    boost
}

/// Compute recency seconds from the oldest entry's timestamp.
pub fn compute_recency(data: &ExtractedData) -> f64 {
    let mut max_ts: i64 = 0;
    let mut min_ts: i64 = i64::MAX;

    for entry in &data.knowledge_entries {
        max_ts = max_ts.max(entry.created_at);
        min_ts = min_ts.min(entry.created_at);
    }
    for event in &data.events {
        max_ts = max_ts.max(event.timestamp);
        min_ts = min_ts.min(event.timestamp);
    }

    if min_ts == i64::MAX {
        return 0.0;
    }

    let now = chrono::Utc::now().timestamp();
    (now - min_ts) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::KnowledgeEntry;
    use agent_context::KnowledgeKind;

    fn make_data_with_tags() -> ExtractedData {
        ExtractedData {
            knowledge_entries: vec![
                KnowledgeEntry {
                    id: 1,
                    content: "a".to_string(),
                    kind: KnowledgeKind::Pattern,
                    tags: vec!["rust".to_string(), "pattern".to_string()],
                    metadata: serde_json::json!({}),
                    weight: 1.0,
                    source_type: "user".to_string(),
                    source_id: "t1".to_string(),
                    provenance_id: "p1".to_string(),
                    active: true,
                    created_at: 1000000,
                },
                KnowledgeEntry {
                    id: 2,
                    content: "b".to_string(),
                    kind: KnowledgeKind::Pattern,
                    tags: vec!["rust".to_string(), "naming".to_string()],
                    metadata: serde_json::json!({}),
                    weight: 1.0,
                    source_type: "agent".to_string(),
                    source_id: "t2".to_string(),
                    provenance_id: "p2".to_string(),
                    active: true,
                    created_at: 1000000,
                },
                KnowledgeEntry {
                    id: 3,
                    content: "c".to_string(),
                    kind: KnowledgeKind::Context,
                    tags: vec!["python".to_string()],
                    metadata: serde_json::json!({}),
                    weight: 1.0,
                    source_type: "user".to_string(),
                    source_id: "t3".to_string(),
                    provenance_id: "p3".to_string(),
                    active: true,
                    created_at: 1000000,
                },
            ],
            events: vec![],
            annotations: vec![],
            chunks: vec![],
            sample_count: 3,
        }
    }

    #[test]
    fn compute_tag_boost_gives_higher_boost_to_frequent_tags() {
        let data = make_data_with_tags();
        let boost = compute_tag_boost(&data, 1);
        assert!((*boost.get("rust").unwrap() - 2.333).abs() < 0.01); // 2/3 ratio -> 1 + 2*0.667 = 2.333
        assert!(*boost.get("pattern").unwrap() < *boost.get("rust").unwrap());
        assert!(*boost.get("naming").unwrap() < *boost.get("rust").unwrap());
    }

    #[test]
    fn compute_tag_boost_respects_min_entries() {
        let data = make_data_with_tags();
        let boost = compute_tag_boost(&data, 3);
        assert!(!boost.contains_key("rust")); // only 2 entries, min is 3
        assert!(boost.is_empty());
    }

    #[test]
    fn weight_config_defaults() {
        let config = WeightConfig::default();
        assert!(!config.tag_boost.contains_key("rust"));
        assert_eq!(config.decay_half_life, 86400.0 * 7.0);
    }

    #[test]
    fn weight_config_builder() {
        let config = WeightConfig::default()
            .recency_days(14.0)
            .half_life_days(3.0)
            .tag_boost("rust", 1.5);

        assert_eq!(config.recency_seconds, 14.0 * 86400.0);
        assert_eq!(config.decay_half_life, 3.0 * 86400.0);
        assert_eq!(*config.tag_boost.get("rust").unwrap(), 1.5);
    }
}
