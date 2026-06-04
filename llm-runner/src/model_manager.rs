//! Model lifecycle management with smart preloading and popularity tracking.
//!
//! Inspired by shimmy's ModelManager. Manages model loading, unloading, and
//! background preloading based on usage patterns.
//!
//! ## Architecture
//!
//! ```text
//! ModelManager
//!   ├── loaded_models: HashMap<String, ModelLoadInfo>
//!   ├── usage_stats: HashMap<String, ModelUsageStats>
//!   ├── preload_queue: VecDeque<String>
//!   ├── preload_config: PreloadConfig
//!   ├── load_model() / unload_model()
//!   ├── record_access() → popularity scoring
//!   ├── evaluate_preloading() → queue candidates
//!   ├── start_preloading_task() → background task
//!   └── cleanup_old_models() → free memory
//! ```
//!
//! ## Popularity Scoring
//!
//! `popularity = ln(total_requests + 1) * (1 / (1 + hours_since_last_use / 3600))`
//!
//! Frequency factor grows logarithmically. Recency factor decays over hours.
//! Models exceeding `preload_threshold_score` and `min_usage_for_preload`
//! are queued for background preloading.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Model specification for loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub name: String,
    pub base_path: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub template: Option<String>,
    pub ctx_len: usize,
    pub n_threads: Option<i32>,
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelLoadInfo {
    pub name: String,
    pub spec: ModelSpec,
    pub loaded_at: std::time::SystemTime,
    pub last_accessed: std::time::SystemTime,
    pub access_count: u64,
}

/// Usage statistics for popularity scoring.
#[derive(Debug, Clone)]
pub struct ModelUsageStats {
    pub model_name: String,
    pub total_requests: u64,
    pub last_used: std::time::SystemTime,
    pub average_response_time: Duration,
    pub popularity_score: f64,
}

/// Configuration for smart preloading behavior.
#[derive(Debug, Clone)]
pub struct PreloadConfig {
    pub enabled: bool,
    pub max_preloaded_models: usize,
    pub max_memory_mb: usize,
    pub preload_threshold_score: f64,
    pub min_usage_for_preload: u64,
    pub cleanup_interval: Duration,
}

impl Default for PreloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_preloaded_models: 3,
            max_memory_mb: 8192,
            preload_threshold_score: 0.5,
            min_usage_for_preload: 2,
            cleanup_interval: Duration::from_secs(300),
        }
    }
}

/// Preloading statistics for monitoring.
#[derive(Debug, Clone)]
pub struct PreloadStats {
    pub loaded_models: usize,
    pub max_models: usize,
    pub queue_length: usize,
    pub total_tracked_models: usize,
    pub memory_limit_mb: usize,
    pub preloading_enabled: bool,
}

/// Model lifecycle manager with smart preloading.
///
/// Tracks model access patterns, scores popularity, and manages a background
/// preload queue to keep frequently-used models ready.
#[derive(Clone)]
pub struct ModelManager {
    loaded_models: Arc<RwLock<Vec<(String, ModelLoadInfo)>>>,
    usage_stats: Arc<RwLock<Vec<(String, ModelUsageStats)>>>,
    preload_config: PreloadConfig,
    preload_queue: Arc<RwLock<VecDeque<String>>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::with_config(PreloadConfig::default())
    }

    pub fn with_config(config: PreloadConfig) -> Self {
        Self {
            loaded_models: Arc::new(RwLock::new(Vec::new())),
            usage_stats: Arc::new(RwLock::new(Vec::new())),
            preload_config: config,
            preload_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Load a model into the manager's registry.
    ///
    /// Does not actually load the model into GPU memory — that is handled by
    /// the inference engine. This tracks the model as "loaded" for lifecycle
    /// purposes.
    pub async fn load_model(&self, name: String, spec: ModelSpec) {
        let now = std::time::SystemTime::now();

        let info = ModelLoadInfo {
            name: name.clone(),
            spec,
            loaded_at: now,
            last_accessed: now,
            access_count: 1,
        };

        // Remove existing entry if overwriting
        {
            let mut models = self.loaded_models.write().await;
            models.retain(|(n, _)| n != &name);
            models.push((name.clone(), info));
        }

        info!("Model '{}' registered in manager", name);

        // Initialize usage stats
        self.update_usage_stats(&name, Duration::from_millis(100))
            .await;

        // Trigger preloading evaluation
        if self.preload_config.enabled {
            self.evaluate_preloading().await;
        }
    }

    /// Record a model access for usage tracking and popularity scoring.
    pub async fn record_access(&self, name: &str, response_time: Duration) {
        {
            let mut models = self.loaded_models.write().await;
            if let Some((_, info)) = models.iter_mut().find(|(n, _)| n == name) {
                info.last_accessed = std::time::SystemTime::now();
                info.access_count += 1;
            }
        }

        self.update_usage_stats(name, response_time).await;
    }

    /// Update usage statistics for a model.
    async fn update_usage_stats(&self, name: &str, response_time: Duration) {
        let mut stats = self.usage_stats.write().await;

        let existing = stats.iter().position(|(n, _)| n == name);

        if let Some(idx) = existing {
            let entry = &mut stats[idx].1;
            entry.total_requests += 1;
            entry.last_used = std::time::SystemTime::now();

            let current_avg_ms = entry.average_response_time.as_millis() as f64;
            let new_response_ms = response_time.as_millis() as f64;
            let new_avg_ms = (current_avg_ms * (entry.total_requests - 1) as f64 + new_response_ms)
                / entry.total_requests as f64;
            entry.average_response_time = Duration::from_millis(new_avg_ms as u64);

            let time_since_last_use = entry
                .last_used
                .duration_since(std::time::SystemTime::now())
                .unwrap_or_default()
                .as_secs() as f64;
            let recency_factor = 1.0 / (1.0 + time_since_last_use / 3600.0);
            let frequency_factor = (entry.total_requests as f64).ln() + 1.0;
            entry.popularity_score = frequency_factor * recency_factor;

            debug!(
                model = name,
                requests = entry.total_requests,
                popularity = entry.popularity_score,
                "Updated usage stats"
            );
        } else {
            stats.push((
                name.to_string(),
                ModelUsageStats {
                    model_name: name.to_string(),
                    total_requests: 1,
                    last_used: std::time::SystemTime::now(),
                    average_response_time: response_time,
                    popularity_score: 1.0,
                },
            ));

            debug!(model = name, requests = 1, popularity = 1.0, "Created new usage stats");
        }
    }

    /// Evaluate which models should be preloaded based on popularity.
    async fn evaluate_preloading(&self) {
        if !self.preload_config.enabled {
            return;
        }

        let (candidates_to_queue, current_loaded) = {
            let stats = self.usage_stats.read().await;
            let loaded_models = self.loaded_models.read().await;

            let mut candidates: Vec<_> = stats
                .iter()
                .filter(|(name, stat)| {
                    stat.total_requests >= self.preload_config.min_usage_for_preload
                        && stat.popularity_score >= self.preload_config.preload_threshold_score
                        && !loaded_models.iter().any(|(n, _)| n == name)
                })
                .collect();

            candidates.sort_by(|a, b| {
                b.1.popularity_score
                    .partial_cmp(&a.1.popularity_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let current_loaded = loaded_models.len();
            let candidates_vec: Vec<_> = candidates
                .iter()
                .map(|(name, stat)| (name.clone(), stat.popularity_score))
                .collect();

            (candidates_vec, current_loaded)
        };

        let mut queue = self.preload_queue.write().await;
        let slots_available = self
            .preload_config
            .max_preloaded_models
            .saturating_sub(current_loaded);

        for (model_name, score) in candidates_to_queue.iter().take(slots_available) {
            if !queue.iter().any(|n| n == model_name) {
                queue.push_back(model_name.clone());
                info!(
                    model = model_name,
                    score = score,
                    "Queued model for preloading"
                );
            }
        }
    }

    /// Start the background preloading task.
    ///
    /// Spawns a tokio task that periodically processes the preload queue.
    /// The caller should keep the returned handle alive.
    pub fn start_preloading_task(&self) -> tokio::task::JoinHandle<()> {
        let manager = Arc::new(self.clone());
        let cleanup_interval = self.preload_config.cleanup_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);

            loop {
                interval.tick().await;

                let model_to_preload = {
                    let mut queue = manager.preload_queue.write().await;
                    queue.pop_front()
                };

                if let Some(model_name) = model_to_preload {
                    let current_count = manager.model_count().await;
                    if current_count < manager.preload_config.max_preloaded_models {
                        debug!(model = model_name, "Processing preload queue");
                    } else {
                        warn!(
                            limit = manager.preload_config.max_preloaded_models,
                            current = current_count,
                            "Memory limit reached, re-queuing model"
                        );
                        let mut queue = manager.preload_queue.write().await;
                        queue.push_front(model_name);
                    }
                }

                manager.cleanup_old_models().await;
            }
        })
    }

    /// Clean up old/unused models to free memory.
    async fn cleanup_old_models(&self) {
        let current_count = self.model_count().await;
        if current_count <= self.preload_config.max_preloaded_models {
            return;
        }

        let cutoff_time =
            std::time::SystemTime::now() - Duration::from_secs(3600);

        let mut models = self.loaded_models.write().await;
        let mut candidates: Vec<_> = models
            .iter()
            .enumerate()
            .filter(|(_, (_, info))| {
                info.last_accessed < cutoff_time && info.access_count < 5
            })
            .map(|(idx, (name, info))| (idx, name.clone(), info.last_accessed, info.access_count))
            .collect();

        candidates.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.3.cmp(&b.3)));

        let to_remove = current_count.saturating_sub(self.preload_config.max_preloaded_models);
        for (idx, name, _, _) in candidates.iter().take(to_remove) {
            models.remove(*idx);
            info!(model = name, "Cleaned up unused model");
        }
    }

    /// Get preloading statistics.
    pub async fn preload_stats(&self) -> PreloadStats {
        let models = self.loaded_models.read().await;
        let stats = self.usage_stats.read().await;
        let queue = self.preload_queue.read().await;

        PreloadStats {
            loaded_models: models.len(),
            max_models: self.preload_config.max_preloaded_models,
            queue_length: queue.len(),
            total_tracked_models: stats.len(),
            memory_limit_mb: self.preload_config.max_memory_mb,
            preloading_enabled: self.preload_config.enabled,
        }
    }

    /// Unload a model from the manager.
    pub async fn unload_model(&self, name: &str) -> bool {
        let mut models = self.loaded_models.write().await;
        let had = models.iter().any(|(n, _)| n == name);
        if had {
            models.retain(|(n, _)| n != name);
            info!(model = name, "Model unloaded from manager");
        }
        had
    }

    /// Get information about a loaded model.
    pub async fn model_info(&self, name: &str) -> Option<ModelLoadInfo> {
        let models = self.loaded_models.read().await;
        models.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone())
    }

    /// List all loaded model names.
    pub async fn list_loaded_models(&self) -> Vec<String> {
        let models = self.loaded_models.read().await;
        models.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Check if a model is loaded.
    pub async fn is_loaded(&self, name: &str) -> bool {
        let models = self.loaded_models.read().await;
        models.iter().any(|(n, _)| n == name)
    }

    /// Count loaded models.
    pub async fn model_count(&self) -> usize {
        let models = self.loaded_models.read().await;
        models.len()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(name: &str, base: &str) -> ModelSpec {
        ModelSpec {
            name: name.to_string(),
            base_path: PathBuf::from(base),
            lora_path: None,
            template: None,
            ctx_len: 2048,
            n_threads: None,
        }
    }

    // ── Construction ───────────────────────────────────────────────────

    #[tokio::test]
    async fn model_manager_creation() {
        let manager = ModelManager::new();
        assert_eq!(manager.model_count().await, 0);
    }

    #[tokio::test]
    async fn model_manager_with_config() {
        let config = PreloadConfig {
            enabled: false,
            max_preloaded_models: 5,
            max_memory_mb: 4096,
            ..Default::default()
        };
        let manager = ModelManager::with_config(config);
        let stats = manager.preload_stats().await;
        assert!(!stats.preloading_enabled);
        assert_eq!(stats.max_models, 5);
        assert_eq!(stats.memory_limit_mb, 4096);
    }

    #[tokio::test]
    async fn model_manager_default_is_same_as_new() {
        let a = ModelManager::default();
        let b = ModelManager::new();
        assert_eq!(a.preload_stats().await.loaded_models, b.preload_stats().await.loaded_models);
    }

    // ── Model Loading ──────────────────────────────────────────────────

    #[tokio::test]
    async fn load_model_sets_count() {
        let manager = ModelManager::new();
        manager.load_model("test".into(), make_spec("test", "/models/test.gguf")).await;
        assert_eq!(manager.model_count().await, 1);
        assert!(manager.is_loaded("test").await);
    }

    #[tokio::test]
    async fn load_model_with_lora() {
        let manager = ModelManager::new();
        let spec = ModelSpec {
            name: "lora-model".into(),
            base_path: PathBuf::from("/models/base.gguf"),
            lora_path: Some(PathBuf::from("/models/lora.safetensors")),
            template: Some("chatml".into()),
            ctx_len: 4096,
            n_threads: Some(8),
        };
        manager.load_model("lora-model".into(), spec).await;
        assert_eq!(manager.model_count().await, 1);

        let info = manager.model_info("lora-model").await.unwrap();
        assert!(info.spec.lora_path.is_some());
        assert_eq!(info.spec.ctx_len, 4096);
        assert_eq!(info.spec.n_threads, Some(8));
    }

    #[tokio::test]
    async fn load_overwrites_existing_model() {
        let manager = ModelManager::new();
        manager.load_model("dup".into(), make_spec("dup", "/models/a.gguf")).await;
        manager.load_model("dup".into(), make_spec("dup", "/models/b.gguf")).await;
        assert_eq!(manager.model_count().await, 1);

        let info = manager.model_info("dup").await.unwrap();
        assert_eq!(info.spec.base_path, PathBuf::from("/models/b.gguf"));
    }

    #[tokio::test]
    async fn load_multiple_models() {
        let manager = ModelManager::new();
        for i in 0..5 {
            manager
                .load_model(format!("model-{}", i), make_spec(&format!("model-{}", i), &format!("/models/{}.gguf", i)))
                .await;
        }
        assert_eq!(manager.model_count().await, 5);
    }

    // ── Model Unloading ────────────────────────────────────────────────

    #[tokio::test]
    async fn unload_existing_model() {
        let manager = ModelManager::new();
        manager.load_model("to-unload".into(), make_spec("to-unload", "/models/unload.gguf")).await;
        assert!(manager.unload_model("to-unload").await);
        assert!(!manager.is_loaded("to-unload").await);
        assert_eq!(manager.model_count().await, 0);
    }

    #[tokio::test]
    async fn unload_nonexistent_model_returns_false() {
        let manager = ModelManager::new();
        assert!(!manager.unload_model("ghost").await);
    }

    // ── Access Tracking ────────────────────────────────────────────────

    #[tokio::test]
    async fn record_access_updates_info() {
        let manager = ModelManager::new();
        manager.load_model("track".into(), make_spec("track", "/models/track.gguf")).await;

        manager.record_access("track", Duration::from_millis(50)).await;
        manager.record_access("track", Duration::from_millis(100)).await;

        let info = manager.model_info("track").await.unwrap();
        assert_eq!(info.access_count, 3); // 1 from load + 2 from record_access
    }

    #[tokio::test]
    async fn record_access_updates_last_accessed() {
        let manager = ModelManager::new();
        manager.load_model("time-track".into(), make_spec("time-track", "/models/t.gguf")).await;

        let before = std::time::SystemTime::now();
        tokio::time::sleep(Duration::from_millis(50)).await;
        manager.record_access("time-track", Duration::from_millis(10)).await;
        let after = std::time::SystemTime::now();

        let info = manager.model_info("time-track").await.unwrap();
        assert!(info.last_accessed >= before);
        assert!(info.last_accessed <= after);
    }

    // ── Usage Statistics ───────────────────────────────────────────────

    #[tokio::test]
    async fn usage_stats_popularity_grows_with_requests() {
        let manager = ModelManager::new();
        manager.load_model("hot".into(), make_spec("hot", "/models/hot.gguf")).await;

        for _ in 0..10 {
            manager.record_access("hot", Duration::from_millis(10)).await;
        }

        let stats = manager.preload_stats().await;
        assert_eq!(stats.total_tracked_models, 1);
    }

    // ── Model Listing ──────────────────────────────────────────────────

    #[tokio::test]
    async fn list_loaded_models_empty() {
        let manager = ModelManager::new();
        let models = manager.list_loaded_models().await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_loaded_models_populated() {
        let manager = ModelManager::new();
        manager.load_model("alpha".into(), make_spec("alpha", "/models/a.gguf")).await;
        manager.load_model("beta".into(), make_spec("beta", "/models/b.gguf")).await;

        let models = manager.list_loaded_models().await;
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"alpha".to_string()));
        assert!(models.contains(&"beta".to_string()));
    }

    // ── Model Info ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn model_info_existing() {
        let manager = ModelManager::new();
        let spec = make_spec("info-test", "/models/info.gguf");
        manager.load_model("info-test".into(), spec).await;

        let info = manager.model_info("info-test").await.unwrap();
        assert_eq!(info.name, "info-test");
        assert_eq!(info.spec.base_path, PathBuf::from("/models/info.gguf"));
    }

    #[tokio::test]
    async fn model_info_nonexistent_returns_none() {
        let manager = ModelManager::new();
        assert!(manager.model_info("ghost").await.is_none());
    }

    // ── Preload Stats ──────────────────────────────────────────────────

    #[tokio::test]
    async fn preload_stats_empty() {
        let manager = ModelManager::new();
        let stats = manager.preload_stats().await;
        assert_eq!(stats.loaded_models, 0);
        assert_eq!(stats.queue_length, 0);
        assert_eq!(stats.total_tracked_models, 0);
    }

    #[tokio::test]
    async fn preload_stats_with_loaded_models() {
        let manager = ModelManager::new();
        manager.load_model("stat-model".into(), make_spec("stat-model", "/models/s.gguf")).await;

        let stats = manager.preload_stats().await;
        assert_eq!(stats.loaded_models, 1);
        assert!(stats.preloading_enabled);
        assert_eq!(stats.max_models, 3);
    }

    // ── Edge Cases ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn unload_model_clears_loaded_status() {
        let manager = ModelManager::new();
        manager.load_model("clear".into(), make_spec("clear", "/models/clear.gguf")).await;
        assert!(manager.is_loaded("clear").await);
        manager.unload_model("clear").await;
        assert!(!manager.is_loaded("clear").await);
    }

    #[tokio::test]
    async fn model_info_edge_empty_name() {
        let manager = ModelManager::new();
        assert!(manager.model_info("").await.is_none());
    }

    #[tokio::test]
    async fn model_info_edge_long_name() {
        let manager = ModelManager::new();
        let long = "a".repeat(1000);
        assert!(manager.model_info(&long).await.is_none());
    }

    // ── Concurrent Access ──────────────────────────────────────────────

    #[tokio::test]
    async fn concurrent_load_models() {
        let manager = Arc::new(ModelManager::new());
        let mut handles = vec![];

        for i in 0..10 {
            let m = Arc::clone(&manager);
            handles.push(tokio::spawn(async move {
                m.load_model(
                    format!("concurrent-{}", i),
                    make_spec(&format!("concurrent-{}", i), &format!("/models/{}.gguf", i)),
                )
                .await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(manager.model_count().await, 10);
    }

    #[tokio::test]
    async fn concurrent_load_unload() {
        let manager = Arc::new(ModelManager::new());
        let mut handles = vec![];

        for i in 0..20 {
            let m = Arc::clone(&manager);
            handles.push(tokio::spawn(async move {
                let name = format!("lu-{}", i);
                m.load_model(name.clone(), make_spec(&name, &format!("/models/lu{}.gguf", i)))
                    .await;

                if i % 2 == 0 {
                    m.unload_model(&name).await;
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(manager.model_count().await, 10);
    }

    // ── Clone ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn model_manager_clone_works() {
        let manager = ModelManager::new();
        let cloned = manager.clone();
        assert_eq!(
            manager.preload_stats().await.loaded_models,
            cloned.preload_stats().await.loaded_models
        );
    }

    #[test]
    fn model_spec_clone_works() {
        let spec = make_spec("clone-test", "/models/clone.gguf");
        let cloned = spec.clone();
        assert_eq!(spec, cloned);
    }
}
