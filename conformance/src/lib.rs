//! PESTI Conformance Testing Framework
//!
//! Differential testing against reference implementations (llama.cpp FFI, candle-core GPU).
//! Implements the pattern from lumen-state and pon: catch silent corruption before it reaches mainline.

use pesti_gguf::GgufHeader;
use std::path::{Path, PathBuf};

/// Configuration for a conformance test run.
#[derive(Debug)]
pub struct ConformanceConfig {
    /// Path to model corpus directory (GGUF files)
    pub corpus_dir: PathBuf,
    /// Reference implementation binary/CLI path for llama.cpp
    pub reference_llama_cpp: Option<PathBuf>,
    /// Expected minimum pass count (floor file threshold)
    pub floor_pass_count: usize,
}

/// Result of running conformance tests.
#[derive(Debug)]
pub struct ConformanceResult {
    /// Total models tested
    pub total_models: usize,
    /// Models that passed (output matches reference within tolerance)
    pub passed: Vec<String>,
    /// Models that failed with divergence details
    pub failures: Vec<FailureInfo>,
}

/// Information about a conformance failure.
#[derive(Debug)]
pub struct FailureInfo {
    /// Model name/path
    pub model_name: String,
    /// Expected output hash (from reference)
    pub expected_hash: String,
    /// Actual pesti output hash
    pub actual_hash: String,
}

/// Run differential conformance tests against a corpus of models.
pub fn run_conformance(
    config: &ConformanceConfig,
) -> Result<ConformanceResult, Box<dyn std::error::Error + Send + Sync>> {
    let mut passed = Vec::new();
    let mut failures = Vec::new();

    // Discover GGUF files in corpus directory
    let models = discover_models(&config.corpus_dir)?;
    tracing::info!("Discovered {} models in {}", models.len(), config.corpus_dir.display());

    for model_path in models {
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        match run_single_model_conformance(&model_path, config) {
            Ok(_) => {
                tracing::info!("✓ PASS: {}", model_name);
                passed.push(model_name);
            }
            Err(e) => {
                tracing::warn!("✗ FAIL: {} - {:?}", model_name, e);
                failures.push(FailureInfo {
                    model_name,
                    expected_hash: "unknown".to_string(), // Would contain hash in real impl
                    actual_hash: "divergent".to_string(),
                });
            }
        }
    }

    let result = ConformanceResult {
        total_models: passed.len() + failures.len(),
        passed,
        failures,
    };

    tracing::info!(
        "Conformance complete: {}/{} passed ({:.1}%)",
        passed.len(),
        result.total_models,
        (passed.len() as f64 / result.total_models as f64) * 100.0
    );

    Ok(result)
}

/// Run conformance test on a single model against reference implementation.
fn run_single_model_conformance(
    model_path: &Path,
    _config: &ConformanceConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // TODO: Implement actual differential testing logic:
    // 1. Load GGUF header to get model config
    // 2. Run pesti inference with small batch (e.g., 5 tokens)
    // 3. If reference_llama_cpp specified, run same input through llama.cpp CLI
    // 4. Compare outputs byte-for-byte or hash comparison
    
    // For now, just verify the model loads without error
    let _header = GgufHeader::parse(model_path)?;

    Ok(())
}

/// Discover all .gguf files in a directory (recursively).
fn discover_models(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
    let mut models = Vec::new();

    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            match path.extension().and_then(|s| s.to_str()) {
                Some("gguf") => models.push(path),
                _ if path.is_dir() => {
                    models.extend(discover_models(&path)?);
                }
                _ => {}
            }
        }
    } else if dir.exists() && path_ends_with(dir, ".gguf") {
        models.push(dir.to_path_buf());
    }

    Ok(models)
}

/// Check if a path ends with a specific suffix.
fn path_ends_with(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|name| name.ends_with(suffix))
        .unwrap_or(false)
}

/// Delta-minimize a divergence between two outputs.
pub fn delta_minimize(expected: &[u8], actual: &[u8]) -> DivergenceInfo {
    // TODO: Implement longest-common-subsequence based diff
    // that produces minimal patch showing where outputs diverge

    let changes: Vec<DivergenceChange> = expected
        .iter()
        .zip(actual.iter())
        .enumerate()
        .filter_map(|(i, (e, a))| {
            if e != a {
                Some(DivergenceChange {
                    position: i,
                    expected: *e,
                    actual: *a,
                })
            } else {
                None
            }
        })
        .collect();

    DivergenceInfo {
        divergence_offset: changes.first().map(|c| c.position),
        changes,
        total_bytes_expected: expected.len(),
        total_bytes_actual: actual.len(),
    }
}

/// Information about where two outputs diverge.
#[derive(Debug)]
pub struct DivergenceInfo {
    pub divergence_offset: Option<usize>,
    pub changes: Vec<DivergenceChange>,
    pub total_bytes_expected: usize,
    pub total_bytes_actual: usize,
}

#[derive(Debug)]
pub struct DivergenceChange {
    pub position: usize,
    pub expected: u8,
    pub actual: u8,
}
