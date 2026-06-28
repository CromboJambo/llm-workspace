# AGENTS.md — crabjar-train-extract

> Purpose: Extract, format, and export knowledge entries as instruction-tuning datasets with provenance tracking.

## Layer

Layer 1: substrate — low-level storage, may depend on layer 0 only.

## Public API

- `run_pipeline()` — full pipeline: extract → format → weight → export
- `quick_export()` — extract all active knowledge and write JSONL
- `weight_samples()` — apply tag boost and recency weighting
- `compute_tag_boost()` — compute tag boost from data distribution
- `compute_recency()` — compute recency score for samples
- `format_samples()` — format raw data into training samples
- `apply_weighting()` — apply weights to samples
- `export()` — export samples in configured format
- `export_jsonl()` — export as JSONL
- `export_safetensors_manifest()` — export safetensors manifest
- `WeightConfig` — weighting configuration
- `ExportConfig`, `ExportFormat` — export configuration
- `DatasetManifest` — export manifest with provenance
- `Sample`, `SampleSource` — training sample types
- `KnowledgeEntry`, `LogEvent`, `Chunk`, `Annotation` — extracted data types
- `TrainExtractError`, `TrainExtractResult` — error types

## Key Files

- `src/lib.rs` — crate entry point, `run_pipeline()`, `quick_export()`, public re-exports
- `src/extract.rs` — `extract()`, `ExtractConfig`, `ExtractedData`, `KnowledgeEntry`, `LogEvent`, `Chunk`, `Annotation`
- `src/format.rs` — `format_samples()`, `apply_weighting()`, `Sample`, `SampleSource`
- `src/export.rs` — `export()`, `export_jsonl()`, `export_safetensors_manifest()`, `ExportConfig`, `ExportFormat`, `DatasetManifest`
- `src/weight.rs` — `weight_samples()`, `WeightConfig`, `compute_tag_boost()`, `compute_recency()`
- `src/error.rs` — `TrainExtractError`, `TrainExtractResult`

## Dependencies

- serde, serde_json, chrono, uuid, rusqlite, thiserror, tracing, sha2, hex
- crabjar-safetensors, agent-context (memory)

## Pitfalls

- **Pipeline is synchronous** — `run_pipeline()` takes `&rusqlite::Connection` directly. No async path. Heavy DB operations block the caller.
- **Weighting is data-driven** — `compute_tag_boost()` analyzes data distribution to derive tag weights. This is a no-op if `tag_boost` is empty in `WeightConfig`.
- **Safetensors export depends on external store** — `export_safetensors_manifest()` requires a `SafetensorsStore`. Pass `None` to skip.
- **Empty dataset error** — `run_pipeline()` returns `TrainExtractError::EmptyDataset` when no samples survive weighting. `quick_export()` propagates this.
- **Provenance tracking** — all exported samples carry `provenance_id` and `source` fields. Never drop these during format transformations.
- **Test DB schema is ad-hoc** — `quick_export` tests create minimal schemas in-memory. Real usage depends on the knowledge store's actual schema.
