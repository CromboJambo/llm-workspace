//! crabjar-llm-runner: LLM inference engine for tensor computation and model loading.
//!
//! Separate workspace member that eventually becomes independent.
//! Interface boundary: consumes WeightManifest from safetensors, emits InferenceResponse to guard.
//!
//! ## Modules
//!
//! - `model-loader`: consumes WeightManifest JSON → loads tensors
//! - `inference-engine`: actual tensor computation
//! - `tokenizer`: prompt encoding
//! - `device`: CUDA/CPU/MKL backend selection
//! - `runner`: external runner bridge (endpoint/protocol)
//! - `plug-in`: implements InferenceRequest/Response protocol
//! - `model`: Model struct with per-layer KV cache, prefill/decode loop

pub mod device;
pub mod error;
pub mod gguf_weight_loader;
pub mod inference_engine;
pub mod kernel;
pub mod model;
pub mod model_loader;
pub mod model_manager;
pub mod plug_in;
pub mod registry;
pub mod runner;
pub mod tokenizer;
pub mod transformer;

pub use device::DeviceBackend;
pub use error::{Result, RunnerError};
pub use gguf_weight_loader::{load_gguf_tensor, load_gguf_weights, GgufWeights};
pub use inference_engine::InferenceEngine;
pub use kernel::{AttentionKernel, CpuAttentionKernel, GemmKernel, GemmBuilder};
pub use kernel::{DeviceBuffer, HostTmaDescriptor, Kvcache};
pub use model::{CpuModel, Model, ModelConfig};
pub use model_loader::ModelLoader;
pub use model_manager::{ModelManager, ModelSpec, PreloadConfig, PreloadStats};
pub use plug_in::PlugInProtocol;
pub use registry::{DiscoveredModel, ModelDiscovery, ModelEntry, ModelFormat, Registry};
pub use runner::RunnerBridge;
pub use tokenizer::Tokenizer;
pub use transformer::{LlamaModel, SamplingConfig, sample, argmax, load_tokenizer_from_gguf, GgufTokenizerConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use crabjar_llm_plug_in::protocol::{InferenceRequest, InferenceResponse, RunnerConfig};
    use tempfile::tempdir;

    // ── DeviceBackend ──────────────────────────────────────────────────

    #[test]
    fn device_backend_new_defaults_to_cpu() {
        let backend = DeviceBackend::new("cuda");
        assert_eq!(backend.preference, "cuda");
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    #[test]
    fn device_backend_select_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    #[test]
    fn device_backend_select_mkl() {
        let mut backend = DeviceBackend::new("mkl");
        backend.select().unwrap();
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    #[test]
    fn device_backend_select_accelerate() {
        let mut backend = DeviceBackend::new("accelerate");
        backend.select().unwrap();
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    #[test]
    fn device_backend_select_unknown_falls_back_to_cpu() {
        let mut backend = DeviceBackend::new("vulkan");
        backend.select().unwrap();
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    #[test]
    fn device_backend_info_cpu() {
        let backend = DeviceBackend::new("cpu");
        let info = backend.info().unwrap();
        assert_eq!(info, "cpu");
    }

    #[test]
    fn device_backend_is_available_cpu() {
        let backend = DeviceBackend::new("cpu");
        assert!(backend.is_available().unwrap());
    }

    // ── InferenceEngine ────────────────────────────────────────────────

    #[test]
    fn inference_engine_new_sets_device_and_dtype() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::F32);
        assert!(matches!(engine.device, candle_core::Device::Cpu));
        assert_eq!(engine.dtype, DType::F32);
    }

    #[test]
    fn inference_engine_device_info_cpu() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::F32);
        assert_eq!(engine.device_info().unwrap(), "cpu");
    }

    #[test]
    fn inference_engine_dtype_info_f32() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::F32);
        assert_eq!(engine.dtype_info().unwrap(), "F32");
    }

    #[test]
    fn inference_engine_dtype_info_f16() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::F16);
        assert_eq!(engine.dtype_info().unwrap(), "F16");
    }

    #[test]
    fn inference_engine_dtype_info_i64() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::I64);
        assert_eq!(engine.dtype_info().unwrap(), "I64");
    }

    #[test]
    fn inference_engine_dtype_info_i32() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::I32);
        assert_eq!(engine.dtype_info().unwrap(), "I32");
    }

    #[test]
    fn inference_engine_dtype_info_u8() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::U8);
        assert_eq!(engine.dtype_info().unwrap(), "U8");
    }

    #[test]
    fn inference_engine_materialize_tensor_fails_missing_file() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::F32);
        let result = engine.materialize_tensor("/nonexistent/tensor.bin", "weight");
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::Asset(_) => {}
            other => panic!("expected Asset error, got {:?}", other),
        }
    }

    // ── RunnerBridge ───────────────────────────────────────────────────

    #[test]
    fn runner_bridge_new_sets_config_and_endpoint() {
        let config = RunnerConfig::default();
        let bridge = RunnerBridge::new(config.clone());
        assert_eq!(bridge.endpoint, config.endpoint);
    }

    #[test]
    fn runner_bridge_send_request_returns_error() {
        let config = RunnerConfig::default();
        let bridge = RunnerBridge::new(config);
        let request = InferenceRequest::new("prov-1", "test-model", "hello");
        let result = bridge.send_request(request);
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::Internal(msg) => assert!(msg.contains("external runner")),
            other => panic!("expected Internal error, got {:?}", other),
        }
    }

    #[test]
    fn runner_bridge_receive_response_returns_error() {
        let config = RunnerConfig::default();
        let bridge = RunnerBridge::new(config);
        let result = bridge.receive_response();
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::Internal(msg) => assert!(msg.contains("external runner")),
            other => panic!("expected Internal error, got {:?}", other),
        }
    }

    #[test]
    fn runner_bridge_update_config_updates_endpoint() {
        let mut bridge = RunnerBridge::new(RunnerConfig::default());
        let new_config = RunnerConfig {
            endpoint: "http://localhost:8080".into(),
            ..Default::default()
        };
        bridge.update_config(new_config).unwrap();
        assert_eq!(bridge.endpoint, "http://localhost:8080");
    }

    #[test]
    fn runner_bridge_config_info_formats_correctly() {
        let config = RunnerConfig {
            runner_name: "test-runner".into(),
            runner_type: "http".into(),
            endpoint: "http://localhost:3000".into(),
            protocol: "json".into(),
            ..RunnerConfig::default()
        };
        let bridge = RunnerBridge::new(config);
        let info = bridge.config_info().unwrap();
        assert!(info.contains("test-runner"));
        assert!(info.contains("http"));
        assert!(info.contains("http://localhost:3000"));
        assert!(info.contains("json"));
    }

    // ── Tokenizer ──────────────────────────────────────────────────────

    #[test]
    fn tokenizer_new_sets_model_name() {
        let tok = Tokenizer::new("gpt2");
        assert_eq!(tok.model, "gpt2");
        assert!(tok.tokenizer.is_none());
    }

    #[test]
    fn tokenizer_encode_without_init_fails() {
        let tok = Tokenizer::new("gpt2");
        let result = tok.encode("hello world");
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::Tokenizer(msg) => assert!(msg.contains("not initialized")),
            other => panic!("expected Tokenizer error, got {:?}", other),
        }
    }

    #[test]
    fn tokenizer_decode_without_init_fails() {
        let tok = Tokenizer::new("gpt2");
        let result = tok.decode(&[1, 2, 3]);
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::Tokenizer(msg) => assert!(msg.contains("not initialized")),
            other => panic!("expected Tokenizer error, got {:?}", other),
        }
    }

    #[test]
    fn tokenizer_token_count_without_init_fails() {
        let tok = Tokenizer::new("gpt2");
        let result = tok.token_count("hello world");
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::Tokenizer(msg) => assert!(msg.contains("not initialized")),
            other => panic!("expected Tokenizer error, got {:?}", other),
        }
    }

    // ── ModelLoader ────────────────────────────────────────────────────

    #[test]
    fn model_loader_new_wraps_connection() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let loader = ModelLoader::new(conn);
        let result = loader.load_manifest("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn model_loader_verify_checksum_no_such_weight() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let loader = ModelLoader::new(conn);
        let result = loader.verify_checksum("nonexistent-id", "abc123");
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn model_loader_verify_checksum_matches() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let weight_id = crabjar_safetensors::schema::insert_model_weights(
            &conn,
            "test-model",
            "test/repo",
            "/tmp/model.safetensors",
            10,
            "F32",
            "CPU",
            1000,
            "abc123",
            "{}",
        )
        .unwrap();

        let loader = ModelLoader::new(conn);
        let result = loader.verify_checksum(&weight_id, "abc123").unwrap();
        assert!(result);
    }

    #[test]
    fn model_loader_verify_checksum_mismatch() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let weight_id = crabjar_safetensors::schema::insert_model_weights(
            &conn,
            "test-model",
            "test/repo",
            "/tmp/model.safetensors",
            10,
            "F32",
            "CPU",
            1000,
            "abc123",
            "{}",
        )
        .unwrap();

        let loader = ModelLoader::new(conn);
        let result = loader.verify_checksum(&weight_id, "wrong").unwrap();
        assert!(!result);
    }

    #[test]
    fn model_loader_list_active_returns_weights() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        crabjar_safetensors::schema::insert_model_weights(
            &conn,
            "model-a",
            "test/repo",
            "/tmp/a.safetensors",
            10,
            "F32",
            "CPU",
            1000,
            "aaa",
            "{}",
        )
        .unwrap();

        let loader = ModelLoader::new(conn);
        let active = loader.list_active(10).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].model_name, "model-a");
    }

    // ── PlugInProtocol ─────────────────────────────────────────────────

    #[test]
    fn plugin_protocol_new_sets_config() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let config = RunnerConfig::default();
        let plugin = PlugInProtocol::new(conn, config.clone());
        assert_eq!(plugin.runner_config.runner_name, config.runner_name);
    }

    #[test]
    fn plugin_protocol_create_request_defaults() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let config = RunnerConfig::default();
        let plugin = PlugInProtocol::new(conn, config);
        let request = plugin.create_request("hello world", "prov-1");

        assert_eq!(request.provenance_id, "prov-1");
        assert_eq!(request.model_name, "default");
        assert_eq!(request.prompt, "hello world");
        assert_eq!(request.max_tokens, 1024);
        assert_eq!(request.temperature, 0.7);
        assert_eq!(request.context.len(), 0);
        assert_eq!(request.skill_refs.len(), 0);
    }

    #[test]
    fn plugin_protocol_create_response_defaults() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let config = RunnerConfig::default();
        let plugin = PlugInProtocol::new(conn, config);
        let response = plugin.create_response("prov-1", "output text", "test-model");

        assert_eq!(response.provenance_id, "prov-1");
        assert_eq!(response.model_name, "test-model");
        assert_eq!(response.output, "output text");
        assert_eq!(response.confidence, 0.5);
        assert_eq!(response.exit_code, 0);
        assert_eq!(response.tokens.len(), 0);
    }

    #[test]
    fn plugin_protocol_update_config() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        crabjar_safetensors::schema::init_db(&conn).unwrap();

        let mut plugin = PlugInProtocol::new(
            conn,
            RunnerConfig {
                runner_name: "old".into(),
                ..RunnerConfig::default()
            },
        );

        let new_config = RunnerConfig {
            runner_name: "new-runner".into(),
            endpoint: "http://localhost:9000".into(),
            ..RunnerConfig::default()
        };
        plugin.update_config(new_config).unwrap();
        assert_eq!(plugin.runner_config.runner_name, "new-runner");
        assert_eq!(plugin.runner_config.endpoint, "http://localhost:9000");
    }

    // ── InferenceRequest builder ───────────────────────────────────────

    #[test]
    fn inference_request_new_defaults() {
        let req = InferenceRequest::new("prov-1", "gpt2", "hello");
        assert_eq!(req.provenance_id, "prov-1");
        assert_eq!(req.model_name, "gpt2");
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.weight_id, "");
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.temperature, 0.7);
        assert_eq!(req.device, "CPU");
        assert_eq!(req.dtype, "F32");
    }

    #[test]
    fn inference_request_builder_chain() {
        let req = InferenceRequest::new("prov-2", "llama", "world")
            .weight_id("w-1")
            .device("cuda")
            .dtype("F16")
            .max_tokens(2048)
            .temperature(0.9)
            .context(vec!["ctx1", "ctx2"])
            .skill_refs(vec!["skill-a", "skill-b"]);

        assert_eq!(req.weight_id, "w-1");
        assert_eq!(req.device, "cuda");
        assert_eq!(req.dtype, "F16");
        assert_eq!(req.max_tokens, 2048);
        assert_eq!(req.temperature, 0.9);
        assert_eq!(req.context.len(), 2);
        assert_eq!(req.skill_refs.len(), 2);
    }

    #[test]
    fn inference_request_temperature_clamped_to_range() {
        let req = InferenceRequest::new("p", "m", "t").temperature(-1.0);
        assert_eq!(req.temperature, 0.0);

        let req = InferenceRequest::new("p", "m", "t").temperature(3.0);
        assert_eq!(req.temperature, 2.0);
    }

    // ── InferenceResponse builder ──────────────────────────────────────

    #[test]
    fn inference_response_new_defaults() {
        let resp = InferenceResponse::new("prov-1", "gpt2", "output");
        assert_eq!(resp.provenance_id, "prov-1");
        assert_eq!(resp.model_name, "gpt2");
        assert_eq!(resp.output, "output");
        assert_eq!(resp.confidence, 0.5);
        assert_eq!(resp.exit_code, 0);
        assert_eq!(resp.weight_id, "");
        assert_eq!(resp.tokens.len(), 0);
        assert_eq!(resp.output_hash, "");
        assert!(resp.skill_residue.is_none());
    }

    #[test]
    fn inference_response_builder_chain() {
        let resp = InferenceResponse::new("prov-1", "gpt2", "output")
            .weight_id("w-1")
            .tokens(vec!["a", "b"])
            .confidence(0.95)
            .output_hash("hash123")
            .skill_residue("residue")
            .exit_code(1);

        assert_eq!(resp.weight_id, "w-1");
        assert_eq!(resp.tokens.len(), 2);
        assert_eq!(resp.confidence, 0.95);
        assert_eq!(resp.output_hash, "hash123");
        assert_eq!(resp.skill_residue, Some("residue".to_string()));
        assert_eq!(resp.exit_code, 1);
    }

    #[test]
    fn inference_response_confidence_clamped_to_range() {
        let resp = InferenceResponse::new("p", "m", "o").confidence(-1.0);
        assert_eq!(resp.confidence, 0.0);

        let resp = InferenceResponse::new("p", "m", "o").confidence(2.0);
        assert_eq!(resp.confidence, 1.0);
    }

    // ── RunnerConfig builder ───────────────────────────────────────────

    #[test]
    fn runner_config_default_values() {
        let config = RunnerConfig::default();
        assert_eq!(config.runner_name, "default");
        assert_eq!(config.runner_type, "external");
        assert_eq!(config.endpoint, "");
        assert_eq!(config.protocol, "json");
        assert_eq!(config.device_preference, "CPU");
        assert_eq!(config.dtype_preference, "F32");
        assert_eq!(config.max_tokens_default, 1024);
        assert_eq!(config.temperature_default, 0.7);
    }

    #[test]
    fn runner_config_builder_chain() {
        let config = RunnerConfig::default()
            .with_runner_name("custom")
            .with_runner_type("http")
            .with_endpoint("http://localhost:3000")
            .with_protocol("grpc")
            .with_device("cuda")
            .with_dtype("F16");

        assert_eq!(config.runner_name, "custom");
        assert_eq!(config.runner_type, "http");
        assert_eq!(config.endpoint, "http://localhost:3000");
        assert_eq!(config.protocol, "grpc");
        assert_eq!(config.device_preference, "cuda");
        assert_eq!(config.dtype_preference, "F16");
    }

    // ── RunnerError conversions ────────────────────────────────────────

    #[test]
    fn runner_error_sqlite_from_rusqlite() {
        let err = RunnerError::Sqlite(rusqlite::Error::QueryReturnedNoRows);
        let msg = err.to_string();
        assert!(
            msg.contains("sqlite")
                || msg.contains("SQLite")
                || msg.contains("query")
                || !msg.is_empty()
        );
    }

    #[test]
    fn runner_error_json_from_serde() {
        let result: std::result::Result<serde_json::Value, RunnerError> =
            serde_json::from_str("not json").map_err(|e: serde_json::Error| RunnerError::from(e));
        assert!(result.is_err());
    }

    #[test]
    fn runner_error_asset_io_error() {
        let err = RunnerError::Asset("file not found".to_string());
        let msg = err.to_string();
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn runner_error_internal() {
        let err = RunnerError::Internal("something broke".to_string());
        let msg = err.to_string();
        assert!(msg.contains("something broke"));
    }

    // ── Error type variants ────────────────────────────────────────────

    #[test]
    fn runner_error_tensor_variant() {
        let err = RunnerError::Tensor("computation failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("computation failed"));
    }

    #[test]
    fn runner_error_model_load_variant() {
        let err = RunnerError::ModelLoad("manifest missing".to_string());
        let msg = err.to_string();
        assert!(msg.contains("manifest missing"));
    }

    #[test]
    fn runner_error_tokenizer_variant() {
        let err = RunnerError::Tokenizer("bpe init failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bpe init failed"));
    }

    #[test]
    fn runner_error_device_variant() {
        let err = RunnerError::Device("cuda unavailable".to_string());
        let msg = err.to_string();
        assert!(msg.contains("cuda unavailable"));
    }

    // ── Tokenizer init_bpe ─────────────────────────────────────────────

    #[test]
    fn tokenizer_init_bpe_gpt2_succeeds() {
        let mut tokenizer = Tokenizer::new("gpt2");
        let result = tokenizer.init_bpe();
        assert!(result.is_ok());
    }

    #[test]
    fn tokenizer_init_bpe_unknown_model_fails() {
        let mut tokenizer = Tokenizer::new("unknown-model-xyz");
        let result = tokenizer.init_bpe();
        assert!(result.is_err());
    }

    #[test]
    fn tokenizer_encode_decode_without_init_fails() {
        let tokenizer = Tokenizer::new("gpt2");
        assert!(tokenizer.encode("hello").is_err());
        assert!(tokenizer.decode(&[1, 2, 3]).is_err());
        assert!(tokenizer.token_count("hello").is_err());
    }
}
