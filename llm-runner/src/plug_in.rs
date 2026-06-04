use crate::error::RunnerError;
use crabjar_llm_plug_in::manifest::WeightManifest;
use crabjar_llm_plug_in::protocol::{InferenceRequest, InferenceResponse, RunnerConfig};
use crabjar_safetensors::error::SafetensorsSchemaError;
use crabjar_safetensors::schema::query_model_weights;

/// Plug-in protocol implementation for llm-runner → crabjar interface.
///
/// implements WeightManifest output, InferenceRequest/Response protocol.
/// interface boundary: only interface with crabjar via plug-in protocol.
pub struct PlugInProtocol {
    pub conn: rusqlite::Connection,
    pub runner_config: RunnerConfig,
}

impl PlugInProtocol {
    pub fn new(conn: rusqlite::Connection, runner_config: RunnerConfig) -> Self {
        Self {
            conn,
            runner_config,
        }
    }

    /// Generate weight manifest for external runner.
    pub fn generate_manifest(
        &self,
        model_name: &str,
    ) -> Result<WeightManifest, crabjar_llm_plug_in::PlugInError> {
        crabjar_llm_plug_in::manifest::generate_weight_manifest(&self.conn, model_name)
    }

    /// Create inference request from model manifest.
    pub fn create_request(
        &self,
        prompt: impl Into<String>,
        provenance_id: impl Into<String>,
    ) -> InferenceRequest {
        InferenceRequest::new(
            provenance_id,
            self.runner_config.runner_name.clone(),
            prompt,
        )
        .weight_id(String::new())
        .device(self.runner_config.device_preference.clone())
        .dtype(self.runner_config.dtype_preference.clone())
        .max_tokens(self.runner_config.max_tokens_default)
        .temperature(self.runner_config.temperature_default)
    }

    /// Create inference response from inference outcome.
    pub fn create_response(
        &self,
        provenance_id: impl Into<String>,
        output: impl Into<String>,
        model_name: impl Into<String>,
    ) -> InferenceResponse {
        InferenceResponse::new(provenance_id, model_name, output)
            .confidence(0.5)
            .exit_code(0)
            .output_hash(String::new())
    }

    /// Query active weights for model selection.
    pub fn query_weights(
        &self,
        model_name: &str,
        limit: usize,
    ) -> Result<Vec<crabjar_safetensors::schema::ModelWeightRow>, RunnerError> {
        query_model_weights(&self.conn, model_name, limit).map_err(|e: SafetensorsSchemaError| {
            RunnerError::Sqlite(match e {
                SafetensorsSchemaError::Sqlite(r) => r,
                _ => rusqlite::Error::QueryReturnedNoRows,
            })
        })
    }

    /// Update runner config.
    pub fn update_config(
        &mut self,
        config: RunnerConfig,
    ) -> Result<(), crabjar_llm_plug_in::PlugInError> {
        self.runner_config = config;
        Ok(())
    }
}
