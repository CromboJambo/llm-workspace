use crate::error::RunnerError;
use crabjar_llm_plug_in::protocol::{InferenceRequest, InferenceResponse, RunnerConfig};
use tracing::debug;

/// Runner bridge for external LLM runtime.
///
/// external runner endpoint/protocol bridge.
pub struct RunnerBridge {
    pub config: RunnerConfig,
    pub endpoint: String,
}

impl RunnerBridge {
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config: config.clone(),
            endpoint: config.endpoint.clone(),
        }
    }

    /// Send inference request to external runner.
    pub fn send_request(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, RunnerError> {
        // External runner endpoint bridge
        // protocol: JSON via HTTP or local pipe
        debug!(
            endpoint = %self.endpoint,
            model_name = %request.model_name,
            "Runner bridge: sending inference request"
        );
        Err(RunnerError::Internal(
            "external runner not implemented".to_string(),
        ))
    }

    /// Receive inference response from external runner.
    pub fn receive_response(&self) -> Result<InferenceResponse, RunnerError> {
        Err(RunnerError::Internal(
            "external runner not implemented".to_string(),
        ))
    }

    /// Update runner config.
    pub fn update_config(&mut self, config: RunnerConfig) -> Result<(), RunnerError> {
        self.config = config.clone();
        self.endpoint = config.endpoint.clone();
        Ok(())
    }

    /// Get runner config info.
    pub fn config_info(&self) -> Result<String, RunnerError> {
        Ok(format!(
            "runner_name={}, runner_type={}, endpoint={}, protocol={}",
            self.config.runner_name,
            self.config.runner_type,
            self.config.endpoint,
            self.config.protocol
        ))
    }
}
