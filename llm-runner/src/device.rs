use crate::error::RunnerError;
use candle_core::Device;
use tracing::debug;

/// Device backend for tensor computation.
///
/// CUDA/CPU/MKL backend selection.
pub struct DeviceBackend {
    pub preference: String,
    pub device: Device,
}

impl DeviceBackend {
    pub fn new(preference: impl Into<String>) -> Self {
        Self {
            preference: preference.into(),
            device: Device::Cpu,
        }
    }

    /// Select device based on preference.
    pub fn select(&mut self) -> Result<(), RunnerError> {
        match self.preference.as_str() {
            "cuda" => {
                self.device = Device::cuda_if_available(0)
                    .map_err(|e: candle_core::Error| RunnerError::Device(e.to_string()))?
            }
            "cpu" => {
                self.device = Device::Cpu;
            }
            "mkl" => {
                self.device = Device::Cpu; // MKL requires intel-mkl-src setup
            }
            "accelerate" => {
                self.device = Device::Cpu; // accelerate-src for macOS not available on linux
            }
            _ => {
                self.device = Device::Cpu;
            }
        }
        debug!(preference = %self.preference, device = %self.info().unwrap_or_default(), "Device backend: selected");
        Ok(())
    }

    /// Get device info.
    pub fn info(&self) -> Result<String, RunnerError> {
        Ok(match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(ordinal) => format!("cuda:{ordinal:?}"),
            Device::Metal(_) => "metal".to_string(),
        })
    }

    pub fn is_available(&self) -> Result<bool, RunnerError> {
        Ok(match self.device {
            Device::Cpu => true,
            Device::Cuda(_) => false,  // cuda check requires runtime
            Device::Metal(_) => false, // metal requires macOS
        })
    }
}
