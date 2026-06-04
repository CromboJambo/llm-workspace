use crate::error::RunnerError;
use crate::kernel::{GemmKernel, GemmBuilder, GemmArch, GemmConfig};
use crate::kernel::{AttentionKernel, AttentionArch, AttentionConfig, CpuAttentionKernel};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use half::f16;

/// Inference engine for tensor computation.
///
/// actual tensor computation layer. separate from crabjar host.
pub struct InferenceEngine {
    pub device: candle_core::Device,
    pub dtype: DType,
    gemm: Box<dyn GemmKernel>,
    attention: Box<dyn AttentionKernel>,
}

impl InferenceEngine {
    pub fn new(device: Device, dtype: DType) -> Self {
        let gemm = GemmBuilder::new()
            .with_arch(GemmArch::Tcgen05)
            .with_config(GemmConfig::default())
            .build();
        let attention = Box::new(CpuAttentionKernel::new());
        Self { device, dtype, gemm, attention }
    }

    /// Create engine with a specific GEMM kernel.
    pub fn with_gemm(device: Device, dtype: DType, gemm: Box<dyn GemmKernel>) -> Self {
        let attention = Box::new(CpuAttentionKernel::new());
        Self { device, dtype, gemm, attention }
    }

    /// Run a GEMM operation: C = alpha * A @ B + beta * C.
    ///
    /// A: [m x k] f16, B: [k x n] f16, C: [m x n] f32
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        alpha: f32,
        a: &crate::kernel::DeviceBuffer<f16>,
        b: &crate::kernel::DeviceBuffer<f16>,
        beta: f32,
        c: &mut crate::kernel::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), RunnerError> {
        self.gemm
            .matmul(alpha, a, b, beta, c, m, n, k)
            .map_err(|e| RunnerError::Tensor(format!("GEMM failed: {e}")))
    }

    /// Get the GEMM kernel's target architecture.
    pub fn gemm_arch(&self) -> GemmArch {
        self.gemm.arch()
    }

    /// Check if the GEMM kernel is available on this system.
    pub fn gemm_available(&self) -> bool {
        self.gemm.is_available()
    }

    /// Run inference on a loaded model.
    pub fn infer(&self, model: &impl Module, input: Tensor) -> Result<Tensor, RunnerError> {
        model
            .forward(&input)
            .map_err(|e: candle_core::Error| RunnerError::Tensor(e.to_string()))
    }

    /// Materialize lazy-loaded tensor from manifest.
    pub fn materialize_tensor(
        &self,
        file_path: &str,
        _tensor_name: &str,
    ) -> Result<Tensor, RunnerError> {
        let data = std::fs::read(file_path)
            .map_err(|e: std::io::Error| RunnerError::Asset(e.to_string()))?;
        Tensor::from_raw_buffer(&data, self.dtype, &[1], &self.device)
            .map_err(|e: candle_core::Error| RunnerError::Tensor(e.to_string()))
    }

    /// Get device info.
    pub fn device_info(&self) -> Result<String, RunnerError> {
        Ok(match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(ordinal) => format!("cuda:{ordinal:?}"),
            Device::Metal(_) => "metal".to_string(),
        })
    }

    /// Get dtype info.
    pub fn dtype_info(&self) -> Result<String, RunnerError> {
        Ok(match self.dtype {
            DType::F32 => "F32".to_string(),
            DType::F16 => "F16".to_string(),
            DType::I64 => "I64".to_string(),
            DType::I32 => "I32".to_string(),
            DType::U8 => "U8".to_string(),
            _ => "unknown".to_string(),
        })
    }

    /// Run scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V.
    ///
    /// `query` — [query_seq_len x (num_heads * head_dim)] f16
    /// `key_cache` — KV cache containing K tensor
    /// `value_cache` — KV cache containing V tensor
    /// `mask` — optional [query_seq_len x cache_seq_len] f32 mask
    /// `config` — attention configuration (num_heads, head_dim, max_seq, arch)
    ///
    /// Returns output tensor [query_seq_len x (num_heads * head_dim)] f32
    pub fn attention(
        &self,
        query: &crate::kernel::DeviceBuffer<f16>,
        key_cache: &crate::kernel::Kvcache,
        value_cache: &crate::kernel::Kvcache,
        mask: Option<&crate::kernel::DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<crate::kernel::DeviceBuffer<f32>, RunnerError> {
        self.attention
            .forward(query, key_cache, value_cache, mask, config)
            .map_err(|e| RunnerError::Tensor(format!("Attention failed: {e}")))
    }

    /// Get the attention kernel's target architecture.
    pub fn attention_arch(&self) -> AttentionArch {
        self.attention.arch()
    }

    /// Check if the attention kernel is available on this system.
    pub fn attention_available(&self) -> bool {
        self.attention.is_available()
    }
}
