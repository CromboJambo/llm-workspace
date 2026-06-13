use crate::cuda_runtime::{enumerate_devices, is_available, CudaRuntime};
use crate::error::RunnerError;
use crate::kernel::{
    AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel,
    CudaGemmKernelBuilder, GemmArch, GemmKernel,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use half::f16;
use std::sync::Arc;
/// Inference engine for tensor computation.
///
/// actual tensor computation layer. separate from PESTI host.
pub struct InferenceEngine {
    pub device: candle_core::Device,
    pub dtype: DType,
    gemm: Box<dyn GemmKernel + Send + Sync>,
    attention: Box<dyn AttentionKernel + Send + Sync>,
    /// CUDA runtime for device memory management (None = CPU-only mode).
    cuda_runtime: Option<Arc<CudaRuntime>>,
    /// CUDA stream for async operations.
    stream: Option<Arc<cuda_core::CudaStream>>,
}

impl InferenceEngine {
    pub fn new(device: Device, dtype: DType) -> Self {
        // Try to initialize CUDA if device preference is GPU
        let (cuda_runtime, stream) = if matches!(device, Device::Cuda(_)) || is_available() {
            match CudaRuntime::for_default_device() {
                Ok(rt) => {
                    let rt = Arc::new(rt);
                    match rt.new_stream() {
                        Ok(stream) => (Some(rt), Some(stream)),
                        Err(_) => (Some(rt), None),
                    }
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        // Initialize GEMM kernel
        let gemm: Box<dyn GemmKernel + Send + Sync> = if let (Some(cuda_rt), Some(s)) = (&cuda_runtime, &stream) {
            // Detect architecture from device compute capability
            let arch = if cuda_rt.device_info().supports_wgmma() {
                GemmArch::Wgmma
            } else if cuda_rt.device_info().supports_tcgen05() {
                GemmArch::Tcgen05
            } else {
                GemmArch::Wgmma // fallback
            };

            match CudaGemmKernelBuilder::new(arch, cuda_rt.context().clone(), s.clone(), cuda_rt.device_info().clone()).build() {
                Ok(kernel) => Box::new(kernel),
                Err(e) => {
                    eprintln!("Failed to initialize CUDA GEMM kernel: {}. Falling back to CPU.", e);
                    Box::new(crate::kernel::CpuGemmKernel::new())
                }
            }
        } else {
            Box::new(crate::kernel::CpuGemmKernel::new())
        };

        // Initialize attention kernel
        let attention: Box<dyn AttentionKernel + Send + Sync> = if is_available() {
            Box::new(crate::kernel::CudaAttentionKernel::new(AttentionArch::Wgmma))
        } else {
            Box::new(crate::kernel::CpuAttentionKernel::new())
        };

        Self {
            device,
            dtype,
            gemm,
            attention,
            cuda_runtime,
            stream,
        }
    }

    /// Create engine with a specific GEMM kernel.
    pub fn with_gemm(device: Device, dtype: DType, gemm: Box<dyn GemmKernel + Send + Sync>) -> Self {
        let attention = Box::new(CpuAttentionKernel::new());

        let (cuda_runtime, stream) = if is_available() {
            match CudaRuntime::for_default_device() {
                Ok(rt) => {
                    let rt = Arc::new(rt);
                    match rt.new_stream() {
                        Ok(stream) => (Some(rt), Some(stream)),
                        Err(_) => (Some(rt), None),
                    }
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        Self {
            device,
            dtype,
            gemm,
            attention,
            cuda_runtime,
            stream,
        }
    }

    /// Get the CUDA stream for device operations.
    fn get_stream(&self) -> Option<&Arc<cuda_core::CudaStream>> {
        self.stream.as_ref()
    }

    /// Check if GPU path is available.
    pub fn gpu_available(&self) -> bool {
        self.cuda_runtime.is_some() && self.gemm.is_available()
    }

    /// Get device info including CUDA details if available.
    pub fn full_device_info(&self) -> Result<String, RunnerError> {
        let base = self.device_info()?;
        if let Some(cuda) = &self.cuda_runtime {
            let info = cuda.device_info();
            Ok(format!(
                "{} | GPU: {} (sm_{}.{}) free={:.1}GiB/total={:.1}GiB",
                base,
                info.name,
                info.compute_capability.0,
                info.compute_capability.1,
                info.free_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            ))
        } else {
            Ok(base)
        }
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

    /// List available CUDA devices.
    pub fn list_devices() -> Result<Vec<crate::cuda_runtime::CudaDeviceInfo>, RunnerError> {
        enumerate_devices().map_err(|e| RunnerError::Device(e.to_string()))
    }
}
