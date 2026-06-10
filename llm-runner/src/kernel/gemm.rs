//! GEMM kernel interface and configuration.
//!
//! Provides the core matmul abstraction used by the LLM inference engine.
//! Supports two architectures:
//! - WGMMA (sm_120, consumer Blackwell: RTX 5060 Ti / 5090)
//! - tcgen05 (sm_100, datacenter Blackwell: B200)
//!
//! The differentiator: proving tcgen05 matmul works for LLM workloads
//! with non-matrix layouts (KV cache updates: M:1×K, K:N×K shapes).

use crate::kernel::device_buf::DeviceBuffer;
use half::f16;

/// Tensor core architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GemmArch {
    /// WGMMA — warp group matrix multiply (sm_120, consumer Blackwell)
    Wgmma,
    /// tcgen05 — tensor core with tensor memory (sm_100, datacenter Blackwell)
    #[default]
    Tcgen05,
}

impl GemmArch {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wgmma => "wgmma",
            Self::Tcgen05 => "tcgen05",
        }
    }

    pub fn supports_tma(&self) -> bool {
        // tcgen05 has native TMA support; WGMMA uses TMA for GMEM->SMEM copies
        true
    }

    pub fn tile_size(&self) -> usize {
        match self {
            Self::Wgmma => 64,    // 64x64x64 tiles
            Self::Tcgen05 => 128, // 128x128x16 tiles
        }
    }
}

/// Configuration for a GEMM kernel launch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GemmConfig {
    /// Target architecture
    pub arch: GemmArch,
    /// Whether to use TMA for async GMEM->SMEM copies
    pub use_tma: bool,
    /// Custom block size override (0 = use arch default)
    pub block_size: usize,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            arch: GemmArch::default(),
            use_tma: true,
            block_size: 0,
        }
    }
}

impl GemmConfig {
    pub fn effective_block_size(&self) -> usize {
        if self.block_size > 0 {
            self.block_size
        } else {
            self.arch.tile_size()
        }
    }

    pub fn with_arch(mut self, arch: GemmArch) -> Self {
        self.arch = arch;
        self
    }

    pub fn with_tma(mut self, use_tma: bool) -> Self {
        self.use_tma = use_tma;
        self
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
}

// --- GemmKernel Trait ---

/// Trait for GEMM (matrix multiply) kernels.
///
/// Both GPU (CUDA) and CPU implementations implement this trait.
pub trait GemmKernel: Send + Sync {
    /// Perform GEMM: C = alpha * A @ B + beta * C
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError>;

    /// Target tensor core architecture
    fn arch(&self) -> GemmArch;

    /// Whether this kernel is available on the current system
    fn is_available(&self) -> bool;
}

// --- GPU Implementation (Real cuda-oxide backed) ---

use std::sync::Arc;

/// CUDA implementation for GEMM kernel using cuda-oxide.
pub struct CudaGemmKernel {
    arch: GemmArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    module: Arc<cuda_core::CudaModule>,
    function: cuda_core::CudaFunction,
}

/// Builder for CudaGemmKernel that handles PTX loading and kernel resolution.
pub struct CudaGemmKernelBuilder {
    arch: GemmArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
}

impl CudaGemmKernelBuilder {
    pub fn new(
        arch: GemmArch,
        context: Arc<cuda_core::CudaContext>,
        stream: Arc<cuda_core::CudaStream>,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
        }
    }

    /// Build the kernel by loading PTX module and resolving function.
    pub fn build(self) -> Result<CudaGemmKernel, GemmError> {
        // Select PTX based on architecture
        let ptx_src = match self.arch {
            GemmArch::Wgmma => include_str!("ptx/gemm_wgmma.ptx"),
            GemmArch::Tcgen05 => include_str!("ptx/gemm_tcgen05.ptx"),
        };

        // Load module from PTX source
        let module = self
            .context
            .load_module_from_ptx_src(ptx_src)
            .map_err(|e| GemmError::Cuda(format!("module load failed: {}", e)))?;

        // Resolve kernel function
        let kernel_name = match self.arch {
            GemmArch::Wgmma => "gemm_wgmma_kernel",
            GemmArch::Tcgen05 => "gemm_tcgen05_kernel",
        };
        let function = module
            .load_function(kernel_name)
            .map_err(|e| GemmError::Cuda(format!("function load failed: {}", e)))?;

        Ok(CudaGemmKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module,
            function,
        })
    }
}

impl CudaGemmKernel {
    /// Get the cuda-oxide context for external operations
    pub fn context(&self) -> &Arc<cuda_core::CudaContext> {
        &self.context
    }

    /// Get the cuda-oxide stream
    pub fn stream(&self) -> &Arc<cuda_core::CudaStream> {
        &self.stream
    }
}

impl GemmKernel for CudaGemmKernel {
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError> {
        // Validate dimensions
        if m == 0 || n == 0 || k == 0 {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        // Validate input buffers have data on device
        let a_ptr = a.device_ptr().ok_or(GemmError::BufferSizeMismatch {
            expected: m * k,
            got: 0,
        })?;
        let b_ptr = b.device_ptr().ok_or(GemmError::BufferSizeMismatch {
            expected: k * n,
            got: 0,
        })?;
        let c_ptr = c.device_ptr().ok_or(GemmError::BufferSizeMismatch {
            expected: m * n,
            got: 0,
        })?;

        // Verify buffer sizes (DeviceBuffer::len() returns element count)
        if a.len() < m * k {
            return Err(GemmError::BufferSizeMismatch {
                expected: m * k,
                got: a.len(),
            });
        }
        if b.len() < k * n {
            return Err(GemmError::BufferSizeMismatch {
                expected: k * n,
                got: b.len(),
            });
        }
        if c.len() < m * n {
            return Err(GemmError::BufferSizeMismatch {
                expected: m * n,
                got: c.len(),
            });
        }

        // Prepare kernel arguments
        // Kernel signature: gemm_kernel(alpha, a, b, beta, c, m, n, k)
        let mut alpha_f32 = alpha;
        let mut beta_f32 = beta;
        let mut m_i32 = m as i32;
        let mut n_i32 = n as i32;
        let mut k_i32 = k as i32;

        let mut kernel_params: [*mut std::ffi::c_void; 8] = [
            &mut alpha_f32 as *mut f32 as *mut std::ffi::c_void,
            &a_ptr as *const u64 as *mut std::ffi::c_void,
            &b_ptr as *const u64 as *mut std::ffi::c_void,
            &mut beta_f32 as *mut f32 as *mut std::ffi::c_void,
            &c_ptr as *const u64 as *mut std::ffi::c_void,
            &mut m_i32 as *mut i32 as *mut std::ffi::c_void,
            &mut n_i32 as *mut i32 as *mut std::ffi::c_void,
            &mut k_i32 as *mut i32 as *mut std::ffi::c_void,
        ];

        // Compute grid/block dimensions
        // Using 2D grid: (N/128, M/128) with blockDim (16, 16) = 256 threads
        // For tcgen05 tile 128x128, block handles one tile
        // For wgmma tile 64x64, block handles 4 tiles (2x2)
        let (block_x, block_y) = match self.arch {
            GemmArch::Wgmma => (16, 16),  // 256 threads, 64x64 tile
            GemmArch::Tcgen05 => (16, 16), // 256 threads, 128x128 tile
        };

        let grid_x = (n + 127) / 128;
        let grid_y = (m + 63) / 64;

        // Bind context and launch kernel
        self.context.bind_to_thread().map_err(|e| {
            GemmError::Cuda(format!("context bind failed: {}", e))
        })?;

        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.function,
                (grid_x as u32, grid_y as u32, 1),
                (block_x as u32, block_y as u32, 1),
                0, // shared_mem_bytes
                &self.stream,
                &mut kernel_params,
            )
        }
        .map_err(|e| GemmError::LaunchFailed(e.to_string()))?;

        // Synchronize to ensure completion (async version available later)
        self.stream
            .synchronize()
            .map_err(|e| GemmError::LaunchFailed(format!("sync failed: {}", e)))?;

        Ok(())
    }

    fn arch(&self) -> GemmArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        // Check that kernel function is valid (not zeroed)
        unsafe { !self.function.cu_function().is_null() }
    }
}

/// GEMM error type.
#[derive(Debug, thiserror::Error)]
pub enum GemmError {
    #[error("invalid matrix dimensions: m={m}, n={n}, k={k}")]
    InvalidDimensions { m: usize, n: usize, k: usize },

    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("kernel not available on this device")]
    NotAvailable,

    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
}

/// CPU fallback GEMM implementation.
///
/// Used when no GPU kernel is available or for verification.
pub struct CpuGemmKernel;

impl CpuGemmKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuGemmKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmKernel for CpuGemmKernel {
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError> {
        let a_host = a.as_slice().ok_or_else(|| GemmError::BufferSizeMismatch {
            expected: m * k,
            got: 0,
        })?;
        if a_host.len() < m * k {
            return Err(GemmError::BufferSizeMismatch {
                expected: m * k,
                got: a_host.len(),
            });
        }

        let b_host = b.as_slice().ok_or_else(|| GemmError::BufferSizeMismatch {
            expected: k * n,
            got: 0,
        })?;
        if b_host.len() < k * n {
            return Err(GemmError::BufferSizeMismatch {
                expected: k * n,
                got: b_host.len(),
            });
        }

        let c_host = c
            .as_mut_slice()
            .ok_or_else(|| GemmError::BufferSizeMismatch {
                expected: m * n,
                got: 0,
            })?;
        if c_host.len() < m * n {
            return Err(GemmError::BufferSizeMismatch {
                expected: m * n,
                got: c_host.len(),
            });
        }

        // Naive GEMM: C[i][j] = alpha * sum_k(A[i][k] * B[k][j]) + beta * C[i][j]
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_host[i * k + kk].to_f32() * b_host[kk * n + j].to_f32();
                }
                c_host[i * n + j] = alpha * sum + beta * c_host[i * n + j];
            }
        }

        Ok(())
    }

    fn arch(&self) -> GemmArch {
        GemmArch::Wgmma // CPU doesn't target a specific arch
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_arch_name() {
        assert_eq!(GemmArch::Wgmma.name(), "wgmma");
        assert_eq!(GemmArch::Tcgen05.name(), "tcgen05");
    }

    #[test]
    fn gemm_arch_tile_size() {
        assert_eq!(GemmArch::Wgmma.tile_size(), 64);
        assert_eq!(GemmArch::Tcgen05.tile_size(), 128);
    }

    #[test]
    fn gemm_arch_supports_tma() {
        assert!(GemmArch::Wgmma.supports_tma());
        assert!(GemmArch::Tcgen05.supports_tma());
    }

    #[test]
    fn gemm_config_default() {
        let config = GemmConfig::default();
        assert_eq!(config.arch, GemmArch::Tcgen05);
        assert!(config.use_tma);
        assert_eq!(config.block_size, 0);
        assert_eq!(config.effective_block_size(), 128);
    }

    #[test]
    fn gemm_config_with_arch() {
        let config = GemmConfig::default().with_arch(GemmArch::Wgmma);
        assert_eq!(config.arch, GemmArch::Wgmma);
        assert_eq!(config.effective_block_size(), 64);
    }

    #[test]
    fn gemm_config_with_block_size() {
        let config = GemmConfig::default().with_block_size(256);
        assert_eq!(config.effective_block_size(), 256);
    }

    #[test]
    fn gemm_config_block_size_takes_precedence() {
        let config = GemmConfig::default()
            .with_arch(GemmArch::Wgmma)
            .with_block_size(128);
        assert_eq!(config.effective_block_size(), 128);
    }

    #[test]
    fn cpu_gemm_kernel_new() {
        let kernel = CpuGemmKernel::new();
        assert!(kernel.is_available());
    }

    #[test]
    fn cpu_gemm_kernel_matmul_basic() {
        let kernel = CpuGemmKernel::new();
        let a = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 4]); // [2x2]
        let b = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 4]); // [2x2]
        let mut c = DeviceBuffer::from_host(vec![0.0f32; 4]); // [2x2]

        let result = kernel.matmul(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        assert!(result.is_ok());

        let c_host = c.to_host();
        assert_eq!(c_host[0], 2.0); // 1*1 + 1*1
        assert_eq!(c_host[1], 2.0);
        assert_eq!(c_host[2], 2.0);
        assert_eq!(c_host[3], 2.0);
    }

    #[test]
    fn cpu_gemm_kernel_matmul_with_beta() {
        let kernel = CpuGemmKernel::new();
        let a = DeviceBuffer::from_host(vec![f16::from_f32(2.0); 1]); // [1x1]
        let b = DeviceBuffer::from_host(vec![f16::from_f32(3.0); 1]); // [1x1]
        let mut c = DeviceBuffer::from_host(vec![10.0f32; 1]); // [1x1]

        let result = kernel.matmul(1.0, &a, &b, 1.0, &mut c, 1, 1, 1);
        assert!(result.is_ok());

        // c = 1.0 * (2.0 * 3.0) + 1.0 * 10.0 = 6.0 + 10.0 = 16.0
        assert_eq!(c.to_host()[0], 16.0);
    }

    #[test]
    fn cpu_gemm_kernel_matmul_buffer_too_small() {
        let kernel = CpuGemmKernel::new();
        let a = DeviceBuffer::from_host(vec![f16::ZERO, f16::ZERO]); // [2x1] - correct for m=2,k=1
        let b = DeviceBuffer::from_host(vec![f16::ZERO]); // [1x1] - correct for k=1,n=1
        let mut c = DeviceBuffer::from_host(vec![0.0f32; 1]); // [1x1] - too small for m=2,n=1

        // C needs m*n = 2*1 = 2 elements, but only has 1
        let result = kernel.matmul(1.0, &a, &b, 0.0, &mut c, 2, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn gemm_error_display() {
        let err = GemmError::InvalidDimensions { m: 1, n: 2, k: 3 };
        let msg = err.to_string();
        assert!(msg.contains("1"));
        assert!(msg.contains("2"));
        assert!(msg.contains("3"));

        let err = GemmError::BufferSizeMismatch {
            expected: 10,
            got: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }
}
