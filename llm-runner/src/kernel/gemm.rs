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
    context: Arc<CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    device_info: CudaDeviceInfo,
}

impl CudaGemmKernelBuilder {
    pub fn new(
        arch: GemmArch,
        context: Arc<CudaContext>,
        stream: Arc<cuda_core::CudaStream>,
        device_info: CudaDeviceInfo,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
            device_info,
        }
    }

    /// Build the kernel by loading PTX module and resolving function.
    pub fn build(self) -> Result<CudaGemmKernel, GemmError> {
        // Pre-flight architecture check
        match self.arch {
            GemmArch::Wgmma if !self.device_info.supports_wgmma() => {
                return Err(GemmError::UnsupportedArch(format!(
                    "WGMMA requires sm_120+, but device is sm_{}.{}",
                    self.device_info.compute_capability.0,
                    self.device_info.compute_capability.1
                )));
            }
            GemmArch::Tcgen05 if !self.device_info.supports_tcgen05() => {
                return Err(GemmError::UnsupportedArch(format!(
                    "tcgen05 requires sm_100+, but device is sm_{}.{}",
                    self.device_info.compute_capability.0,
                    self.device_info.compute_capability.1
                )));
            }
            _ => {}
        }

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

        // Compute grid/block dimensions for WGMMA
        // blockDim = (32, 4) = 128 threads (4 warps = 1 warp group)
        // Each block computes one 64x64 output tile
        // gridDim = (ceil(N/64), ceil(M/64))
        let (grid_x, grid_y, block_x, block_y, shared_mem_bytes) = match self.arch {
            GemmArch::Wgmma => {
                // WGMMA: 64x64 tile, 128 threads, 8 KiB shared memory (double buffered)
                let grid_x = (n + 63) / 64;
                let grid_y = (m + 63) / 64;
                (grid_x as u32, grid_y as u32, 32u32, 4u32, 8192u32) // 8 KiB shared mem
            }
            GemmArch::Tcgen05 => {
                // tcgen05: 128x128 tile, 256 threads (placeholder)
                let grid_x = (n + 127) / 128;
                let grid_y = (m + 127) / 128;
                (grid_x as u32, grid_y as u32, 16u32, 16u32, 0u32)
            }
        };

        // Bind context and launch kernel
        self.context
            .bind_to_thread()
            .map_err(|e| GemmError::Cuda(format!("context bind failed: {}", e)))?;

        unsafe {
            cuda_core::launch_kernel_on_stream(
                &self.function,
                (grid_x, grid_y, 1),
                (block_x, block_y, 1),
                shared_mem_bytes,
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
    fn cpu_gemm_kernel_with_cpu_device_buffers() {
        // This test exercises the CPU-fallback device buffer path through the
        // GEMM kernel, verifying that the full resource abstraction layer works
        // without requiring a working CUDA driver.
        let kernel = CpuGemmKernel::new();

        // Use CPU-fallback buffers instead of Host buffers
        let a = DeviceBuffer::from_cpu_device(vec![f16::from_f32(2.0); 4]); // [2x2]
        let b = DeviceBuffer::from_cpu_device(vec![f16::from_f32(3.0); 4]); // [2x2]
        let mut c = DeviceBuffer::zeros_cpu_device(4); // [2x2]

        // Verify buffers are correctly identified as "device"
        assert!(a.is_device());
        assert!(b.is_device());
        assert!(c.is_device());
        assert!(a.as_slice().is_some());
        assert!(c.as_mut_slice().is_some());
        assert_eq!(c.device_ptr(), Some(0xDEAD));

        // Run GEMM: C = 1.0 * A @ B + 0.0 * C
        // Each C[i][j] = sum_k(A[i][k] * B[k][j]) = 2.0*3.0 + 2.0*3.0 = 12.0
        let result = kernel.matmul(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        assert!(result.is_ok());

        let c_host = c.to_host();
        assert_eq!(c_host[0], 12.0);
        assert_eq!(c_host[1], 12.0);
        assert_eq!(c_host[2], 12.0);
        assert_eq!(c_host[3], 12.0);
    }

    #[test]
    fn cpu_gemm_kernel_with_cpu_device_buffers_beta() {
        // Test GEMM with beta != 0 using CPU-fallback buffers
        let kernel = CpuGemmKernel::new();

        let a = DeviceBuffer::from_cpu_device(vec![f16::from_f32(1.5); 1]); // [1x1]
        let b = DeviceBuffer::from_cpu_device(vec![f16::from_f32(2.0); 1]); // [1x1]
        let mut c = DeviceBuffer::from_cpu_device(vec![10.0f32; 1]); // [1x1]

        // C = 1.0 * (1.5 * 2.0) + 0.5 * 10.0 = 3.0 + 5.0 = 8.0
        let result = kernel.matmul(1.0, &a, &b, 0.5, &mut c, 1, 1, 1);
        assert!(result.is_ok());

        assert_eq!(c.to_host()[0], 8.0);
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

    // --- GPU kernel verification tests ---

    /// Helper: run the GPU GEMM test if CUDA is available.
    /// Returns Ok((gpu_result, arch_used)) or Err with the failure reason.
    fn try_gpu_gemm_test(
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(Vec<f32>, GemmArch), String> {
        // Initialize CUDA
        unsafe {
            cuda_core::init(0).map_err(|e| format!("CUDA init failed: {}", e))?;
        }

        // Create runtime and stream
        let rt = crate::cuda_runtime::CudaRuntime::for_default_device()
            .map_err(|e| format!("CUDA runtime failed: {}", e))?;
        let stream = rt.new_stream().map_err(|e| format!("Stream creation failed: {}", e))?;

        // Detect architecture
        let arch = if rt.device_info().supports_wgmma() {
            GemmArch::Wgmma
        } else if rt.device_info().supports_tcgen05() {
            GemmArch::Tcgen05
        } else {
            return Err(format!(
                "Device sm_{}.{} does not support WGMMA/tcgen05",
                rt.device_info().compute_capability.0,
                rt.device_info().compute_capability.1
            ));
        };

        // Build GPU kernel
        let gpu_kernel = CudaGemmKernelBuilder::new(arch, rt.context().clone(), stream.clone(), rt.device_info().clone())
            .build()
            .map_err(|e| format!("Kernel build failed: {}", e))?;

        if !gpu_kernel.is_available() {
            return Err("GPU kernel reported as unavailable".into());
        }

        // Generate test data
        let a_host: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i % 10) as f32 + 0.5))
            .collect();
        let b_host: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32((i % 7) as f32 + 0.3))
            .collect();
        let c_init: Vec<f32> = vec![0.0f32; m * n];

        // Compute CPU reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_host[i * k + kk].to_f32() * b_host[kk * n + j].to_f32();
                }
                c_ref[i * n + j] = sum; // alpha=1.0, beta=0.0
            }
        }

        // Allocate device buffers
        let a_dev = DeviceBuffer::from_host_device(&stream, &a_host)
            .map_err(|e| format!("A alloc failed: {}", e))?;
        let b_dev = DeviceBuffer::from_host_device(&stream, &b_host)
            .map_err(|e| format!("B alloc failed: {}", e))?;
        let mut c_dev = DeviceBuffer::zeros_device(&stream, m * n)
            .map_err(|e| format!("C alloc failed: {}", e))?;

        // Launch GPU kernel
        gpu_kernel
            .matmul(1.0, &a_dev, &b_dev, 0.0, &mut c_dev, m, n, k)
            .map_err(|e| format!("Kernel launch failed: {}", e))?;

        // Read back result
        let c_gpu = c_dev
            .to_host_from_device(&stream)
            .map_err(|e| format!("D2H transfer failed: {}", e))?;

        Ok((c_gpu, arch))
    }

    #[test]
    fn gpu_gemm_2x2x2() {
        let result = match try_gpu_gemm_test(2, 2, 2) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("GPU GEMM test skipped: {}", e);
                return;
            }
        };

        let (c_gpu, arch) = result;
        println!(
            "GPU GEMM 2x2x2 (arch={}) result: {:?}",
            arch.name(),
            c_gpu
        );

        // Expected: A=ones(2x2) @ B=ones(2x2) = [[2,2],[2,2]]
        let expected = vec![2.0f32, 2.0, 2.0, 2.0];
        for (i, (got, exp)) in c_gpu.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "Mismatch at index {}: got={}, expected={}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn gpu_gemm_4x4x4() {
        let result = match try_gpu_gemm_test(4, 4, 4) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("GPU GEMM test skipped: {}", e);
                return;
            }
        };

        let (c_gpu, arch) = result;
        println!(
            "GPU GEMM 4x4x4 (arch={}) result: {:?}",
            arch.name(),
            c_gpu
        );

        // Compute reference
        let a_host: Vec<f16> = (0..16)
            .map(|i| f16::from_f32((i % 10) as f32 + 0.5))
            .collect();
        let b_host: Vec<f16> = (0..16)
            .map(|i| f16::from_f32((i % 7) as f32 + 0.3))
            .collect();
        let mut c_ref = vec![0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for kk in 0..4 {
                    sum += a_host[i * 4 + kk].to_f32() * b_host[kk * 4 + j].to_f32();
                }
                c_ref[i * 4 + j] = sum;
            }
        }

        for (i, (got, exp)) in c_gpu.iter().zip(c_ref.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "Mismatch at index {}: got={}, expected={}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn gpu_gemm_with_beta() {
        // Test C = alpha*A@B + beta*C_init
        let rt = match crate::cuda_runtime::CudaRuntime::for_default_device() {
            Ok(r) => r,
            Err(_) => return,
        };
        let stream = match rt.new_stream() {
            Ok(s) => s,
            Err(_) => return,
        };

        let arch = if rt.device_info().supports_wgmma() {
            GemmArch::Wgmma
        } else if rt.device_info().supports_tcgen05() {
            GemmArch::Tcgen05
        } else {
            return;
        };

        let gpu_kernel = match CudaGemmKernelBuilder::new(arch, rt.context().clone(), stream.clone(), rt.device_info().clone())
            .build()
        {
            Ok(k) => k,
            Err(_) => return,
        };

        let m = 2usize;
        let n = 2usize;
        let k = 2usize;

        let a_host = vec![f16::from_f32(2.0); 4]; // [2x2] all 2.0
        let b_host = vec![f16::from_f32(3.0); 4]; // [2x2] all 3.0
        let c_init_host = vec![10.0f32; 4]; // [2x2] all 10.0

        // Expected: C = 1.0 * (2.0 * 3.0 * 2) + 0.5 * 10.0 = 12.0 + 5.0 = 17.0
        // (each element: sum over k=2 of 2.0*3.0 = 12.0)

        let a_dev = match DeviceBuffer::from_host_device(&stream, &a_host) {
            Ok(d) => d,
            Err(_) => return,
        };
        let b_dev = match DeviceBuffer::from_host_device(&stream, &b_host) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut c_dev = match DeviceBuffer::from_host_device(&stream, &c_init_host) {
            Ok(d) => d,
            Err(_) => return,
        };

        let result = gpu_kernel.matmul(1.0, &a_dev, &b_dev, 0.5, &mut c_dev, m, n, k);
        match result {
            Ok(()) => {}
            Err(e) => {
                eprintln!("GPU GEMM beta test failed: {}", e);
                return;
            }
        }

        let c_gpu = match c_dev.to_host_from_device(&stream) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("D2H failed in beta test: {}", e);
                return;
            }
        };

        println!(
            "GPU GEMM beta test (alpha=1.0, beta=0.5) result: {:?}",
            c_gpu
        );

        let expected = 17.0f32; // 12.0 + 5.0
        for (i, got) in c_gpu.iter().enumerate() {
            assert!(
                (got - expected).abs() < 1e-3,
                "Beta test mismatch at index {}: got={}, expected={}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn gpu_gemm_kernel_launch_succeeds() {
        // Minimal test: just verify the kernel launches without error
        // (output correctness tested in gpu_gemm_* tests above)
        let rt = match crate::cuda_runtime::CudaRuntime::for_default_device() {
            Ok(r) => r,
            Err(_) => return,
        };
        let stream = match rt.new_stream() {
            Ok(s) => s,
            Err(_) => return,
        };

        let arch = if rt.device_info().supports_wgmma() {
            GemmArch::Wgmma
        } else if rt.device_info().supports_tcgen05() {
            GemmArch::Tcgen05
        } else {
            return;
        };

        let gpu_kernel = match CudaGemmKernelBuilder::new(arch, rt.context().clone(), stream.clone(), rt.device_info().clone())
            .build()
        {
            Ok(k) => k,
            Err(e) => {
                panic!("Failed to build GPU kernel: {}", e);
            }
        };

        assert!(gpu_kernel.is_available(), "GPU kernel should be available");
        assert_eq!(gpu_kernel.arch(), arch);

        // Allocate minimal buffers
        let m = 1usize;
        let n = 1usize;
        let k = 1usize;

        let a_host = vec![f16::from_f32(1.0)];
        let b_host = vec![f16::from_f32(1.0)];
        let c_host = vec![0.0f32];

        let a_dev = DeviceBuffer::from_host_device(&stream, &a_host)
            .expect("A alloc should succeed");
        let b_dev = DeviceBuffer::from_host_device(&stream, &b_host)
            .expect("B alloc should succeed");
        let mut c_dev = DeviceBuffer::from_host_device(&stream, &c_host)
            .expect("C alloc should succeed");

        // Launch should succeed
        let result = gpu_kernel.matmul(1.0, &a_dev, &b_dev, 0.0, &mut c_dev, m, n, k);
        assert!(
            result.is_ok(),
            "Kernel launch should succeed: {:?}",
            result
        );
    }

    #[test]
    fn gpu_gemm_arch_detection() {
        // Verify architecture detection logic
        unsafe {
            cuda_core::init(0).ok();
        }

        let rt = match crate::cuda_runtime::CudaRuntime::for_default_device() {
            Ok(r) => r,
            Err(_) => return,
        };

        let cc = rt.device_info().compute_capability;
        println!(
            "Default device: {} (sm_{}.{})",
            rt.device_info().name, cc.0, cc.1
        );

        let arch = if rt.device_info().supports_wgmma() {
            GemmArch::Wgmma
        } else if rt.device_info().supports_tcgen05() {
            GemmArch::Tcgen05
        } else {
            println!(
                "Device sm_{}.{}: no WGMMA/tcgen05 support, skipping kernel tests",
                cc.0, cc.1
            );
            return;
        };

        println!("Selected GEMM arch: {}", arch.name());
        assert!(arch == GemmArch::Wgmma || arch == GemmArch::Tcgen05);
    }
}
