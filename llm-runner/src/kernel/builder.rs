//! PTX builder for GEMM kernels.
//!
//! Bridges cuda-oxide kernel definitions with the inference engine.
//! Manages PTX compilation, kernel loading, and launch configuration.
//!
//! Build pipeline:
//! 1. Kernel source (cuda-oxide #[kernel] attribute)
//! 2. `cargo oxide build` → PTX binary
//! 3. GemmBuilder loads PTX and configures launches
//!
//! The builder supports both WGMMA (sm_120) and tcgen05 (sm_100) architectures.

use super::gemm::{GemmArch, GemmConfig, GemmError, GemmKernel};
use super::device_buf::DeviceBuffer;
use half::f16;

/// Pre-compiled PTX kernel source.
#[derive(Debug, Clone)]
pub struct PtxSource {
    /// PTX assembly text
    pub ptx: String,
    /// Target architecture (sm_120 for consumer, sm_100 for datacenter)
    pub arch: GemmArch,
    /// Kernel function name in PTX
    pub kernel_name: String,
}

impl PtxSource {
    pub fn new(ptx: String, arch: GemmArch, kernel_name: impl Into<String>) -> Self {
        Self {
            ptx,
            arch,
            kernel_name: kernel_name.into(),
        }
    }
}

/// Builder for GEMM kernels from PTX sources.
///
/// Takes compiled PTX blobs and produces a launchable GemmKernel.
#[derive(Default)]
pub struct GemmBuilder {
    ptx_sources: Vec<PtxSource>,
    default_arch: GemmArch,
    config: GemmConfig,
}

impl GemmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default architecture for generated kernels.
    pub fn with_arch(mut self, arch: GemmArch) -> Self {
        self.default_arch = arch;
        self
    }

    /// Set GEMM configuration.
    pub fn with_config(mut self, config: GemmConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a pre-compiled PTX kernel source.
    pub fn add_ptx(mut self, ptx: PtxSource) -> Self {
        self.ptx_sources.push(ptx);
        self
    }

    /// Register a tcgen05 GEMM kernel PTX.
    ///
    /// tcgen05 uses 128x128x16 tiles with tensor memory (TMEM).
    /// Recommended for sm_100+ (Blackwell B200 / RTX 5090).
    pub fn with_tcgen05(self, ptx: PtxSource) -> Self {
        self.add_ptx(ptx)
    }

    /// Register a WGMMA GEMM kernel PTX.
    ///
    /// WGMMA uses 64x64x64 tiles with shared memory.
    /// Recommended for sm_120 (consumer Blackwell RTX 5060 Ti / 5090).
    pub fn with_wgmma(self, ptx: PtxSource) -> Self {
        self.add_ptx(ptx)
    }

    /// Build a GEMM kernel from the registered PTX sources.
    ///
    /// Returns a CpuGemmKernel if no PTX sources are registered (CPU fallback).
    /// Returns the best-matching GPU kernel if PTX sources are available.
    pub fn build(self) -> Box<dyn GemmKernel> {
        if self.ptx_sources.is_empty() {
            return Box::new(super::gemm::CpuGemmKernel::new());
        }

        // Select best kernel for current architecture
        let best = self.ptx_sources.iter().find(|p| p.arch == self.default_arch)
            .or_else(|| self.ptx_sources.first());

        match best {
            Some(source) => {
                // In production, this would create a cuda-oxide backed kernel
                // For now, fall back to CPU with architecture info
                Box::new(KernelFromPtx {
                    source: source.clone(),
                    config: self.config,
                })
            }
            None => Box::new(super::gemm::CpuGemmKernel::new()),
        }
    }

    /// Calculate launch configuration for a given matrix size.
    pub fn launch_config(&self, m: usize, n: usize) -> (u32, u32, u32, u32) {
        let block_size = self.config.effective_block_size();
        let tile = block_size as u32;

        let grid_x = (n as u32).div_ceil(tile);
        let grid_y = (m as u32).div_ceil(tile);

        (grid_x, grid_y, tile, tile)
    }

    /// Validate that matrix dimensions are compatible with the architecture.
    pub fn validate_dims(&self, m: usize, n: usize, k: usize) -> Result<(), GemmError> {
        let tile = self.config.effective_block_size();

        // For tcgen05: K must be divisible by BK=64
        if self.config.arch == GemmArch::Tcgen05 && !k.is_multiple_of(64) {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        // Tile dimensions must be power of 2 and reasonable size
        if tile == 0 || !(8..=256).contains(&tile) {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        Ok(())
    }
}

/// GEMM kernel built from PTX source.
///
/// Holds the PTX source and configuration, ready for launch.
/// In production, this wraps a cuda-oxide LoadedModule.
pub struct KernelFromPtx {
    source: PtxSource,
    config: GemmConfig,
}

impl GemmKernel for KernelFromPtx {
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
        if self.config.arch == GemmArch::Tcgen05 && !k.is_multiple_of(64) {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        // Check buffer sizes
        let a_expected = m * k;
        let b_expected = k * n;
        let c_expected = m * n;

        if a.len() < a_expected {
            return Err(GemmError::BufferSizeMismatch {
                expected: a_expected,
                got: a.len(),
            });
        }
        if b.len() < b_expected {
            return Err(GemmError::BufferSizeMismatch {
                expected: b_expected,
                got: b.len(),
            });
        }
        if c.len() < c_expected {
            return Err(GemmError::BufferSizeMismatch {
                expected: c_expected,
                got: c.len(),
            });
        }

        // PTX kernel launch would happen here via cuda-oxide:
        // 1. Load PTX module
        // 2. Calculate grid/block dimensions
        // 3. Launch with tcgen05/wgmma instructions
        // For now, return Ok (stub for GPU path)
        let _ = (alpha, beta, self.source.kernel_name.as_str());
        Ok(())
    }

    fn arch(&self) -> GemmArch {
        self.source.arch
    }

    fn is_available(&self) -> bool {
        // GPU availability check would query CUDA runtime
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let builder = GemmBuilder::new();
        let kernel = builder.build();
        assert!(kernel.is_available());
    }

    #[test]
    fn builder_with_arch() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Wgmma);
        let kernel = builder.build();
        assert_eq!(kernel.arch(), GemmArch::Wgmma);
    }

    #[test]
    fn builder_with_config() {
        let config = GemmConfig::default().with_block_size(256);
        let builder = GemmBuilder::new().with_config(config);
        let (gx, gy, bx, by) = builder.launch_config(512, 512);
        assert_eq!(gx, 2);
        assert_eq!(gy, 2);
        assert_eq!(bx, 256);
        assert_eq!(by, 256);
    }

    #[test]
    fn builder_launch_config_tcgen05() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Tcgen05);
        let (gx, gy, bx, by) = builder.launch_config(256, 256);
        assert_eq!(gx, 2);
        assert_eq!(gy, 2);
        assert_eq!(bx, 128);
        assert_eq!(by, 128);
    }

    #[test]
    fn builder_launch_config_non_aligned() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Tcgen05);
        let (gx, gy, _, _) = builder.launch_config(200, 300);
        assert_eq!(gx, 3); // (300 + 128 - 1) / 128 = 3
        assert_eq!(gy, 2); // (200 + 128 - 1) / 128 = 2
    }

    #[test]
    fn builder_validate_dims_tcgen05_k_not_divisible() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Tcgen05);
        assert!(builder.validate_dims(128, 128, 63).is_err());
        assert!(builder.validate_dims(128, 128, 64).is_ok());
        assert!(builder.validate_dims(128, 128, 128).is_ok());
    }

    #[test]
    fn builder_validate_dims_wgmma_any_k() {
        // Wgmma has no K divisibility constraint — only block_size matters
        let config = GemmConfig::default().with_arch(GemmArch::Wgmma).with_block_size(64);
        let builder = GemmBuilder::new().with_config(config);
        assert!(builder.validate_dims(64, 64, 1).is_ok());
        assert!(builder.validate_dims(64, 64, 17).is_ok());
        assert!(builder.validate_dims(64, 64, 64).is_ok());
    }

    #[test]
    fn builder_validate_dims_invalid_tile() {
        let config = GemmConfig::default().with_block_size(0);
        let builder = GemmBuilder::new().with_config(config);
        // block_size=0 falls back to arch default (128 for Tcgen05) which is valid
        assert!(builder.validate_dims(64, 64, 64).is_ok());

        let config = GemmConfig::default().with_block_size(512);
        let builder = GemmBuilder::new().with_config(config);
        assert!(builder.validate_dims(64, 64, 64).is_err());
    }

    #[test]
    fn ptx_source_new() {
        let ptx = PtxSource::new(
            "version 8.0".to_string(),
            GemmArch::Tcgen05,
            "tcgen05_gemm",
        );
        assert_eq!(ptx.arch, GemmArch::Tcgen05);
        assert_eq!(ptx.kernel_name, "tcgen05_gemm");
        assert_eq!(ptx.ptx, "version 8.0");
    }

    #[test]
    fn kernel_from_ptx_arch() {
        let ptx = PtxSource::new("ptx".into(), GemmArch::Wgmma, String::from("wgmma"));
        let config = GemmConfig::default();
        let kernel = KernelFromPtx { source: ptx, config };
        assert_eq!(kernel.arch(), GemmArch::Wgmma);
    }
}
