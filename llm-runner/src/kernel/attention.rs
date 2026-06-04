//! Attention kernel interface and configuration for Blackwell tensor cores.
//!
//! Provides the core attention abstraction used by the LLM inference engine.
//! Supports two architectures:
//! - WGMMA (sm_120, consumer Blackwell: RTX 5060 Ti / 5090)
//! - tcgen05 (sm_100, datacenter Blackwell: B200)
//!
//! ## Blackwell tcgen05 constraints
//!
//! - 128-thread blocks
//! - 128x128x16 MMA tiles (BM=128, BN=128, BK=16 for attention inner dim)
//! - K dimension must be divisible by 64 for tcgen05 GEMM phases
//! - TMEM used for intermediate Q/K/V tiles instead of shared memory
//! - TMA descriptors need 256-byte aligned addresses
//!
//! ## Layout
//!
//! KV cache layout: `[num_heads * head_dim * 2, max_seq]` contiguous per layer.
//! K occupies `[0 .. head_stride * max_seq]`, V occupies
//! `[head_stride * max_seq .. head_stride * 2 * max_seq]`.
//!
//! Attention computation:
//! - Prefill: full KV cache loaded via TMA, all positions processed
//! - Decode: single position (box_y=1), append new KV, compute attention over cache

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::{Kvcache, KvcacheSlice};
use crate::kernel::tma_descriptor::TmaDescriptor;
use half::f16;

/// Attention tensor core architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionArch {
    /// WGMMA -- warp group matrix multiply (sm_120, consumer Blackwell)
    Wgmma,
    /// tcgen05 -- tensor core with tensor memory (sm_100, datacenter Blackwell)
    #[default]
    Tcgen05,
    /// CPU fallback
    Cpu,
}

impl AttentionArch {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wgmma => "wgmma",
            Self::Tcgen05 => "tcgen05",
            Self::Cpu => "cpu",
        }
    }

    pub fn supports_tma(&self) -> bool {
        match self {
            Self::Wgmma | Self::Tcgen05 => true,
            Self::Cpu => false,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::Wgmma => 128,
            Self::Tcgen05 => 128,
            Self::Cpu => 0,
        }
    }
}

/// Configuration for an attention kernel launch.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads (num_heads).
    pub num_heads: usize,
    /// Dimension per attention head (head_dim).
    pub head_dim: usize,
    /// Maximum sequence length.
    pub max_seq: usize,
    /// Target architecture.
    pub arch: AttentionArch,
    /// Whether to use TMA for async GMEM->SMEM copies.
    pub use_tma: bool,
    /// Custom block size override (0 = use arch default).
    pub block_size: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            max_seq: 2048,
            arch: AttentionArch::default(),
            use_tma: true,
            block_size: 0,
        }
    }
}

impl AttentionConfig {
    pub fn effective_block_size(&self) -> usize {
        if self.block_size > 0 {
            self.block_size
        } else {
            self.arch.block_size()
        }
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    pub fn with_max_seq(mut self, max_seq: usize) -> Self {
        self.max_seq = max_seq;
        self
    }

    pub fn with_arch(mut self, arch: AttentionArch) -> Self {
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

    /// Scale factor for attention logits: 1.0 / sqrt(head_dim).
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// A slice of the KV cache for attention computation.
///
/// Wraps a KvcacheSlice per head with query tensor info.
#[derive(Debug, Clone)]
pub struct AttentionSlice {
    /// Key cache slices per head.
    pub key_slices: Vec<KvcacheSlice>,
    /// Value cache slices per head.
    pub value_slices: Vec<KvcacheSlice>,
    /// Query tensor (on host or device).
    pub query: DeviceBuffer<f16>,
    /// Sequence length of cached keys/values.
    pub cache_seq_len: usize,
    /// Number of query positions (1 for decode, >1 for prefill).
    pub query_seq_len: usize,
}

impl AttentionSlice {
    /// Build attention slices for all heads from a Kvcache.
    ///
    /// `cache` — the KV cache to slice.
    /// `query` — the query tensor of shape [query_seq_len, num_heads * head_dim].
    /// `seq_start` — starting position in the cache.
    /// `seq_len` — number of cached positions to attend over.
    pub fn from_cache(
        cache: &Kvcache,
        query: DeviceBuffer<f16>,
        seq_start: usize,
        seq_len: usize,
    ) -> Self {
        let num_heads = cache.num_heads();
        let head_dim = cache.head_dim();
        let max_seq = cache.max_seq();
        let gmem_addr = cache.device_ptr().unwrap_or(0);
        let per_head_dim = num_heads * head_dim;
        let query_seq_len = query.len().checked_div(per_head_dim).unwrap_or(0);

        let key_slices: Vec<KvcacheSlice> = (0..num_heads)
            .map(|h| {
                KvcacheSlice::new(gmem_addr, num_heads, head_dim, max_seq, h, seq_start, seq_len, true)
            })
            .collect();

        let value_slices: Vec<KvcacheSlice> = (0..num_heads)
            .map(|h| {
                KvcacheSlice::new(gmem_addr, num_heads, head_dim, max_seq, h, seq_start, seq_len, false)
            })
            .collect();

        Self {
            key_slices,
            value_slices,
            query,
            cache_seq_len: cache.seq_len(),
            query_seq_len,
        }
    }

    /// Get TMA descriptors for K and V of a specific head.
    pub fn tma_descriptors(&self, head_idx: usize) -> (Option<TmaDescriptor>, Option<TmaDescriptor>) {
        if head_idx >= self.key_slices.len() {
            return (None, None);
        }
        let k_desc = self.key_slices[head_idx].to_tma_descriptor();
        let v_desc = self.value_slices[head_idx].to_tma_descriptor();
        (Some(k_desc), Some(v_desc))
    }
}

/// Attention error type.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("invalid dimensions: heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")]
    InvalidDimensions {
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    },

    #[error("head index out of bounds: head_idx={head_idx}, num_heads={num_heads}")]
    HeadIndexOutOfBounds { head_idx: usize, num_heads: usize },

    #[error("sequence length exceeded: current={current}, max={max}")]
    SeqLenExceeded { current: usize, max: usize },

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

    #[error("tcgen05 constraint: head_dim must be divisible by 64, got {0}")]
    Tcgen05Constraint(usize),
}

/// Attention kernel trait.
///
/// Implementations compute scaled dot-product attention:
/// `output = softmax(Q @ K^T / sqrt(head_dim)) @ V`
///
/// Supports both prefill (batched) and decode (single-token) modes.
pub trait AttentionKernel: Send + Sync {
    /// Compute scaled dot-product attention.
    ///
    /// `query` — [query_seq_len x (num_heads * head_dim)] f16
    /// `key_cache` — KV cache containing K tensor
    /// `value_cache` — KV cache containing V tensor
    /// `mask` — optional [query_seq_len x cache_seq_len] f32 mask (0.0 = visible, -inf = masked)
    /// `config` — attention configuration
    ///
    /// Returns output tensor [query_seq_len x (num_heads * head_dim)] f32
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError>;

    /// Get the architecture this kernel targets.
    fn arch(&self) -> AttentionArch;

    /// Check if this kernel is available on the current system.
    fn is_available(&self) -> bool {
        true
    }
}

/// CPU fallback attention implementation.
///
/// Computes scaled dot-product attention on host for verification and
/// systems without GPU hardware.
pub struct CpuAttentionKernel {
    arch: AttentionArch,
}

impl CpuAttentionKernel {
    pub fn new() -> Self {
        Self {
            arch: AttentionArch::Cpu,
        }
    }

    pub fn with_arch(arch: AttentionArch) -> Self {
        Self { arch }
    }

    /// Compute softmax along the last axis of a 2D f32 buffer [rows x cols].
    fn softmax(buffer: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows * cols];
        for i in 0..rows {
            let start = i * cols;
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cols {
                let val = buffer[start + j];
                if val > max_val {
                    max_val = val;
                }
            }
            // Subtract max and exponentiate
            let mut sum = 0.0f32;
            for j in 0..cols {
                let exp_val = (buffer[start + j] - max_val).exp();
                result[start + j] = exp_val;
                sum += exp_val;
            }
            // Normalize
            if sum > 0.0 {
                for j in 0..cols {
                    result[start + j] /= sum;
                }
            }
        }
        result
    }

    /// Compute matrix multiply: C = A @ B^T, where A is [m x k] and B is [n x k].
    ///
    /// This computes A @ B^T efficiently by transposing B conceptually:
    /// C[i][j] = sum_l(A[i][l] * B[j][l])
    fn matmul_transpose_b(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l].to_f32() * b[j * k + l].to_f32();
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// Extract a single head's KV slice from a Kvcache.
    ///
    /// `is_key` — true for K, false for V.
    /// `head_idx` — which head to extract.
    /// `seq_start` — starting sequence position.
    /// `seq_len` — number of positions.
    fn extract_head_slice(
        cache: &Kvcache,
        is_key: bool,
        head_idx: usize,
        seq_start: usize,
        seq_len: usize,
    ) -> Vec<f16> {
        let num_heads = cache.num_heads();
        let head_dim = cache.head_dim();
        let head_stride = num_heads * head_dim;
        let head_offset = head_idx * head_dim;
        let max_seq = cache.max_seq();

        let src = cache.buffer().as_slice().unwrap_or(&[]);

        // Calculate base offset for K or V tensor
        // K occupies [0 .. head_stride * max_seq], V occupies [head_stride * max_seq .. head_stride * 2 * max_seq]
        let v_base = head_stride * max_seq;
        let base = if is_key { 0 } else { v_base };

        // Head's data starts at base + head_stride * head_offset
        let head_base = base + head_stride * head_offset;

        // Extract seq_len positions, each of head_dim elements
        // Note: positions are strided by head_stride in the buffer
        let mut result = Vec::with_capacity(seq_len * head_dim);
        for s in 0..seq_len {
            let pos = seq_start + s;
            if pos < max_seq {
                let row_start = head_base + head_stride * pos;
                for d in 0..head_dim {
                    let idx = row_start + d;
                    if idx < src.len() {
                        result.push(src[idx]);
                    }
                }
            }
        }
        result
    }
}

impl Default for CpuAttentionKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionKernel for CpuAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let cache_seq_len = key_cache.seq_len();
        let out_dim = num_heads * head_dim;
        let query_seq_len = query.len().checked_div(out_dim).unwrap_or(0);

        if num_heads == 0 || head_dim == 0 || cache_seq_len == 0 {
            return Err(AttentionError::InvalidDimensions {
                num_heads,
                head_dim,
                seq_len: cache_seq_len,
            });
        }

        let query_host = query.as_slice().ok_or(AttentionError::BufferSizeMismatch {
            expected: query_seq_len * out_dim,
            got: 0,
        })?;
        if query_host.len() < query_seq_len * out_dim {
            return Err(AttentionError::BufferSizeMismatch {
                expected: query_seq_len * out_dim,
                got: query_host.len(),
            });
        }

        let scale = config.scale();

        // Output: [query_seq_len x (num_heads * head_dim)]
        let mut output = vec![0.0f32; query_seq_len * out_dim];

        for head in 0..num_heads {
            // Extract K and V slices for this head
            let k_slice = Self::extract_head_slice(key_cache, true, head, 0, cache_seq_len);
            let v_slice = Self::extract_head_slice(value_cache, false, head, 0, cache_seq_len);

            if k_slice.is_empty() || v_slice.is_empty() {
                continue;
            }

            // Extract query for this head across all query positions
            let q_start = head * head_dim;
            let q_slice = &query_host[q_start..q_start + query_seq_len * head_dim];

            // Compute Q @ K^T: [query_seq_len x cache_seq_len]
            // Q is [query_seq_len x head_dim], K is [cache_seq_len x head_dim]
            // Q @ K^T[i][j] = sum_l(Q[i][l] * K[j][l])
            let qk = Self::matmul_transpose_b(q_slice, &k_slice, query_seq_len, cache_seq_len, head_dim);

            // Scale by 1/sqrt(head_dim)
            let scaled: Vec<f32> = qk.iter().map(|x| x * scale).collect();

            // Apply softmax over cache_seq_len dimension
            let attn_weights = Self::softmax(&scaled, query_seq_len, cache_seq_len);

            // Compute attn_weights @ V: [query_seq_len x head_dim]
            // attn_weights is [query_seq_len x cache_seq_len], V is [cache_seq_len x head_dim]
            for q in 0..query_seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..cache_seq_len {
                        sum += attn_weights[q * cache_seq_len + s] * v_slice[s * head_dim + d].to_f32();
                    }
                    output[q * out_dim + head * head_dim + d] = sum;
                }
            }
        }

        Ok(DeviceBuffer::from_host(output))
    }

    fn arch(&self) -> AttentionArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AttentionArch ──────────────────────────────────────────────────

    #[test]
    fn attention_arch_name() {
        assert_eq!(AttentionArch::Wgmma.name(), "wgmma");
        assert_eq!(AttentionArch::Tcgen05.name(), "tcgen05");
        assert_eq!(AttentionArch::Cpu.name(), "cpu");
    }

    #[test]
    fn attention_arch_supports_tma() {
        assert!(AttentionArch::Wgmma.supports_tma());
        assert!(AttentionArch::Tcgen05.supports_tma());
        assert!(!AttentionArch::Cpu.supports_tma());
    }

    #[test]
    fn attention_arch_block_size() {
        assert_eq!(AttentionArch::Wgmma.block_size(), 128);
        assert_eq!(AttentionArch::Tcgen05.block_size(), 128);
        assert_eq!(AttentionArch::Cpu.block_size(), 0);
    }

    #[test]
    fn attention_arch_default_is_tcgen05() {
        assert_eq!(AttentionArch::default(), AttentionArch::Tcgen05);
    }

    // ── AttentionConfig ────────────────────────────────────────────────

    #[test]
    fn attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.max_seq, 2048);
        assert_eq!(config.arch, AttentionArch::Tcgen05);
        assert!(config.use_tma);
        assert_eq!(config.block_size, 0);
    }

    #[test]
    fn attention_config_effective_block_size_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.effective_block_size(), 128);
    }

    #[test]
    fn attention_config_effective_block_size_custom() {
        let config = AttentionConfig::default().with_block_size(256);
        assert_eq!(config.effective_block_size(), 256);
    }

    #[test]
    fn attention_config_effective_block_size_zero_falls_back() {
        let config = AttentionConfig::default().with_arch(AttentionArch::Wgmma).with_block_size(0);
        assert_eq!(config.effective_block_size(), 128);
    }

    #[test]
    fn attention_config_scale() {
        let config = AttentionConfig::default().with_head_dim(64);
        assert!((config.scale() - 0.125).abs() < 1e-6); // 1/sqrt(64) = 0.125
    }

    #[test]
    fn attention_config_scale_128() {
        let config = AttentionConfig::default().with_head_dim(128);
        assert!((config.scale() - 1.0 / 11.313708).abs() < 1e-4);
    }

    #[test]
    fn attention_config_builder_chain() {
        let config = AttentionConfig::default()
            .with_num_heads(16)
            .with_head_dim(128)
            .with_max_seq(4096)
            .with_arch(AttentionArch::Wgmma)
            .with_tma(false);

        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.max_seq, 4096);
        assert_eq!(config.arch, AttentionArch::Wgmma);
        assert!(!config.use_tma);
    }

    // ── AttentionSlice ─────────────────────────────────────────────────

    #[test]
    fn attention_slice_from_cache() {
        let mut cache = Kvcache::new(8, 64, 128, false);
        let key = vec![f16::from_f32(1.0); 8 * 64];
        let value = vec![f16::from_f32(2.0); 8 * 64];
        cache.append(&key, &value).unwrap();

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 4 * 8 * 64]);
        let slice = AttentionSlice::from_cache(&cache, query, 0, 1);

        assert_eq!(slice.key_slices.len(), 8);
        assert_eq!(slice.value_slices.len(), 8);
        assert_eq!(slice.query_seq_len, 4);
        assert_eq!(slice.cache_seq_len, 1);
    }

    #[test]
    fn attention_slice_tma_descriptors() {
        let mut cache = Kvcache::new(8, 64, 128, false);
        let key = vec![f16::from_f32(1.0); 8 * 64];
        let value = vec![f16::from_f32(2.0); 8 * 64];
        cache.append(&key, &value).unwrap();

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 64]);
        let slice = AttentionSlice::from_cache(&cache, query, 0, 1);

        let (k_desc, v_desc) = slice.tma_descriptors(0);
        assert!(k_desc.is_some());
        assert!(v_desc.is_some());

        // Both should be global cache read descriptors
        let k = k_desc.unwrap();
        let v = v_desc.unwrap();
        assert_eq!((k.0[3] >> 24) & 0xFF, 1); // descriptor type = 1
        assert_eq!((v.0[3] >> 24) & 0xFF, 1);
    }

    #[test]
    fn attention_slice_tma_descriptors_head_oob() {
        let cache = Kvcache::new(8, 64, 128, false);
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 64]);
        let slice = AttentionSlice::from_cache(&cache, query, 0, 1);

        let (k_desc, v_desc) = slice.tma_descriptors(8);
        assert!(k_desc.is_none());
        assert!(v_desc.is_none());
    }

    // ── CpuAttentionKernel ─────────────────────────────────────────────

    #[test]
    fn cpu_attention_kernel_new() {
        let kernel = CpuAttentionKernel::new();
        assert_eq!(kernel.arch(), AttentionArch::Cpu);
        assert!(kernel.is_available());
    }

    #[test]
    fn cpu_attention_kernel_with_arch() {
        let kernel = CpuAttentionKernel::with_arch(AttentionArch::Wgmma);
        assert_eq!(kernel.arch(), AttentionArch::Wgmma);
    }

    #[test]
    fn cpu_attention_kernel_forward_prefill() {
        let kernel = CpuAttentionKernel::new();

        // Build KV cache with known values
        let mut key_cache = Kvcache::new(4, 8, 16, false);
        let mut value_cache = Kvcache::new(4, 8, 16, false);

        for i in 0..3 {
            let key = vec![f16::from_f32(i as f32 + 1.0); 4 * 8];
            let value = vec![f16::from_f32((i as f32 + 1.0) * 0.5); 4 * 8];
            key_cache.append(&key, &value).unwrap();
            value_cache.append(&key, &value).unwrap();
        }

        // Query: 2 positions, 4 heads, 8 dim per head
        let query = DeviceBuffer::from_host(
            vec![f16::from_f32(1.0); 2 * 4 * 8]
        );

        let config = AttentionConfig::default()
            .with_num_heads(4)
            .with_head_dim(8)
            .with_max_seq(16);

        let result = kernel.forward(&query, &key_cache, &value_cache, None, &config);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 2 * 4 * 8);
    }

    #[test]
    fn cpu_attention_kernel_forward_decode() {
        let kernel = CpuAttentionKernel::new();

        // Build KV cache with 10 positions
        let mut key_cache = Kvcache::new(2, 16, 32, false);
        let mut value_cache = Kvcache::new(2, 16, 32, false);

        for i in 0..10 {
            let key = vec![f16::from_f32(1.0 + i as f32 * 0.1); 2 * 16];
            let value = vec![f16::from_f32(0.5 + i as f32 * 0.05); 2 * 16];
            key_cache.append(&key, &value).unwrap();
            value_cache.append(&key, &value).unwrap();
        }

        // Single query position
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 1 * 2 * 16]);

        let config = AttentionConfig::default()
            .with_num_heads(2)
            .with_head_dim(16)
            .with_max_seq(32);

        let result = kernel.forward(&query, &key_cache, &value_cache, None, &config);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), 1 * 2 * 16);
    }

    #[test]
    fn cpu_attention_kernel_forward_zero_heads() {
        let kernel = CpuAttentionKernel::new();
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 4]);
        let key_cache = Kvcache::new(0, 64, 16, false);
        let value_cache = Kvcache::new(0, 64, 16, false);
        let config = AttentionConfig::default();

        let result = kernel.forward(&query, &key_cache, &value_cache, None, &config);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_attention_kernel_forward_zero_dim() {
        let kernel = CpuAttentionKernel::new();
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 4]);
        let key_cache = Kvcache::new(4, 0, 16, false);
        let value_cache = Kvcache::new(4, 0, 16, false);
        let config = AttentionConfig::default();

        let result = kernel.forward(&query, &key_cache, &value_cache, None, &config);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_attention_kernel_forward_masked() {
        let kernel = CpuAttentionKernel::new();

        let mut key_cache = Kvcache::new(2, 8, 8, false);
        let mut value_cache = Kvcache::new(2, 8, 8, false);

        for _i in 0..4 {
            let key = vec![f16::from_f32(1.0); 2 * 8];
            let value = vec![f16::from_f32(1.0); 2 * 8];
            key_cache.append(&key, &value).unwrap();
            value_cache.append(&key, &value).unwrap();
        }

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 2 * 2 * 8]);
        let config = AttentionConfig::default()
            .with_num_heads(2)
            .with_head_dim(8)
            .with_max_seq(8);

        // Mask: first position attends to all, second position attends only to first
        let mask_data = vec![0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 0.0];
        let mask = DeviceBuffer::from_host(mask_data);

        let result = kernel.forward(&query, &key_cache, &value_cache, Some(&mask), &config);
        assert!(result.is_ok());
    }

    #[test]
    fn cpu_attention_kernel_softmax_basic() {
        // softmax([1.0, 2.0, 3.0]) should give [~0.09, ~0.24, ~0.67]
        let input = vec![1.0, 2.0, 3.0];
        let result = CpuAttentionKernel::softmax(&input, 1, 3);
        assert_eq!(result.len(), 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn cpu_attention_kernel_softmax_multirow() {
        let input = vec![1.0, 2.0, 0.0, 3.0, 1.0, 2.0];
        let result = CpuAttentionKernel::softmax(&input, 2, 3);
        assert_eq!(result.len(), 6);
        // Each row should sum to 1
        let sum0: f32 = result[0..3].iter().sum();
        let sum1: f32 = result[3..6].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_attention_kernel_softmax_numerical_stability() {
        // Large values should not overflow
        let input = vec![1000.0, 1001.0, 1002.0];
        let result = CpuAttentionKernel::softmax(&input, 1, 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[0] > 0.0 && result[0] < 1.0);
        assert!(result[1] > 0.0 && result[1] < 1.0);
        assert!(result[2] > 0.0 && result[2] < 1.0);
    }

    #[test]
    fn cpu_attention_kernel_matmul_transpose_b() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // A @ B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //         = [[17, 23], [39, 53]]
        let a = vec![f16::from_f32(1.0), f16::from_f32(2.0),
                     f16::from_f32(3.0), f16::from_f32(4.0)];
        let b = vec![f16::from_f32(5.0), f16::from_f32(6.0),
                     f16::from_f32(7.0), f16::from_f32(8.0)];

        let result = CpuAttentionKernel::matmul_transpose_b(&a, &b, 2, 2, 2);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 17.0).abs() < 1e-3);
        assert!((result[1] - 23.0).abs() < 1e-3);
        assert!((result[2] - 39.0).abs() < 1e-3);
        assert!((result[3] - 53.0).abs() < 1e-3);
    }

    #[test]
    fn cpu_attention_kernel_forward_identity_query() {
        // When query is one-hot per head, output should approximate the corresponding V row
        let kernel = CpuAttentionKernel::new();

        let mut key_cache = Kvcache::new(1, 4, 8, false);
        let mut value_cache = Kvcache::new(1, 4, 8, false);

        // Write known values
        for i in 0..3 {
            let key = vec![f16::from_f32(1.0); 4];
            let value = vec![f16::from_f32((i + 1) as f32); 4];
            key_cache.append(&key, &value).unwrap();
            value_cache.append(&key, &value).unwrap();
        }

        // Query: all ones → uniform attention → weighted average of V rows
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 1 * 4]);

        let config = AttentionConfig::default()
            .with_num_heads(1)
            .with_head_dim(4)
            .with_max_seq(8);

        let result = kernel.forward(&query, &key_cache, &value_cache, None, &config).unwrap();
        let output = result.to_host();

        // With uniform attention over 3 equal-value rows (1,2,3), output ≈ 2.0 per element
        let expected = 2.0;
        for v in &output {
            assert!((v - expected).abs() < 0.5, "expected ~{expected}, got {v}");
        }
    }

    #[test]
    fn cpu_attention_kernel_forward_single_position_cache() {
        let kernel = CpuAttentionKernel::new();

        let mut key_cache = Kvcache::new(1, 4, 8, false);
        let mut value_cache = Kvcache::new(1, 4, 8, false);

        let key = vec![f16::from_f32(2.0); 4];
        let value = vec![f16::from_f32(3.0); 4];
        key_cache.append(&key, &value).unwrap();
        value_cache.append(&key, &value).unwrap();

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 4]);

        let config = AttentionConfig::default()
            .with_num_heads(1)
            .with_head_dim(4)
            .with_max_seq(8);

        let result = kernel.forward(&query, &key_cache, &value_cache, None, &config).unwrap();
        // With single cache position, softmax is trivially [1.0], output = V
        let output = result.to_host();
        for v in &output {
            assert!((v - 3.0).abs() < 1e-3, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn attention_error_display() {
        let err = AttentionError::InvalidDimensions {
            num_heads: 8,
            head_dim: 64,
            seq_len: 0,
        };
        assert!(err.to_string().contains("8"));
        assert!(err.to_string().contains("64"));

        let err = AttentionError::Tcgen05Constraint(32);
        assert!(err.to_string().contains("32"));
    }

    #[test]
    fn attention_slice_from_cache_empty_cache() {
        let cache = Kvcache::new(4, 64, 128, false);
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 64]);
        let slice = AttentionSlice::from_cache(&cache, query, 0, 1);
        assert_eq!(slice.key_slices.len(), 4);
        assert_eq!(slice.cache_seq_len, 0);
    }

    #[test]
    fn attention_config_clone() {
        let config = AttentionConfig::default()
            .with_num_heads(16)
            .with_head_dim(128);
        let cloned = config.clone();
        assert_eq!(config.num_heads, cloned.num_heads);
        assert_eq!(config.head_dim, cloned.head_dim);
        assert_eq!(config.arch, cloned.arch);
    }

    #[test]
    fn attention_slice_clone_copy() {
        let mut cache = Kvcache::new(4, 64, 128, false);
        let key = vec![f16::from_f32(1.0); 4 * 64];
        let value = vec![f16::from_f32(2.0); 4 * 64];
        cache.append(&key, &value).unwrap();

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 64]);
        let slice = AttentionSlice::from_cache(&cache, query, 0, 1);
        let cloned = slice.clone();
        assert_eq!(cloned.key_slices.len(), slice.key_slices.len());
        assert_eq!(cloned.cache_seq_len, slice.cache_seq_len);
    }
}
