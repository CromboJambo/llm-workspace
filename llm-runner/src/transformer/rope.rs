//! RoPE (Rotary Positional Embeddings).
//!
//! Applies rotary embeddings to query and key vectors for attention.
//! For each head, position m:
//!   q_m' = q_m * cos(m * theta) - q_{m+head_dim/2} * sin(m * theta)
//!   k_m' = k_m * cos(m * theta) - k_{m+head_dim/2} * sin(m * theta)

use half::f16;

/// Rotary positional embeddings configuration.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Dimension per head (must be even).
    pub head_dim: usize,
    /// Base for frequency computation.
    pub base: f32,
    /// Maximum context length for precomputing frequencies.
    pub max_position_embeddings: usize,
    /// Rope scaling factor (1.0 = no scaling).
    pub scaling_factor: Option<f32>,
    /// Rope scaling type ("linear", "yarn", etc.).
    pub scaling_type: Option<String>,
}

impl RopeConfig {
    pub fn new(head_dim: usize, base: f32, max_position_embeddings: usize) -> Self {
        Self {
            head_dim,
            base,
            max_position_embeddings,
            scaling_factor: None,
            scaling_type: None,
        }
    }

    /// Compute frequencies: theta_i = base^(-2i/dim) for i in 0..head_dim/2.
    fn compute_theta(&self) -> Vec<f32> {
        let dim_half = self.head_dim / 2;
        (0..dim_half)
            .map(|i| self.base.powf(-(i as f32) / dim_half as f32))
            .collect()
    }

    /// Apply RoPE to query and key vectors in-place.
    ///
    /// q: [batch, seq_len, num_heads, head_dim] flattened row-major
    /// k: [batch, seq_len, num_heads, head_dim] flattened row-major
    /// seq_len: sequence length
    /// start_pos: starting position in the context (for scaling)
    pub fn apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        num_heads: usize,
        seq_len: usize,
        start_pos: usize,
    ) {
        let theta = self.compute_theta();
        let dim_half = self.head_dim / 2;

        for pos in 0..seq_len {
            let actual_pos = start_pos + pos;
            for head in 0..num_heads {
                for (i, &freq) in theta.iter().enumerate() {
                    let angle = actual_pos as f32 * freq;
                    let cos = angle.cos();
                    let sin = angle.sin();

                    let q_idx = pos * num_heads * self.head_dim + head * self.head_dim + i;
                    let k_idx = pos * num_heads * self.head_dim + head * self.head_dim + i;

                    let q_next = q_idx + dim_half;
                    let k_next = k_idx + dim_half;

                    let q_orig = q[q_idx];
                    let k_orig = k[k_idx];

                    q[q_idx] = q_orig * cos - q[q_next] * sin;
                    q[q_next] = q_orig * sin + q[q_next] * cos;

                    k[k_idx] = k_orig * cos - k[k_next] * sin;
                    k[k_next] = k_orig * sin + k[k_next] * cos;
                }
            }
        }
    }
}

/// Apply RoPE to f16 query and key tensors.
pub fn apply_rope_f16(
    q: &mut [f16],
    k: &mut [f16],
    head_dim: usize,
    num_heads: usize,
    seq_len: usize,
    start_pos: usize,
    base: f32,
) {
    let dim_half = head_dim / 2;
    let theta: Vec<f32> = (0..dim_half)
        .map(|i| base.powf(-(i as f32) / dim_half as f32))
        .collect();

    for pos in 0..seq_len {
        let actual_pos = start_pos + pos;
        for head in 0..num_heads {
            for (i, &freq) in theta.iter().enumerate() {
                let angle = actual_pos as f32 * freq;
                let cos = angle.cos();
                let sin = angle.sin();

                let q_idx = pos * num_heads * head_dim + head * head_dim + i;
                let q_next = q_idx + dim_half;
                let k_idx = q_idx;
                let k_next = k_idx + dim_half;

                let q_orig = q[q_idx].to_f32();
                let q_next_val = q[q_next].to_f32();
                let k_orig = k[k_idx].to_f32();
                let k_next_val = k[k_next].to_f32();

                q[q_idx] = f16::from_f32(q_orig * cos - q_next_val * sin);
                q[q_next] = f16::from_f32(q_orig * sin + q_next_val * cos);

                k[k_idx] = f16::from_f32(k_orig * cos - k_next_val * sin);
                k[k_next] = f16::from_f32(k_orig * sin + k_next_val * cos);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_compute_theta() {
        let config = RopeConfig::new(8, 10000.0, 2048);
        let theta = config.compute_theta();
        assert_eq!(theta.len(), 4);
        // theta[0] = 10000^0 = 1.0
        assert!((theta[0] - 1.0).abs() < 1e-6);
        // theta values should be decreasing
        for i in 1..theta.len() {
            assert!(theta[i] < theta[i - 1]);
        }
    }

    #[test]
    fn rope_apply_identity_at_zero() {
        let config = RopeConfig::new(4, 10000.0, 64);
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![1.0, 2.0, 3.0, 4.0];

        config.apply(&mut q, &mut k, 1, 1, 0);

        // At position 0: cos(0)=1, sin(0)=0 → identity
        assert!((q[0] - 1.0).abs() < 1e-5);
        assert!((q[1] - 2.0).abs() < 1e-5);
        assert!((q[2] - 3.0).abs() < 1e-5);
        assert!((q[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn rope_apply_rotation() {
        let config = RopeConfig::new(4, 10000.0, 64);
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = vec![1.0, 0.0, 0.0, 1.0];

        config.apply(&mut q, &mut k, 1, 1, 1);

        // Position 1: angle = 1 * theta[i]
        // Should produce non-trivial rotation
        let norm_q: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_k: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Norm should be preserved
        assert!((norm_q - 1.414213).abs() < 1e-5);
        assert!((norm_k - 1.414213).abs() < 1e-5);
    }

    #[test]
    fn rope_apply_preserves_norm() {
        let config = RopeConfig::new(8, 10000.0, 2048);
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut k = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

        let orig_norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();

        // apply expects [batch * seq_len * num_heads * head_dim]
        // batch=1, seq_len=1, num_heads=1, head_dim=8 → 8 elements
        config.apply(&mut q, &mut k, 1, 1, 0);

        let new_norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((orig_norm - new_norm).abs() < 1e-4);
    }

    #[test]
    fn apply_rope_f16_identity_at_zero() {
        let mut q = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let mut k = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let orig_q = q.clone();

        apply_rope_f16(&mut q, &mut k, 4, 1, 1, 0, 10000.0);

        // At position 0: cos(0)=1, sin(0)=0 → identity
        for i in 0..4 {
            assert!((q[i].to_f32() - orig_q[i].to_f32()).abs() < 1e-5);
            assert!((k[i].to_f32() - orig_q[i].to_f32()).abs() < 1e-5);
        }
    }

    #[test]
    fn apply_rope_f16_preserves_norm() {
        let mut q = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let mut k = vec![f16::from_f32(0.5), f16::from_f32(1.5), f16::from_f32(2.5), f16::from_f32(3.5)];
        let orig_norm = q.iter().map(|x| x.to_f32() * x.to_f32()).sum::<f32>().sqrt();

        apply_rope_f16(&mut q, &mut k, 4, 1, 1, 1, 10000.0);

        let new_norm = q.iter().map(|x| x.to_f32() * x.to_f32()).sum::<f32>().sqrt();
        assert!((orig_norm - new_norm).abs() < 1e-3);
    }

    #[test]
    fn apply_rope_f16_multi_head() {
        let mut q = vec![
            f16::from_f32(1.0), f16::from_f32(0.0),
            f16::from_f32(0.0), f16::from_f32(1.0),
            f16::from_f32(1.0), f16::from_f32(0.0),
            f16::from_f32(0.0), f16::from_f32(1.0),
        ];
        let mut k = q.clone();
        apply_rope_f16(&mut q, &mut k, 4, 2, 1, 1, 10000.0);
        // Should produce non-trivial rotation (not all zeros)
        let norm: f32 = q.iter().map(|x| x.to_f32() * x.to_f32()).sum::<f32>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn rope_config_new_defaults() {
        let config = RopeConfig::new(64, 10000.0, 2048);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.base, 10000.0);
        assert_eq!(config.max_position_embeddings, 2048);
        assert!(config.scaling_factor.is_none());
        assert!(config.scaling_type.is_none());
    }
}
