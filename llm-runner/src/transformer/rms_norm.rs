//! RMSNorm: root mean square layer normalization.
//!
//! RMSNorm(x) = x * gain / sqrt(e^2 + variance), where variance = mean(x^2).
//! Used in Llama-style models instead of LayerNorm.

/// RMS normalization layer.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
    pub dim: usize,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        let dim = weight.len();
        Self { weight, eps, dim }
    }

    /// Forward pass: normalize each row of input.
    ///
    /// x: [batch_size, dim]
    /// Returns: [batch_size, dim]
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * self.dim];

        for b in 0..batch_size {
            let start = b * self.dim;
            let mut variance = 0.0f32;

            for i in 0..self.dim {
                variance += x[start + i] * x[start + i];
            }
            variance /= self.dim as f32;
            let rms = (variance + self.eps).sqrt();

            for i in 0..self.dim {
                output[start + i] = x[start + i] / rms * self.weight[i];
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_forward_simple() {
        let weight = vec![1.0, 1.0];
        let eps = 1e-5;
        let norm = RmsNorm::new(weight, eps);

        // x = [3.0, 4.0] → rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let x = vec![3.0, 4.0];
        let output = norm.forward(&x, 1);

        let rms = ((3.0f32.powi(2) + 4.0f32.powi(2)) / 2.0 + eps).sqrt();
        assert!((output[0] - 3.0 / rms).abs() < 1e-5);
        assert!((output[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rmsnorm_batch() {
        let weight = vec![1.0, 1.0];
        let norm = RmsNorm::new(weight, 1e-5);

        let x = vec![1.0, 0.0, 0.0, 1.0]; // batch=2
        let output = norm.forward(&x, 2);

        // Row 0: [1, 0] → rms = sqrt(0.5) ≈ 0.7071
        let rms0 = ((1.0f32 + 0.0) / 2.0 + 1e-5).sqrt();
        // Row 1: [0, 1] → rms = sqrt(0.5) ≈ 0.7071
        let rms1 = ((0.0f32 + 1.0) / 2.0 + 1e-5).sqrt();

        assert!((output[0] - 1.0 / rms0).abs() < 1e-4);
        assert!((output[1] - 0.0 / rms0).abs() < 1e-4);
        assert!((output[2] - 0.0 / rms1).abs() < 1e-4);
        assert!((output[3] - 1.0 / rms1).abs() < 1e-4);
    }

    #[test]
    fn rmsnorm_uniform_input() {
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let norm = RmsNorm::new(weight, 1e-5);

        let x = vec![2.0, 2.0, 2.0, 2.0];
        let output = norm.forward(&x, 1);

        // rms = sqrt(16/4) = sqrt(4) = 2.0
        // output = 2.0 / 2.0 * 1.0 = 1.0
        for v in &output {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }
}
