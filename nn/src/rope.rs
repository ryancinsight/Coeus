//! Rotary Position Embedding (RoPE) for Transformers
//!
//! RoPE introduces relative position information by rotating token embeddings
//! based on their absolute positions. This provides better length generalization
//! and is used in modern transformer architectures like GPT-J, GPT-NeoX, and PaLM.

use crate::error::{NNError, Result};
use crate::backend::Backend;
use crate::storage::{Storage, DenseStorage, StorageFromVec, StorageToDense};
use crate::dtype::{DataType, FloatExt};
use crate::tensor::Tensor;

/// Rotary Position Embedding (RoPE) implementation
///
/// RoPE applies a rotation to query and key vectors based on their positions:
/// RoPE(x, m) = x * cos(mθ) + R(x) * sin(mθ)
/// where R(x) is the rotation of x by 90 degrees and θ is a function of position
pub struct RoPE<B, S, T> {
    /// Pre-computed cos and sin values for all positions
    cos_cache: Tensor<B, S, T>,
    sin_cache: Tensor<B, S, T>,
    /// Maximum sequence length this RoPE was initialized for
    max_seq_len: usize,
    /// Head dimension (must be even for complex number representation)
    head_dim: usize,
    /// Base for the frequency computation
    theta_base: f64,
}

impl<B, S, T> RoPE<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new RoPE instance
    ///
    /// # Arguments
    /// * `head_dim` - Head dimension (must be even)
    /// * `max_seq_len` - Maximum sequence length to pre-compute
    /// * `theta_base` - Base for frequency computation (typically 10000.0)
    pub fn new(head_dim: usize, max_seq_len: usize, theta_base: f64) -> Result<Self> {
        if head_dim % 2 != 0 {
            return Err(NNError::InvalidConfiguration(
                "Head dimension must be even for RoPE".to_string(),
            ));
        }

        let (cos_cache, sin_cache) = Self::precompute_rope_cache(head_dim, max_seq_len, theta_base)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            max_seq_len,
            head_dim,
            theta_base,
        })
    }

    /// Apply RoPE to query tensor
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
    /// * `positions` - Optional position indices (uses 0..seq_len if None)
    pub fn apply_to_query(
        &self,
        q: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
    ) -> Result<Tensor<B, S, T>> {
        self.apply_rope(q, positions, true)
    }

    /// Apply RoPE to key tensor
    ///
    /// # Arguments
    /// * `k` - Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
    /// * `positions` - Optional position indices (uses 0..seq_len if None)
    pub fn apply_to_key(
        &self,
        k: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
    ) -> Result<Tensor<B, S, T>> {
        self.apply_rope(k, positions, false)
    }

    /// Apply RoPE to both query and key tensors
    ///
    /// # Arguments
    /// * `q` - Query tensor
    /// * `k` - Key tensor
    /// * `positions` - Optional position indices
    pub fn apply_to_qk(
        &self,
        q: &Tensor<B, S, T>,
        k: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
    ) -> Result<(Tensor<B, S, T>, Tensor<B, S, T>)> {
        let q_rotated = self.apply_to_query(q, positions)?;
        let k_rotated = self.apply_to_key(k, positions)?;

        Ok((q_rotated, k_rotated))
    }

    /// Core RoPE application logic
    fn apply_rope(
        &self,
        x: &Tensor<B, S, T>,
        positions: Option<&[usize]>,
        is_query: bool,
    ) -> Result<Tensor<B, S, T>> {
        let shape = x.shape().dims();

        // Expect shape: [batch_size, seq_len, num_heads, head_dim]
        if shape.len() != 4 {
            return Err(NNError::InvalidInput(
                "RoPE input must be 4D tensor [batch_size, seq_len, num_heads, head_dim]".to_string(),
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];

        if head_dim != self.head_dim {
            return Err(NNError::InvalidInput(
                format!("Head dimension mismatch: expected {}, got {}", self.head_dim, head_dim),
            ));
        }

        // Handle positions
        let positions = positions.unwrap_or(&(0..seq_len).collect::<Vec<_>>());

        if positions.len() != seq_len {
            return Err(NNError::InvalidInput(
                "Positions length must match sequence length".to_string(),
            ));
        }

        // Check max position
        let max_pos = positions.iter().max().unwrap_or(&0);
        if *max_pos >= self.max_seq_len {
            return Err(NNError::InvalidInput(
                format!("Position {} exceeds max sequence length {}", max_pos, self.max_seq_len),
            ));
        }

        // Apply RoPE rotation
        self.apply_rope_rotation(x, positions, is_query)
    }

    /// Apply the actual RoPE rotation computation
    fn apply_rope_rotation(
        &self,
        x: &Tensor<B, S, T>,
        positions: &[usize],
        _is_query: bool,
    ) -> Result<Tensor<B, S, T>> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];

        // Create output tensor
        let mut output = Tensor::zeros_like(x)?;

        // For each batch, sequence position, and head
        for b in 0..batch_size {
            for seq_pos in 0..seq_len {
                let pos_idx = positions[seq_pos];

                for h in 0..num_heads {
                    // Extract the head vector: [head_dim]
                    let head_start = ((b * seq_len * num_heads + seq_pos * num_heads + h) * self.head_dim) as i32;
                    let head_indices = (head_start..head_start + self.head_dim as i32).collect::<Vec<_>>();
                    let head_vec = x.index_select(&head_indices)?;

                    // Get cos and sin for this position: [head_dim/2]
                    let cos_start = (pos_idx * (self.head_dim / 2)) as i32;
                    let cos_indices = (cos_start..cos_start + (self.head_dim / 2) as i32).collect::<Vec<_>>();
                    let cos_vals = self.cos_cache.index_select(&cos_indices)?;

                    let sin_start = (pos_idx * (self.head_dim / 2)) as i32;
                    let sin_indices = (sin_start..sin_start + (self.head_dim / 2) as i32).collect::<Vec<_>>();
                    let sin_vals = self.sin_cache.index_select(&sin_indices)?;

                    // Apply RoPE rotation: treat head_dim/2 complex pairs
                    let rotated_head = self.rotate_half(&head_vec, &cos_vals, &sin_vals)?;

                    // Store result
                    output.set_values(&head_indices, &rotated_head)?;
                }
            }
        }

        Ok(output)
    }

    /// Rotate half of the vector using RoPE formula
    fn rotate_half(
        &self,
        x: &Tensor<B, S, T>,
        cos_vals: &Tensor<B, S, T>,
        sin_vals: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let half_dim = self.head_dim / 2;

        // Split into even and odd indices for complex rotation
        let mut rotated = Vec::new();

        for i in 0..half_dim {
            // x[2*i] and x[2*i+1] form a complex number
            let real_part = x.get_value(&[2 * i])?;
            let imag_part = x.get_value(&[2 * i + 1])?;
            let cos_val = cos_vals.get_value(&[i])?;
            let sin_val = sin_vals.get_value(&[i])?;

            // RoPE rotation: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
            let rotated_real = real_part * cos_val - imag_part * sin_val;
            let rotated_imag = real_part * sin_val + imag_part * cos_val;

            rotated.push(rotated_real);
            rotated.push(rotated_imag);
        }

        Tensor::new_from_vec(rotated, &[self.head_dim])
    }

    /// Pre-compute RoPE cos and sin cache
    fn precompute_rope_cache(
        head_dim: usize,
        max_seq_len: usize,
        theta_base: f64,
    ) -> Result<(Tensor<B, S, T>, Tensor<B, S, T>)> {
        let half_dim = head_dim / 2;
        let mut cos_vals = Vec::new();
        let mut sin_vals = Vec::new();

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                // Compute frequency: theta_i = theta_base^(-2*i/head_dim)
                let theta = theta_base.powf(-2.0 * i as f64 / head_dim as f64);

                // Compute angle: angle = pos * theta
                let angle = pos as f64 * theta;

                cos_vals.push(T::from(angle.cos()).unwrap());
                sin_vals.push(T::from(angle.sin()).unwrap());
            }
        }

        let cos_cache = Tensor::new_from_vec(cos_vals, &[max_seq_len, half_dim])?;
        let sin_cache = Tensor::new_from_vec(sin_vals, &[max_seq_len, half_dim])?;

        Ok((cos_cache, sin_cache))
    }

    /// Get the maximum sequence length this RoPE can handle
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Extend RoPE to handle longer sequences (recompute cache)
    pub fn extend_max_seq_len(&mut self, new_max_seq_len: usize) -> Result<()> {
        if new_max_seq_len <= self.max_seq_len {
            return Ok(());
        }

        let (cos_cache, sin_cache) = Self::precompute_rope_cache(
            self.head_dim,
            new_max_seq_len,
            self.theta_base,
        )?;

        self.cos_cache = cos_cache;
        self.sin_cache = sin_cache;
        self.max_seq_len = new_max_seq_len;

        Ok(())
    }
}

/// RoPE configuration for different transformer architectures
pub struct RoPEConfig {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub theta_base: f64,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            head_dim: 64,      // Typical head dimension
            max_seq_len: 2048, // Sufficient for most tasks
            theta_base: 10000.0, // Standard RoPE base
        }
    }
}

impl RoPEConfig {
    /// Configuration for GPT-J style models
    pub fn gpt_j() -> Self {
        Self {
            head_dim: 64,
            max_seq_len: 2048,
            theta_base: 10000.0,
        }
    }

    /// Configuration for GPT-NeoX style models
    pub fn gpt_neox() -> Self {
        Self {
            head_dim: 128,
            max_seq_len: 4096,
            theta_base: 10000.0,
        }
    }

    /// Configuration for PaLM style models
    pub fn palm() -> Self {
        Self {
            head_dim: 128,
            max_seq_len: 2048,
            theta_base: 10000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::dtype::float::Float32;
    use crate::storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;
    type TestDataType = Float32;

    #[test]
    fn test_rope_creation() {
        let config = RoPEConfig::default();
        let rope = RoPE::<TestBackend, TestStorage, TestDataType>::new(
            config.head_dim,
            config.max_seq_len,
            config.theta_base,
        ).unwrap();

        assert_eq!(rope.head_dim(), config.head_dim);
        assert_eq!(rope.max_seq_len(), config.max_seq_len);
    }

    #[test]
    fn test_rope_odd_head_dim() {
        // Head dimension must be even
        let result = RoPE::<TestBackend, TestStorage, TestDataType>::new(63, 1024, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_extension() {
        let mut rope = RoPE::<TestBackend, TestStorage, TestDataType>::new(64, 1024, 10000.0).unwrap();
        assert_eq!(rope.max_seq_len(), 1024);

        rope.extend_max_seq_len(2048).unwrap();
        assert_eq!(rope.max_seq_len(), 2048);
    }

    #[test]
    fn test_rope_preserves_shape() {
        let rope = RoPE::<TestBackend, TestStorage, TestDataType>::new(64, 1024, 10000.0).unwrap();

        // Create test input: [batch_size=1, seq_len=2, num_heads=1, head_dim=64]
        let input_shape = [1, 2, 1, 64];
        let input_data = vec![0.1; 1 * 2 * 1 * 64]; // 128 elements
        let input = Tensor::new_from_vec(input_data, &input_shape).unwrap();

        let output = rope.apply_to_query(&input, None).unwrap();

        // Output should have same shape
        assert_eq!(output.shape().dims(), &input_shape);
    }

    #[test]
    fn test_rope_different_positions() {
        let rope = RoPE::<TestBackend, TestStorage, TestDataType>::new(64, 1024, 10000.0).unwrap();

        // Create identical inputs but apply different positions
        let input_shape = [1, 1, 1, 64];
        let input_data = vec![1.0; 64];
        let input = Tensor::new_from_vec(input_data, &input_shape).unwrap();

        let pos_0 = rope.apply_to_query(&input, Some(&[0])).unwrap();
        let pos_1 = rope.apply_to_query(&input, Some(&[1])).unwrap();

        // Results should be different due to position encoding
        let pos_0_data = pos_0.as_slice();
        let pos_1_data = pos_1.as_slice();

        // At least some values should be different
        let mut has_difference = false;
        for (a, b) in pos_0_data.iter().zip(pos_1_data.iter()) {
            if (a - b).abs() > 1e-6 {
                has_difference = true;
                break;
            }
        }
        assert!(has_difference, "RoPE should produce different outputs for different positions");
    }

    #[test]
    fn test_rope_configs() {
        let gpt_j = RoPEConfig::gpt_j();
        assert_eq!(gpt_j.head_dim, 64);
        assert_eq!(gpt_j.max_seq_len, 2048);

        let gpt_neox = RoPEConfig::gpt_neox();
        assert_eq!(gpt_neox.head_dim, 128);
        assert_eq!(gpt_neox.max_seq_len, 4096);
    }
}





