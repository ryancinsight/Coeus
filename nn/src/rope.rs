//! Rotary Position Embedding (RoPE) for Transformers
//!
//! RoPE introduces relative position information by rotating token embeddings
//! based on their absolute positions. This provides better length generalization
//! and is used in modern transformer architectures like GPT-J, GPT-NeoX, and PaLM.

use crate::error::{NNError, Result};
use backend::Backend;
use storage::{Storage, DenseStorage, StorageFromVec, StorageToDense};
use dtype::DataType;
use tensor::{FloatExt, Tensor, ops::arithmetic::*};

/// Rotary Position Embedding (RoPE) implementation
///
/// RoPE applies a rotation to query and key vectors based on their positions:
/// RoPE(x, m) = x * cos(mθ) + R(x) * sin(mθ)
/// where R(x) is the rotation of x by 90 degrees and θ is a function of position
pub struct RoPE<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
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
            return Err(NNError::InvalidConfiguration {
                message: "Head dimension must be even for RoPE".to_string(),
            });
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
            return Err(NNError::InvalidInput {
                message: "RoPE input must be 4D tensor [batch_size, seq_len, num_heads, head_dim]".to_string(),
            });
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];

        if head_dim != self.head_dim {
            return Err(NNError::InvalidInput {
                message: format!("Head dimension mismatch: expected {}, got {}", self.head_dim, head_dim),
            });
        }

        // Handle positions
        let default_positions = (0..seq_len).collect::<Vec<_>>();
        let positions = positions.unwrap_or(&default_positions);

        if positions.len() != seq_len {
            return Err(NNError::InvalidInput {
                message: "Positions length must match sequence length".to_string(),
            });
        }

        // Check max position
        let max_pos = positions.iter().max().unwrap_or(&0);
        if *max_pos >= self.max_seq_len {
            return Err(NNError::InvalidInput {
                message: format!("Position {} exceeds max sequence length {}", max_pos, self.max_seq_len),
            });
        }

        // Apply RoPE rotation
        self.apply_rope_rotation(x, positions, is_query)
    }

    /// Apply the actual RoPE rotation computation
    fn apply_rope_rotation(
        &self,
        _x: &Tensor<B, S, T>,
        _positions: &[usize],
        _is_query: bool,
    ) -> Result<Tensor<B, S, T>> {
        // TODO: Implement tensor indexing and selection operations
        Err(NNError::NotImplemented {
            operation: "apply_rope_rotation".to_string(),
        })
    }

    /// Rotate half of the vector using RoPE formula
    fn rotate_half(
        &self,
        _x: &Tensor<B, S, T>,
        _cos_vals: &Tensor<B, S, T>,
        _sin_vals: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        // TODO: Implement tensor value access operations
        Err(NNError::NotImplemented {
            operation: "rotate_half".to_string(),
        })
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

        let cos_cache = Tensor::from_vec(cos_vals, &[max_seq_len, half_dim])?;
        let sin_cache = Tensor::from_vec(sin_vals, &[max_seq_len, half_dim])?;

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
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use num_traits::Float;

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
        let input_data = vec![TestDataType::from(0.1); 1 * 2 * 1 * 64]; // 128 elements
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();

        let output = rope.apply_to_query(&input, None).unwrap();

        // Output should have same shape
        assert_eq!(output.shape().dims(), &input_shape);
    }

    #[test]
    fn test_rope_different_positions() {
        let rope = RoPE::<TestBackend, TestStorage, TestDataType>::new(64, 1024, 10000.0).unwrap();

        // Create identical inputs but apply different positions
        let input_shape = [1, 1, 1, 64];
        let input_data = vec![TestDataType::from(1.0); 64];
        let input1 = Tensor::from_vec(input_data.clone(), &input_shape).unwrap();
        let input2 = Tensor::from_vec(input_data, &input_shape).unwrap();

        // Apply RoPE with different positions
        // pos=0: no rotation (if theta^0 = 1, cos(0)=1, sin(0)=0)
        let positions1 = vec![0];
        let output1 = rope.apply_to_query(&input1, Some(&positions1)).unwrap();
        
        // pos=1: some rotation
        let positions2 = vec![1];
        let output2 = rope.apply_to_query(&input2, Some(&positions2)).unwrap();

        // Outputs should be different
        let out1_data = output1.as_slice();
        let out2_data = output2.as_slice();
        
        let mut different = false;
        for i in 0..out1_data.len() {
            if (out1_data[i] - out2_data[i]).abs() > TestDataType::from(1e-6) {
                different = true;
                break;
            }
        }
        assert!(different, "Outputs should be different for different positions");
    }

    #[test]
    fn test_rope_cos_sin_cache() {
        let rope = RoPE::<TestBackend, TestStorage, TestDataType>::new(64, 1024, 10000.0).unwrap();
        
        // Access cache (this requires making cache public or adding a method, 
        // but for now we just test public interface behavior)
        
        // Request a large position, should trigger cache update or work correctly
        let input_shape = [1, 1, 1, 64];
        let input_data = vec![TestDataType::from(1.0); 64];
        let input = Tensor::from_vec(input_data, &input_shape).unwrap();
        
        // Position within initial limit
        let positions1 = vec![512];
        let _ = rope.apply_to_query(&input, Some(&positions1)).unwrap();
        
        // Position beyond initial limit? No, we set 1024.
        // But let's try extending
        
        let mut rope = rope;
        rope.extend_max_seq_len(2048).unwrap();
        
        // Now position 1500 should work
        let positions2 = vec![1500];
        let _ = rope.apply_to_query(&input, Some(&positions2)).unwrap();
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

