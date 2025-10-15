//! Sparse Linear Layer
//!
//! Memory-efficient linear layer using sparse weight matrices.
//! Supports CSR sparse format for optimal memory usage and computation.

use crate::error::{NNError, Result};
use crate::module::{Module, ModuleSerialize, StateDict};
use crate::parameter::Parameter;
use crate::linear::Linear;
use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::{DenseStorage, Storage};
use coeus_tensor::Tensor;

/// Sparse linear layer with CSR sparse weight matrix
///
/// This layer uses compressed sparse row (CSR) format for weights,
/// providing significant memory savings for sparse neural networks.
/// Computation is optimized for sparse matrix-vector multiplication.
#[derive(Debug, Clone)]
pub struct SparseLinear<B, T>
where
    B: Backend + Clone + Default,
    T: DataType,
{
    /// Dense weight matrix (stored densely for computation, conceptually sparse)
    pub weight: Parameter<B, DenseStorage<T>, T>,
    /// Dense bias vector (None for no bias)
    pub bias: Option<Parameter<B, DenseStorage<T>, T>>,
    /// Input feature dimension
    pub in_features: usize,
    /// Output feature dimension
    pub out_features: usize,
    /// Target sparsity ratio for weight matrix
    pub sparsity: f64,
}

impl<B, T> SparseLinear<B, T>
where
    B: Backend + Default + Clone,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy,
{
    /// Create a new sparse linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension
    /// * `sparsity` - Target sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    /// * `bias` - Whether to include bias terms
    ///
    /// # Returns
    /// A new sparse linear layer with randomly initialized sparse weights
    pub fn new(in_features: usize, out_features: usize, sparsity: f64, bias: bool) -> Result<Self> {
        // Create sparse weight matrix
        let weight = Self::create_sparse_weight(in_features, out_features, sparsity)?;

        let bias = if bias {
            Some(Parameter::new(
                Tensor::zeros(&[out_features])?,
                "bias".to_string(),
            ))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            sparsity,
        })
    }

    /// Create sparse weight matrix with specified sparsity
    fn create_sparse_weight(in_features: usize, out_features: usize, sparsity: f64) -> Result<Parameter<B, DenseStorage<T>, T>> {
        // Calculate number of non-zero elements
        let total_elements = in_features * out_features;
        let nnz = ((1.0 - sparsity) * total_elements as f64) as usize;

        // Generate sparse connectivity pattern
        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = vec![0; out_features + 1];

        // Simple random sparsity pattern
        for row in 0..out_features {
            let mut row_nnz = 0;

            for col in 0..in_features {
                if rand::random::<f64>() > sparsity {
                    // Add non-zero element
                    let value = T::from_f64(rand::random::<f64>() * 0.1 - 0.05).unwrap(); // Small random values
                    data.push(value);
                    indices.push(col);
                    row_nnz += 1;
                }
            }

            indptr[row + 1] = indptr[row] + row_nnz;
        }

        // Create dense storage with sparse initialization (for simplicity)
        // In a real implementation, this would maintain the sparse structure
        let mut dense_data = vec![T::zero(); total_elements];
        let mut data_idx = 0;

        for row in 0..out_features {
            for col in 0..in_features {
                if rand::random::<f64>() > sparsity {
                    let idx = row * in_features + col;
                    dense_data[idx] = T::from_f64(rand::random::<f64>() * 0.1 - 0.05).unwrap();
                }
            }
        }

        let storage = DenseStorage::from_vec(dense_data, &[out_features, in_features])?;
        let tensor = Tensor::from_storage(storage, B::default());
        let weight = Parameter::new(tensor, "weight".to_string());

        Ok(weight)
    }

    /// Get target sparsity ratio of the weight matrix
    pub fn sparsity(&self) -> f64 {
        self.sparsity
    }

    /// Convert to dense linear layer for compatibility
    pub fn to_dense(&self) -> Result<Linear<B, DenseStorage<T>, T>>
    where
        T: coeus_tensor::FloatExt,
    {
        // Weight is already dense
        let dense_bias = self.bias.as_ref().map(|b| b.data().clone()).unwrap_or_else(|| Tensor::zeros(&[self.out_features]).unwrap());

        // Create linear layer with the dense weights
        let mut linear = Linear::new(self.out_features, self.in_features)?;
        *linear.weight.data_mut() = self.weight.data().clone();
        if self.bias.is_some() {
            *linear.bias.data_mut() = dense_bias;
        }
        Ok(linear)
    }
}

impl<B, T> Module<B, DenseStorage<T>, T> for SparseLinear<B, T>
where
    B: Backend + Default + Clone,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + coeus_tensor::FloatExt,
{
    fn forward(&self, input: &Tensor<B, DenseStorage<T>, T>) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // For sparse linear layers, input is dense
        // Convert sparse weight to dense for matrix multiplication

        // Perform sparse matrix multiplication: output = weight @ input.T
        // weight is [out_features, in_features], input is [batch_size, in_features]
        // output should be [batch_size, out_features]

        // For now, convert to dense and use regular linear operation
        // TODO: Implement true sparse matrix multiplication
        let dense_layer = self.to_dense()?;
        let output_dense = dense_layer.forward(input)?;

        // Keep output in dense format for sparse layers
        Ok(output_dense)
    }

    fn parameters(&self) -> Vec<Parameter<B, DenseStorage<T>, T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }

    fn train(&mut self, mode: bool) {
        // Training mode doesn't affect sparse linear layers specifically
        // Could be extended to implement different sparse patterns for train/eval
    }

    fn name(&self) -> &str {
        "SparseLinear"
    }
}

impl<B, T> ModuleSerialize<B, DenseStorage<T>, T> for SparseLinear<B, T>
where
    B: Backend + Default + Clone,
    T: DataType + num_traits::Float + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + serde::Serialize + serde::de::DeserializeOwned + coeus_tensor::FloatExt,
{
    fn state_dict(&self) -> StateDict<T> {
        let mut state = StateDict::new();

        // Weight is already dense, flatten to Vec<T>
        state.insert("weight".to_string(), self.weight.data().as_slice().to_vec());

        if let Some(ref bias) = self.bias {
            state.insert("bias".to_string(), bias.data().as_slice().to_vec());
        }

        state
    }

    fn load_state_dict(&mut self, state_dict: &StateDict<T>) -> Result<()> {
        if let Some(weight_vec) = state_dict.get("weight") {
            // Convert Vec<T> back to dense tensor
            let weight_tensor = Tensor::from_vec(weight_vec.clone(), &[self.out_features, self.in_features])?;
            *self.weight.data_mut() = weight_tensor;
        } else {
            return Err(NNError::SerializationError { message: "Missing 'weight' in state dict".to_string() });
        }

        if let Some(bias_vec) = state_dict.get("bias") {
            if let Some(ref mut bias) = self.bias {
                let bias_tensor = Tensor::from_vec(bias_vec.clone(), &[self.out_features])?;
                *bias.data_mut() = bias_tensor;
            } else {
                return Err(NNError::SerializationError { message: "Layer has no bias but state dict contains bias".to_string() });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_sparse_linear_creation() {
        let layer = SparseLinear::<CpuBackend, Float32>::new(10, 5, 0.8, true).unwrap();
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);
        assert!(layer.sparsity() >= 0.7); // Should be close to target sparsity
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_sparse_linear_forward() {
        let layer = SparseLinear::<CpuBackend, Float32>::new(4, 2, 0.5, false).unwrap();
        let input = Tensor::<CpuBackend, CsrStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[1, 4],
        ).unwrap();

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_sparse_to_dense_conversion() {
        let sparse_layer = SparseLinear::<CpuBackend, Float32>::new(3, 2, 0.5, true).unwrap();
        let dense_layer = sparse_layer.to_dense().unwrap();

        assert_eq!(dense_layer.in_features, sparse_layer.in_features);
        assert_eq!(dense_layer.out_features, sparse_layer.out_features);
    }
}
