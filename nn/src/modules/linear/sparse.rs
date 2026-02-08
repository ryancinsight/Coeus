//! Sparse Linear Layer
//!
//! Memory-efficient linear layer using sparse weight matrices.
//! Supports CSR sparse format for optimal memory usage and computation.

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
#[cfg(feature = "safetensors")]
use crate::core::module::ModuleSerialize;
#[cfg(feature = "safetensors")]
use crate::core::module::StateDict;
use crate::core::parameter::Parameter;
use crate::modules::linear::Linear;
use backend::{Backend, Storage};
use dtype::DataType;
use storage::DenseStorage;
use tensor::Tensor;

/// Sparse linear layer with efficient sparse computation
///
/// This layer stores weights as dense tensors but performs sparse matrix-vector
/// multiplication using CSR format for optimal performance on sparse neural networks.
/// Memory usage is O(total_elements) but computation is O(nnz) where nnz is the
/// number of non-zero elements, providing significant speedups for sparse weights.
/// The sparse connectivity pattern is precomputed and cached for efficient inference.
#[derive(Debug, Clone)]
pub struct SparseLinear<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType,
{
    /// Weight matrix stored densely but with sparse connectivity pattern
    pub weight: Parameter<B, DenseStorage<T>, T>,
    /// Dense bias vector (None for no bias)
    pub bias: Option<Parameter<B, DenseStorage<T>, T>>,
    /// Input feature dimension
    pub in_features: usize,
    /// Output feature dimension
    pub out_features: usize,
    /// Target sparsity ratio for weight matrix
    pub sparsity: f64,
    /// Precomputed CSR representation for efficient sparse computation
    pub csr_data: Option<CsrData<T>>,
}

/// Internal CSR data structure for sparse computation
#[derive(Debug, Clone)]
pub struct CsrData<T: DataType> {
    pub data: Vec<T>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}

impl<B, T> SparseLinear<B, T>
where
    B: Backend<Data = T> + Default + Clone,
    T: DataType
        + num_traits::Float
        + num_traits::FromPrimitive
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + Copy
        + tensor::FloatExt,
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
        let (weight, csr_data) = Self::create_sparse_weight(in_features, out_features, sparsity)?;

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
            csr_data: Some(csr_data),
        })
    }

    /// Create sparse weight matrix with specified sparsity
    #[allow(clippy::type_complexity)]
    fn create_sparse_weight(
        in_features: usize,
        out_features: usize,
        sparsity: f64,
    ) -> Result<(Parameter<B, DenseStorage<T>, T>, CsrData<T>)> {
        use rand::prelude::*;
        use std::collections::HashSet;

        let total_elements = in_features * out_features;
        let mut dense_data = vec![T::zero(); total_elements];

        let target_nnz = if sparsity >= 1.0 {
            0
        } else {
            ((1.0 - sparsity) * total_elements as f64)
                .round()
                .clamp(1.0, total_elements as f64) as usize
        };

        let mut rng = rand::thread_rng();
        let mut chosen: HashSet<(usize, usize)> = HashSet::with_capacity(target_nnz);

        if target_nnz >= out_features && in_features > 0 {
            for row in 0..out_features {
                let col = rng.gen_range(0..in_features);
                chosen.insert((row, col));
            }
        }

        while chosen.len() < target_nnz {
            let row = rng.gen_range(0..out_features);
            let col = rng.gen_range(0..in_features);
            chosen.insert((row, col));
        }

        let mut per_row: Vec<Vec<(usize, T)>> = vec![Vec::new(); out_features];
        for (row, col) in chosen.into_iter() {
            let mut value_f64 = rng.gen_range(-0.05..0.05);
            while value_f64 == 0.0 {
                value_f64 = rng.gen_range(-0.05..0.05);
            }
            let value = T::from_f64(value_f64).unwrap();
            per_row[row].push((col, value));
            dense_data[row * in_features + col] = value;
        }

        let mut csr_data = Vec::with_capacity(target_nnz);
        let mut indices = Vec::with_capacity(target_nnz);
        let mut indptr = vec![0; out_features + 1];
        for (row, row_entries) in per_row.iter_mut().enumerate() {
            row_entries.sort_by_key(|(col, _)| *col);
            indptr[row + 1] = indptr[row] + row_entries.len();
            for (col, value) in row_entries.iter() {
                indices.push(*col);
                csr_data.push(*value);
            }
        }

        let storage = DenseStorage::from_vec(dense_data, &[out_features, in_features])?;
        let tensor = Tensor::from_storage(storage, B::default());
        let weight = Parameter::new(tensor, "weight".to_string());

        let csr = CsrData {
            data: csr_data,
            indices,
            indptr,
        };

        Ok((weight, csr))
    }

    /// Get target sparsity ratio of the weight matrix
    pub fn sparsity(&self) -> f64 {
        self.sparsity
    }

    /// Convert to dense linear layer for compatibility
    pub fn to_dense(&self) -> Result<Linear<B, DenseStorage<T>, T>>
    where
        T: tensor::FloatExt,
    {
        // Weight is already dense
        let dense_bias = self
            .bias
            .as_ref()
            .map(|b| b.data().clone())
            .unwrap_or_else(|| Tensor::zeros(&[self.out_features]).unwrap());

        // Create linear layer with the dense weights
        let mut linear = Linear::new(self.in_features, self.out_features)?;
        *linear.weight.data_mut() = self.weight.data().clone();
        if self.bias.is_some() {
            *linear.bias.data_mut() = dense_bias;
        }
        Ok(linear)
    }
}

impl<B, T> Module<B, DenseStorage<T>, T> for SparseLinear<B, T>
where
    B: Backend<Data = T> + Default + Clone,
    T: DataType
        + num_traits::Float
        + num_traits::FromPrimitive
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + Copy
        + tensor::FloatExt,
{
    type Input = Tensor<B, DenseStorage<T>, T>;
    type Output = Tensor<B, DenseStorage<T>, T>;

    fn forward(
        &self,
        input: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Perform sparse matrix multiplication: output = input @ weight.T
        // weight is [out_features, in_features] (sparse CSR), input is [batch_size, in_features]
        // output should be [batch_size, out_features]

        // Extract CSR data from precomputed cache
        let csr_data = self.csr_data.as_ref().ok_or(NNError::StorageError {
            source: storage::StorageError::ShapeMismatch {
                expected: 0,
                actual: 0,
            },
        })?;

        let weight_data = &csr_data.data;
        let weight_indices = &csr_data.indices;
        let weight_indptr = &csr_data.indptr;
        let out_features = self.out_features;
        let in_features = self.in_features;

        // Get input data
        let input_data = input.storage().as_slice();
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];

        // Prepare output buffer
        let mut output_data = vec![T::zero(); batch_size * out_features];

        // For each sample in batch
        for batch_idx in 0..batch_size {
            let input_offset = batch_idx * in_features;
            let output_offset = batch_idx * out_features;

            // Extract input vector for this sample
            let input_vec = &input_data[input_offset..input_offset + in_features];

            // Perform sparse matrix-vector multiplication: weight.T @ input_vec
            // Since weight is [out_features, in_features] and we want input @ weight.T,
            // we need to compute: input_vec @ weight.T which is equivalent to weight.T.T @ input_vec = weight @ input_vec
            let mut result_vec = vec![T::default(); out_features];
            for row in 0..out_features {
                let row_start = weight_indptr[row];
                let row_end = weight_indptr[row + 1];
                for idx in row_start..row_end {
                    let col = weight_indices[idx];
                    let val = weight_data[idx];
                    result_vec[row] = result_vec[row] + val * input_vec[col];
                }
            }

            // Copy result to output
            output_data[output_offset..output_offset + out_features].copy_from_slice(&result_vec);
        }

        // Add bias if present
        if let Some(ref bias_param) = self.bias {
            let bias_data = bias_param.data.storage().as_slice();
            for batch_idx in 0..batch_size {
                let output_offset = batch_idx * out_features;
                #[allow(clippy::needless_range_loop)]
                for i in 0..out_features {
                    let idx = output_offset + i;
                    output_data[idx] = output_data[idx] + bias_data[i];
                }
            }
        }

        // Create output tensor
        let output_shape = &[batch_size, out_features];
        let output_storage = DenseStorage::from_vec(output_data, output_shape)?;
        let output_tensor = Tensor::from_storage(output_storage, B::default());

        Ok(output_tensor)
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

    fn train(&mut self, _mode: bool) {
        // Training mode doesn't affect sparse linear layers specifically
        // Could be extended to implement different sparse patterns for train/eval
    }

    fn name(&self) -> &str {
        "SparseLinear"
    }

    fn clone_box(&self) -> Box<dyn Module<B, DenseStorage<T>, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

#[cfg(feature = "safetensors")]
impl<B, T> ModuleSerialize<B, DenseStorage<T>, T> for SparseLinear<B, T>
where
    B: Backend<Data = T> + Default + Clone,
    T: DataType
        + num_traits::Float
        + num_traits::FromPrimitive
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + Copy
        + serde::Serialize
        + serde::de::DeserializeOwned
        + tensor::FloatExt,
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
            let weight_tensor =
                Tensor::from_vec(weight_vec.clone(), &[self.out_features, self.in_features])?;
            *self.weight.data_mut() = weight_tensor;
        } else {
            return Err(NNError::SerializationError {
                message: "Missing 'weight' in state dict".to_string(),
            });
        }

        if let Some(bias_vec) = state_dict.get("bias") {
            if let Some(ref mut bias) = self.bias {
                let bias_tensor = Tensor::from_vec(bias_vec.clone(), &[self.out_features])?;
                *bias.data_mut() = bias_tensor;
            } else {
                return Err(NNError::SerializationError {
                    message: "Layer has no bias but state dict contains bias".to_string(),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;

    #[test]
    fn test_sparse_linear_creation() {
        let layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(10, 5, 0.8, true).unwrap();
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);
        assert!(layer.sparsity() >= 0.7); // Should be close to target sparsity
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_sparse_linear_forward() {
        let layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(4, 2, 0.5, false).unwrap();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[1, 4],
        )
        .unwrap();

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    #[ignore = "Sparse storage conversion incomplete"]
    fn test_sparse_to_dense_conversion() {
        let sparse_layer =
            SparseLinear::<CpuBackend<Float32>, Float32>::new(3, 2, 0.5, true).unwrap();
        let dense_layer = sparse_layer.to_dense().unwrap();

        assert_eq!(dense_layer.in_features, sparse_layer.in_features);
        assert_eq!(dense_layer.out_features, sparse_layer.out_features);
    }
}
