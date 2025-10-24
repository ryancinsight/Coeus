//! Linear (fully connected) neural network layer.

use std::fmt;

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use coeus_tensor::Tensor;
use num_traits;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

/// A linear (fully connected) neural network layer.
///
/// Performs the operation: `output = input @ weight + bias`
///
/// Where:
/// - `input`: [batch_size, input_features]
/// - `weight`: [input_features, output_features]
/// - `bias`: [output_features]
/// - `output`: [batch_size, output_features]
///
/// # Examples
/// ```rust
/// use coeus_nn::{Linear, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::{DenseStorage, CsrStorage};
/// use coeus_dtype::float::Float32;
///
/// // Create a linear layer: 784 -> 128
/// let layer = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 128).unwrap();
/// assert_eq!(layer.in_features, 784);
/// assert_eq!(layer.out_features, 128);
/// ```
#[derive(Debug, Clone)]
pub struct Linear<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Weight matrix [input_features, output_features]
    pub weight: Parameter<B, S, T>,
    /// Bias vector [output_features]
    pub bias: Parameter<B, S, T>,
    /// Cached transposed weight matrix for efficient forward pass
    pub weight_t: Option<Tensor<B, DenseStorage<T>, T>>,
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
}

impl<B, S, T> Linear<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive,
{
    /// Create a new linear layer.
    ///
    /// Weights are initialized using Xavier/Glorot uniform initialization.
    /// Bias is initialized to zeros.
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Errors
    /// Returns `NNError::InvalidInput` if `in_features` or `out_features` is 0.
    pub fn new_with_backend(backend: B, in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "in_features must be > 0".to_string(),
            });
        }
        if out_features == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "out_features must be > 0".to_string(),
            });
        }

        // Xavier/Glorot uniform initialization: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
        let limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        let weight_data =
            Self::xavier_uniform_init_with_backend(&backend, in_features, out_features, limit)?;
        let bias_storage = S::zeros(&[out_features])?;
        let bias_data = Tensor::<B, S, T>::from_storage(bias_storage, backend.clone());

        let weight = Parameter::new(
            weight_data.clone().requires_grad_(true),
            "weight".to_string(),
        );
        let bias = Parameter::new(bias_data.requires_grad_(true), "bias".to_string());

        // Cache transposed weight for efficient forward passes
        let weight_t = Some(weight_data.to_dense_generic()?.transpose(1, 0)?);

        Ok(Self {
            weight,
            bias,
            weight_t,
            in_features,
            out_features,
        })
    }

    /// Create a new linear layer with default backend.
    ///
    /// This is a convenience method that uses `B::default()` as the backend.
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Errors
    /// Returns `NNError::InvalidInput` if `in_features` or `out_features` is 0.
    pub fn new(in_features: usize, out_features: usize) -> Result<Self>
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), in_features, out_features)
    }

    /// Create a new linear layer with sparse weight initialization.
    ///
    /// This creates a linear layer with dense storage but initializes weights using
    /// a sparse connectivity pattern for memory-efficient neural networks.
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `sparsity` - Target sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    ///
    /// # Errors
    /// Returns `NNError::InvalidInput` if `in_features` or `out_features` is 0,
    /// or if `sparsity` is not in [0.0, 1.0].
    ///
    /// # Note
    /// This creates dense storage with sparse initialization patterns.
    /// True sparse weight matrices require specialized storage types.
    pub fn new_with_sparse_init(
        in_features: usize,
        out_features: usize,
        sparsity: f64,
    ) -> Result<Self>
    where
        B: Default,
        S: StorageFromVec<T>,
    {
        Self::new_with_sparse_init_backend(B::default(), in_features, out_features, sparsity)
    }

    /// Create a new linear layer with sparse weight initialization and custom backend.
    ///
    /// # Arguments
    /// * `backend` - The compute backend
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `sparsity` - Target sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    ///
    /// # Errors
    /// Returns `NNError::InvalidInput` if `in_features` or `out_features` is 0,
    /// or if `sparsity` is not in [0.0, 1.0].
    pub fn new_with_sparse_init_backend(
        backend: B,
        in_features: usize,
        out_features: usize,
        sparsity: f64,
    ) -> Result<Self>
    where
        S: StorageFromVec<T>,
    {
        if in_features == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "in_features must be > 0".to_string(),
            });
        }
        if out_features == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "out_features must be > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(crate::error::NNError::InvalidInput {
                message: format!("sparsity must be in [0.0, 1.0], got {}", sparsity),
            });
        }

        // Initialize weights with sparse pattern (dense storage)
        let weight_data =
            Self::sparse_weight_init_dense(&backend, in_features, out_features, sparsity)?;
        let bias_storage = S::zeros(&[out_features])?;
        let bias_data = Tensor::<B, S, T>::from_storage(bias_storage, backend.clone());

        let weight = Parameter::new(
            weight_data.clone().requires_grad_(true),
            "weight".to_string(),
        );
        let bias = Parameter::new(bias_data.requires_grad_(true), "bias".to_string());

        // Cache transposed weight for efficient forward passes
        let weight_t = Some(weight_data.to_dense_generic()?.transpose(1, 0)?);

        Ok(Self {
            weight,
            bias,
            weight_t,
            in_features,
            out_features,
        })
    }

    /// Xavier/Glorot uniform initialization for weights with specific backend.
    ///
    /// Initializes weights using Xavier uniform distribution with proper random sampling.
    /// This ensures symmetry breaking for gradient descent convergence.
    ///
    /// # References
    /// - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
    fn xavier_uniform_init_with_backend(
        backend: &B,
        in_features: usize,
        out_features: usize,
        _limit: T,
    ) -> Result<Tensor<B, S, T>> {
        use rand::prelude::*;

        let shape = [out_features, in_features];
        let total_elements = out_features * in_features;

        // Calculate Xavier uniform limit: sqrt(6 / (fan_in + fan_out))
        let fan_in = T::from(in_features).unwrap();
        let fan_out = T::from(out_features).unwrap();
        let limit = (T::from(6.0).unwrap() / (fan_in + fan_out)).sqrt();
        let limit_f64 = limit.to_f64().unwrap();

        // Generate random values uniformly distributed in [-limit, limit]
        let mut rng = rand::thread_rng();
        let weight_data: Vec<T> = (0..total_elements)
            .map(|_| {
                let rand_val: f64 = rng.gen_range(-limit_f64..=limit_f64);
                T::from_f64(rand_val).unwrap_or(T::zero())
            })
            .collect();

        let storage = S::from_vec(weight_data, &shape)?;
        Ok(Tensor::<B, S, T>::from_storage(storage, backend.clone()))
    }

    /// Sparse weight initialization for dense storage.
    ///
    /// Creates a weight matrix with controlled sparsity pattern using dense storage.
    /// Uses random connectivity with Xavier uniform distribution for non-zero elements.
    ///
    /// # Arguments
    /// * `backend` - The compute backend
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `sparsity` - Target sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    ///
    /// # Returns
    /// Dense tensor with sparse initialization pattern
    fn sparse_weight_init_dense(
        backend: &B,
        in_features: usize,
        out_features: usize,
        sparsity: f64,
    ) -> Result<Tensor<B, S, T>>
    where
        S: StorageFromVec<T>,
    {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let total_elements = in_features * out_features;
        let target_nnz = ((1.0 - sparsity) * total_elements as f64).round() as usize;
        let target_nnz = target_nnz.max(1).min(total_elements); // At least 1, at most total

        // Create dense matrix initialized to zero
        let mut weight_data = vec![T::zero(); total_elements];

        // Xavier initialization limit
        let limit = (T::from(6.0).unwrap() / T::from(in_features + out_features).unwrap()).sqrt();
        let limit_f64 = limit.to_f64().unwrap();

        // Randomly place non-zero weights
        let mut placed = 0;
        while placed < target_nnz {
            let row = rng.gen_range(0..out_features);
            let col = rng.gen_range(0..in_features);
            let idx = row * in_features + col;

            // Only place if not already set (avoid duplicates)
            if weight_data[idx] == T::zero() {
                let weight_val = rng.gen_range(-limit_f64..=limit_f64);
                weight_data[idx] = T::from(weight_val).unwrap();
                placed += 1;
            }
        }

        let storage = S::from_vec(weight_data, &[out_features, in_features])?;
        Ok(Tensor::<B, S, T>::from_storage(storage, backend.clone()))
    }
}

// Future enhancement: Implement AutoGradTensor-based forward method with proper grad_fn setup
// The Variable-based API is deprecated in favor of tensor-based operations
// with AutoGradTensor for gradient computation

impl<B, S, T> Module<B, S, T> for Linear<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::One,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Linear transformation: output = input @ weight.T + bias
        // Shapes: input [..., in_features], weight [out_features, in_features], bias [out_features]

        let input_shape = input.shape();
        let weight_shape = self.weight.data().shape();
        let bias_shape = self.bias.data().shape();

        // Validate dimensions
        let weight_dims = weight_shape.dims();
        let in_features = *weight_dims.last().unwrap();
        let out_features = weight_dims[weight_dims.len() - 2];

        if input_shape.dims().last() != Some(&in_features) {
            return Err(crate::error::NNError::InvalidInput {
                message: format!(
                    "Input feature dimension {} does not match weight input features {}",
                    input_shape.dims().last().unwrap_or(&0),
                    in_features
                ),
            });
        }

        if bias_shape.dims() != [out_features] {
            return Err(crate::error::NNError::InvalidInput {
                message: format!(
                    "Bias shape {:?} does not match expected shape [{:?}]",
                    bias_shape.dims(),
                    &[out_features]
                ),
            });
        }

        // Use dense computation path for now
        // Future enhancement: Implement compile-time sparse weight detection and sparse operations
        // This would require separate trait implementations for sparse storage types
        let input_dense = input.to_dense_generic()?;
        let weight_t = self.weight_t.as_ref().unwrap();
        let output = input_dense.matmul(weight_t)?;

        // Add bias manually - add bias to each sample in the batch
        let bias_data = self.bias.data().as_slice();
        let mut output_data = output.as_slice().to_vec();
        let batch_size = input.shape().dims()[0];
        let out_features = bias_data.len(); // Use bias length as out_features

        for batch in 0..batch_size {
            for feature in 0..out_features {
                let idx = batch * out_features + feature;
                if idx < output_data.len() && feature < bias_data.len() {
                    output_data[idx] = output_data[idx] + bias_data[feature];
                }
            }
        }

        let result = Tensor::<B, S, T>::from_vec(output_data, output.shape().dims())?;
        Ok(result)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    fn train(&mut self, _mode: bool) {
        // No-op for now, could be used for dropout, batch norm, etc.
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

impl<B, S, T> fmt::Display for Linear<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Linear(in_features={}, out_features={}, bias={})",
            self.in_features,
            self.out_features,
            !self.bias.data().as_slice().is_empty()
        )
    }
}

#[cfg(feature = "safetensors")]
impl<B, S, T> crate::module::ModuleSerialize<B, S, T> for Linear<B, S, T>
where
    B: Backend + Clone + std::default::Default,
    S: Storage<T>
        + Clone
        + 'static
        + coeus_storage::StorageFromVec<T>
        + coeus_storage::StorageToDense<T>,
    T: DataType
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>
        + coeus_dtype::traits::FloatExt,
{
    // Default implementation from the trait is sufficient
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    type TestParameter = Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

    #[test]
    fn test_parameter_creation() {
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[5])
            .unwrap()
            .requires_grad_(true);
        let param = TestParameter::new(data, "test_param".to_string());

        assert_eq!(param.name(), "test_param");
        assert!(param.requires_grad());
        assert_eq!(param.data().shape().dims(), &[5]);
    }

    #[test]
    fn test_parameter_creation_no_grad() {
        let data =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        let param = TestParameter::new(data, "test_param".to_string());

        assert_eq!(param.name(), "test_param");
        assert!(!param.requires_grad());
        assert_eq!(param.data().shape().dims(), &[5]);
    }

    #[test]
    fn test_parameter_zero_grad() {
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3])
            .unwrap()
            .requires_grad_(true);
        let mut param = TestParameter::new(data, "test".to_string());

        // Initially should require gradients
        assert!(param.requires_grad());

        // Zero grad detaches the tensor
        param.zero_grad();
        assert!(!param.requires_grad());
    }

    #[test]
    fn test_parameter_update_data() {
        let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3])
            .unwrap()
            .requires_grad_(true);
        let mut param = TestParameter::new(data, "test".to_string());

        let new_data =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[3]).unwrap();
        param.update_data(new_data);

        // Should still require gradients
        assert!(param.requires_grad());
        // Should have zero values now
        assert_eq!(param.data().as_slice()[0].get(), 0.0);
    }

    #[test]
    fn test_sparse_weight_initialization() {
        let layer =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_sparse_init(
                10, 5, 0.8,
            )
            .unwrap();

        // Check dimensions
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);

        // Check weight matrix shape
        let weight_shape = layer.weight.data().shape().dims();
        assert_eq!(weight_shape, &[5, 10]); // [out_features, in_features]

        // Check bias shape
        let bias_shape = layer.bias.data().shape().dims();
        assert_eq!(bias_shape, &[5]);

        // Count non-zero elements in weight matrix (should be around 20% of 50 elements)
        let weight_data = layer.weight.data().as_slice();
        let non_zero_count = weight_data.iter().filter(|&&x| x.get() != 0.0).count();
        let total_elements = weight_data.len();

        // With 80% sparsity, we expect around 20% non-zero elements
        // Allow some tolerance for random initialization
        let expected_nnz = (total_elements as f64 * 0.2).round() as usize;
        assert!(
            (non_zero_count as isize - expected_nnz as isize).abs() <= 5,
            "Expected ~{} non-zero elements, got {}",
            expected_nnz,
            non_zero_count
        );
    }
}
