//! Parameter management for neural network modules.

use std::fmt;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use backend::Backend;
use dtype::DataType;
use storage::{CooStorage, CscStorage, CsrStorage, Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic, Tensor};

/// Trait for parameter-like objects that can be used in modules.
pub trait ParameterTrait {
    /// Get the parameter name
    fn name(&self) -> &str;
}

/// A learnable parameter in a neural network module.
///
/// Parameters store tensor data and automatically handle gradient computation
/// through the new tensor-based autograd system. Clean separation between
/// data (Tensor) and computation (AutoGradTensor) enables zero-cost inference.
#[derive(Debug, Clone)]
pub struct Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    /// The parameter data tensor - may require gradients
    pub(crate) data: Tensor<B, S, T>,
    /// Human-readable name for debugging
    pub(crate) name: String,
}

impl<B, S, T> Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    /// Create a new parameter from tensor data.
    ///
    /// # Arguments
    /// * `data` - The parameter tensor data
    /// * `requires_grad` - Whether gradients should be computed for this parameter
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Examples
    /// ```rust
    /// use nn::Parameter;
    /// use tensor::TensorCpuDense;
    /// use dtype::float::Float32;
    ///
    /// let data = TensorCpuDense::<Float32>::zeros(&[10, 5]).unwrap()
    ///     .requires_grad_(true);
    /// let param = Parameter::new(data, "weight".to_string());
    /// assert!(param.requires_grad());
    /// ```
    #[must_use]
    pub fn new(data: Tensor<B, S, T>, name: String) -> Self {
        Self { data, name }
    }

    /// Transpose the parameter tensor.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to transpose
    /// * `dim1` - Second dimension to transpose
    ///
    /// # Returns
    /// A new tensor with dimensions transposed
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor<B, S, T>>
    where
        B: Clone + Default + 'static,
        S: tensor::ops::dispatch::TensorStorageOps<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
        T: DataType + Clone + std::ops::Neg<Output = T>,
    {
        Ok(self.data.transpose(dim0, dim1)?)
    }
}

impl<B, S, T> Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + std::ops::Neg<Output = T>,
{
    /// Update parameter data using gradient descent.
    ///
    /// This performs: `parameter = parameter - learning_rate * gradient`
    ///
    /// # Arguments
    /// * `gradient` - The gradient tensor with the same shape as the parameter
    /// * `learning_rate` - The learning rate for the update
    ///
    /// # Errors
    /// Returns an error if the gradient shape doesn't match the parameter shape
    pub fn update_with_gradient(
        &mut self,
        gradient: &Tensor<B, S, T>,
        learning_rate: f64,
    ) -> Result<()> {
        if self.data.shape() != gradient.shape() {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Gradient shape {:?} does not match parameter shape {:?}",
                    gradient.shape().dims(),
                    self.data.shape().dims()
                ),
            });
        }

        // Convert learning rate to parameter type
        let lr = T::from(learning_rate).unwrap_or(T::from(learning_rate).unwrap_or(T::zero()));

        // Create learning rate tensor for broadcasting
        let lr_tensor = Tensor::<B, S, T>::from_vec(vec![lr], &[1])?;

        // Compute update: lr * gradient
        let update = arithmetic::mul(gradient, &lr_tensor)?;

        // Update parameter: parameter -= update
        self.data = arithmetic::sub(&self.data, &update)?;

        Ok(())
    }
}

impl<B, S, T> ParameterTrait for Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    fn name(&self) -> &str {
        &self.name
    }
}

impl<B, S, T> Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    /// Get a reference to the parameter tensor data.
    ///
    /// # Returns
    /// Reference to the parameter tensor
    pub fn tensor(&self) -> &Tensor<B, S, T> {
        &self.data
    }

    /// Get a mutable reference to the parameter tensor data.
    ///
    /// This is used by optimizers and other components that need to modify
    /// parameter values directly.
    ///
    /// # Returns
    /// Mutable reference to the parameter tensor
    pub fn data_mut(&mut self) -> &mut Tensor<B, S, T> {
        &mut self.data
    }

    /// Create a new sparse CSR parameter from dense initialization with sparsity pattern.
    ///
    /// This creates a sparse parameter by initializing with a dense pattern and then
    /// converting to CSR format, maintaining the sparsity structure.
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `dims` - The tensor dimensions [rows, cols]
    /// * `sparsity` - The sparsity ratio (0.0 = fully dense, 1.0 = fully sparse)
    /// * `requires_grad` - Whether gradients should be computed
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Returns
    /// A new sparse parameter in CSR format
    pub fn new_sparse_csr(
        backend: B,
        dims: &[usize],
        sparsity: f64,
        requires_grad: bool,
        name: String,
    ) -> std::result::Result<Parameter<B, CsrStorage<T>, T>, NNError>
    where
        S: StorageFromVec<T>,
        T: DataType
            + dtype::traits::FloatExt
            + dtype::num_traits::Zero
            + std::ops::Mul<Output = T>
            + Copy
            + num_traits::FromPrimitive,
    {
        if dims.len() != 2usize {
            return Err(NNError::InvalidInput {
                message: "Sparse parameters must be 2D tensors".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(NNError::InvalidInput {
                message: "Sparsity must be in [0.0, 1.0]".to_string(),
            });
        }

        // Create dense initialization with sparse pattern
        let dense_data = Self::create_sparse_dense_init(&backend, dims, sparsity)?;

        // Convert to CSR sparse format by creating sparse storage directly
        let csr_storage =
            CsrStorage::<T>::from_vec(dense_data.as_slice().to_vec(), dense_data.shape().dims())?;
        let csr_tensor = Tensor::<B, CsrStorage<T>, T>::from_storage(csr_storage, backend);

        let data = if requires_grad {
            csr_tensor.requires_grad_(true)
        } else {
            csr_tensor
        };

        Ok(Parameter::new(data, name))
    }

    /// Create a new sparse CSC parameter from dense initialization with sparsity pattern.
    ///
    /// This creates a sparse parameter by initializing with a dense pattern and then
    /// converting to CSC format, maintaining the sparsity structure.
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `dims` - The tensor dimensions [rows, cols]
    /// * `sparsity` - The sparsity ratio (0.0 = fully dense, 1.0 = fully sparse)
    /// * `requires_grad` - Whether gradients should be computed
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Returns
    /// A new sparse parameter in CSC format
    pub fn new_sparse_csc(
        backend: B,
        dims: &[usize],
        sparsity: f64,
        requires_grad: bool,
        name: String,
    ) -> std::result::Result<Parameter<B, CscStorage<T>, T>, NNError>
    where
        S: StorageFromVec<T>,
        T: DataType
            + dtype::traits::FloatExt
            + dtype::num_traits::Zero
            + std::ops::Mul<Output = T>
            + Copy
            + num_traits::FromPrimitive,
    {
        if dims.len() != 2usize {
            return Err(NNError::InvalidInput {
                message: "Sparse parameters must be 2D tensors".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(NNError::InvalidInput {
                message: "Sparsity must be in [0.0, 1.0]".to_string(),
            });
        }

        // Create dense initialization with sparse pattern
        let dense_data = Self::create_sparse_dense_init(&backend, dims, sparsity)?;

        // Convert to CSC sparse format by creating sparse storage directly
        let csc_storage =
            CscStorage::<T>::from_vec(dense_data.as_slice().to_vec(), dense_data.shape().dims())?;
        let csc_tensor = Tensor::<B, CscStorage<T>, T>::from_storage(csc_storage, backend);

        let data = if requires_grad {
            csc_tensor.requires_grad_(true)
        } else {
            csc_tensor
        };

        Ok(Parameter::new(data, name))
    }

    /// Create a new sparse COO parameter from dense initialization with sparsity pattern.
    ///
    /// This creates a sparse parameter by initializing with a dense pattern and then
    /// converting to COO format, maintaining the sparsity structure.
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `dims` - The tensor dimensions [rows, cols]
    /// * `sparsity` - The sparsity ratio (0.0 = fully dense, 1.0 = fully sparse)
    /// * `requires_grad` - Whether gradients should be computed
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Returns
    /// A new sparse parameter in COO format
    pub fn new_sparse_coo(
        backend: B,
        dims: &[usize],
        sparsity: f64,
        requires_grad: bool,
        name: String,
    ) -> std::result::Result<Parameter<B, CooStorage<T>, T>, NNError>
    where
        S: StorageFromVec<T>,
        T: DataType
            + dtype::traits::FloatExt
            + dtype::num_traits::Zero
            + std::ops::Mul<Output = T>
            + Copy
            + num_traits::FromPrimitive,
    {
        if dims.len() != 2usize {
            return Err(NNError::InvalidInput {
                message: "Sparse parameters must be 2D tensors".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(NNError::InvalidInput {
                message: "Sparsity must be in [0.0, 1.0]".to_string(),
            });
        }

        // Create dense initialization with sparse pattern
        let dense_data = Self::create_sparse_dense_init(&backend, dims, sparsity)?;

        // Convert to COO sparse format by creating sparse storage directly
        let coo_storage =
            CooStorage::<T>::from_vec(dense_data.as_slice().to_vec(), dense_data.shape().dims())?;
        let coo_tensor = Tensor::<B, CooStorage<T>, T>::from_storage(coo_storage, backend);

        let data = if requires_grad {
            coo_tensor.requires_grad_(true)
        } else {
            coo_tensor
        };

        Ok(Parameter::new(data, name))
    }

    /// Create an empty sparse CSR parameter.
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `dims` - The tensor dimensions [rows, cols]
    /// * `requires_grad` - Whether gradients should be computed
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Returns
    /// A new empty sparse parameter in CSR format
    pub fn new_empty_sparse_csr(
        backend: B,
        dims: &[usize],
        requires_grad: bool,
        name: String,
    ) -> std::result::Result<Parameter<B, CsrStorage<T>, T>, NNError> {
        // Create empty CSR storage with proper indptr initialization
        // indptr needs rows + 1 elements, all set to 0 for empty matrix
        let rows = dims[0];
        let indptr = vec![0; rows + 1];
        let storage = CsrStorage::new(vec![], vec![], indptr, dims)
            .map_err(|e| NNError::StorageError { source: e })?;

        let mut tensor = Tensor::from_storage(storage, backend);
        if requires_grad {
            tensor = tensor.requires_grad_(true);
        }

        Ok(Parameter { data: tensor, name })
    }

    /// Create an empty sparse CSC parameter.
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `dims` - The tensor dimensions [rows, cols]
    /// * `requires_grad` - Whether gradients should be computed
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Returns
    /// A new empty sparse parameter in CSC format
    pub fn new_empty_sparse_csc(
        backend: B,
        dims: &[usize],
        requires_grad: bool,
        name: String,
    ) -> std::result::Result<Parameter<B, CscStorage<T>, T>, NNError> {
        // Create empty CSC storage with proper indptr initialization
        // indptr needs cols + 1 elements, all set to 0 for empty matrix
        let cols = dims[1];
        let indptr = vec![0; cols + 1];
        let storage = CscStorage::new(vec![], vec![], indptr, dims)
            .map_err(|e| NNError::StorageError { source: e })?;

        let mut tensor = Tensor::from_storage(storage, backend);
        if requires_grad {
            tensor = tensor.requires_grad_(true);
        }

        Ok(Parameter { data: tensor, name })
    }

    /// Create an empty sparse COO parameter.
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `dims` - The tensor dimensions [rows, cols]
    /// * `requires_grad` - Whether gradients should be computed
    /// * `name` - Human-readable name for the parameter
    ///
    /// # Returns
    /// A new empty sparse parameter in COO format
    pub fn new_empty_sparse_coo(
        backend: B,
        dims: &[usize],
        requires_grad: bool,
        name: String,
    ) -> std::result::Result<Parameter<B, CooStorage<T>, T>, NNError> {
        // Create empty COO storage with empty vectors
        let storage = CooStorage::new(vec![], vec![], vec![], dims)
            .map_err(|e| NNError::StorageError { source: e })?;

        let mut tensor = Tensor::from_storage(storage, backend);
        if requires_grad {
            tensor = tensor.requires_grad_(true);
        }

        Ok(Parameter { data: tensor, name })
    }

    /// Get the parameter data tensor (immutable).
    ///
    /// # Returns
    /// A reference to the underlying tensor data.
    #[must_use]
    pub const fn data(&self) -> &Tensor<B, S, T> {
        &self.data
    }

    /// Check if this parameter requires gradient computation.
    ///
    /// # Returns
    /// `true` if gradients should be computed for this parameter.
    #[must_use]
    pub const fn requires_grad(&self) -> bool {
        self.data.requires_grad()
    }

    /// Get the parameter name.
    ///
    /// # Returns
    /// The human-readable name of this parameter.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Convert this parameter to an AutoGradTensor for computation.
    ///
    /// # Panics
    ///
    /// Zero the gradients for this parameter by detaching.
    ///
    /// This creates a new tensor without gradient requirements,
    /// effectively resetting gradient computation.
    pub fn zero_grad(&mut self) {
        self.data = self.data.clone().detach();
    }

    /// Update parameter data in-place (for optimizers).
    ///
    /// This method allows optimizers to update parameter values directly.
    /// The parameter's gradient requirements are preserved.
    ///
    /// # Arguments
    /// * `new_data` - The new parameter data tensor
    pub fn update_data(&mut self, new_data: Tensor<B, S, T>) {
        let requires_grad = self.data.requires_grad();
        self.data = new_data.requires_grad_(requires_grad);
    }

    /// Helper function to create dense tensor with sparse initialization pattern.
    ///
    /// This creates a dense tensor but initializes it with a sparse pattern,
    /// which can then be converted to true sparse storage formats.
    fn create_sparse_dense_init(
        backend: &B,
        dims: &[usize],
        sparsity: f64,
    ) -> std::result::Result<Tensor<B, S, T>, NNError>
    where
        S: StorageFromVec<T>,
        T: DataType
            + dtype::traits::FloatExt
            + dtype::num_traits::Zero
            + std::ops::Mul<Output = T>
            + Copy
            + num_traits::FromPrimitive,
    {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let rows = dims[0];
        let cols = dims[1];
        let total_elements = rows * cols;
        let target_nnz = ((1.0 - sparsity) * total_elements as f64).round() as usize;
        let target_nnz = target_nnz.max(1).min(total_elements); // At least 1, at most total

        // Create dense matrix initialized to zero
        let mut weight_data = vec![T::zero(); total_elements];

        // Xavier initialization limit
        let limit = (T::from(6.0).unwrap() / T::from(rows + cols).unwrap()).sqrt();
        let limit_f64 = limit.to_f64().unwrap();

        // Randomly place non-zero weights
        let mut placed = 0;
        while placed < target_nnz {
            let row = rng.gen_range(0..rows);
            let col = rng.gen_range(0..cols);
            let idx = row * cols + col;

            // Only place if not already set (avoid duplicates)
            if weight_data[idx] == T::zero() {
                // Generate random value in [-limit, limit] using FloatExt
                let rand_val: f64 = if limit_f64 <= f64::EPSILON {
                    0.0
                } else {
                    rng.gen_range(-limit_f64..=limit_f64)
                };
                weight_data[idx] = T::from_f64(rand_val).unwrap_or(T::zero());
                placed += 1;
            }
        }

        let storage = S::from_vec(weight_data, dims)?;
        Ok(Tensor::<B, S, T>::from_storage(storage, backend.clone()))
    }
}

impl<B, S, T> fmt::Display for Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Parameter(name={}, shape={:?}, requires_grad={})",
            self.name,
            self.data().shape().dims(),
            self.requires_grad()
        )
    }
}

// Future enhancement: Add rkyv zero-copy serialization for parameters

// Module implementation for Parameter
// Parameter acts as a leaf module containing a single tensor
impl<B, S, T> Module<B, S, T> for Parameter<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + dtype::traits::FloatExt,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Parameter doesn't transform input - just return input unchanged
        // This is unusual but allows Parameter to be used as a module
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.clone()]
    }

    fn zero_grad(&mut self) {
        // Handle empty/invalid parameter shapes gracefully to prevent ShapeMismatch errors
        // This handles edge cases with empty sparse parameters that may have malformed shapes
        let shape_dims = self.data.shape().dims();

        // Skip gradient operations for empty or invalid tensors
        if shape_dims.is_empty() || shape_dims.contains(&0) {
            // Empty/invalid parameters - no gradients to zero
            return;
        }

        // Create zero gradient tensor with validated shape
        match Tensor::<B, S, T>::zeros(shape_dims) {
            Ok(zero_grad) => {
                let _ = self.data.set_grad(zero_grad);
            }
            Err(_) => {
                // If gradient tensor creation fails, silently ignore
                // Handles cases where sparse parameter shapes are not supported for gradient operations
            }
        }
    }

    fn train(&mut self, mode: bool) {
        // Parameters don't have training-specific behavior like dropout,
        // but we can set requires_grad based on training mode
        self.data = self.data.clone().requires_grad_(mode);
    }

    fn name(&self) -> &str {
        "Parameter"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

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
    fn test_sparse_csr_parameter_creation() {
        let backend = CpuBackend::<Float32>::new();
        let param =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_sparse_csr(
                backend.clone(),
                &[4, 6],
                0.5,  // 50% sparsity
                true, // requires_grad
                "sparse_csr_weight".to_string(),
            )
            .unwrap();

        assert_eq!(param.name(), "sparse_csr_weight");
        assert!(param.requires_grad());
        assert_eq!(param.data().shape().dims(), &[4, 6]);

        // Check that the parameter is actually sparse (has some zeros in dense representation)
        let dense_tensor = param.data().to_dense_generic().unwrap();
        let data = dense_tensor.as_slice();
        let zero_count = data.iter().filter(|&&x| x == Float32(0.0)).count();
        assert!(
            zero_count > 0,
            "Sparse parameter should contain zeros in dense representation"
        );
    }

    #[test]
    fn test_sparse_csc_parameter_creation() {
        let backend = CpuBackend::<Float32>::new();
        let param =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_sparse_csc(
                backend.clone(),
                &[3, 5],
                0.7,   // 70% sparsity
                false, // no gradients
                "sparse_csc_weight".to_string(),
            )
            .unwrap();

        assert_eq!(param.name(), "sparse_csc_weight");
        assert!(!param.requires_grad());
        assert_eq!(param.data().shape().dims(), &[3, 5]);
    }

    #[test]
    fn test_sparse_coo_parameter_creation() {
        let backend = CpuBackend::<Float32>::new();
        let param =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_sparse_coo(
                backend.clone(),
                &[2, 4],
                0.8, // 80% sparsity
                true,
                "sparse_coo_weight".to_string(),
            )
            .unwrap();

        assert_eq!(param.name(), "sparse_coo_weight");
        assert!(param.requires_grad());
        assert_eq!(param.data().shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_empty_sparse_parameters() {
        let backend = CpuBackend::<Float32>::new();

        // Test empty CSR
        let csr_param =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_empty_sparse_csr(
                backend.clone(),
                &[3, 4],
                true,
                "empty_csr".to_string(),
            )
            .unwrap();
        assert_eq!(csr_param.data().shape().dims(), &[3, 4]);

        // Test empty CSC
        let csc_param =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_empty_sparse_csc(
                backend.clone(),
                &[2, 3],
                true,
                "empty_csc".to_string(),
            )
            .unwrap();
        assert_eq!(csc_param.data().shape().dims(), &[2, 3]);

        // Test empty COO
        let coo_param =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_empty_sparse_coo(
                backend.clone(),
                &[4, 5],
                true,
                "empty_coo".to_string(),
            )
            .unwrap();
        assert_eq!(coo_param.data().shape().dims(), &[4, 5]);
    }

    #[test]
    fn test_sparse_parameter_validation() {
        let backend = CpuBackend::<Float32>::new();

        // Test invalid dimensions (1D)
        let result =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_sparse_csr(
                backend.clone(),
                &[10], // 1D tensor
                0.5,
                true,
                "invalid".to_string(),
            );
        assert!(result.is_err());

        // Test invalid sparsity (negative)
        let result =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_sparse_csr(
                backend.clone(),
                &[2, 3],
                -0.1, // negative sparsity
                true,
                "invalid".to_string(),
            );
        assert!(result.is_err());

        // Test invalid sparsity (> 1.0)
        let result =
            Parameter::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_sparse_csr(
                backend.clone(),
                &[2, 3],
                1.5, // sparsity > 1.0
                true,
                "invalid".to_string(),
            );
        assert!(result.is_err());
    }
}
