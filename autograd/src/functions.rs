//! Function objects for automatic differentiation
//!
//! This module defines the backward functions that implement gradient computation
//! for various tensor operations in the computation graph.

extern crate alloc;

use alloc::vec::Vec;
use alloc::sync::Arc;
use backend::Backend;
use dtype::DataType;
use dtype::traits::FloatExt;
use storage::{Storage, StorageFromVec, StorageToDense, DenseStorage};
pub use tensor::{Tensor, DifferentiableFunction, Function};

/// Type alias for tensor references used in automatic differentiation
/// Generic over `Backend<B>`, `Storage<S>`, and `DataType<T>`
pub type TensorRef<B, S, T> = alloc::sync::Arc<tensor::Tensor<B, S, T>>;

/// Type-erased function reference for tensor `grad_fn` fields
/// Uses the existing `DifferentiableFunction` trait for compatibility
pub type FunctionRef<B, S, T> = alloc::sync::Arc<dyn DifferentiableFunction<B, S, T>>;

// AsAny implementations for all Function structs
impl<B, S, T> tensor::AsAny for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for SumFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for ExpFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for LogFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for SinFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for CosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> tensor::AsAny for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Sub function for element-wise subtraction
#[derive(Debug)]
pub struct SubFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [lhs, rhs]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> tensor::AsAny for SubFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> SubFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new SubFunction with the given inputs
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}


/// Add function for element-wise addition
#[derive(Debug)]
pub struct AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [lhs, rhs]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Add function
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For addition: d/da = 1, d/db = 1, so gradients are just grad_output
        // Convert grad_output to storage type S
        let grad_data = grad_output.storage_ref().as_slice().to_vec();
        let grad_dims = grad_output.shape().dims().to_vec();
        let grad_tensor = Tensor::from_vec_with_backend(grad_data, &grad_dims, grad_output.backend().clone())?;

        Ok(vec![grad_tensor.clone(), grad_tensor])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// MatMul function for matrix multiplication
#[derive(Debug)]
pub struct MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [lhs, rhs]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new MatMul function
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for MatMulFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + 'static,
    T: DataType + Clone,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        // Convert to dense for matrix operations
        let lhs_dense = lhs.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let rhs_dense = rhs.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let grad_output_dense = grad_output.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        // For C = A @ B:
        // dC/dA = grad_output @ B^T
        // dC/dB = A^T @ grad_output
        let rhs_t = rhs_dense.transpose(0, 1).map_err(|e| anyhow::anyhow!("Transpose error: {:?}", e))?;
        let lhs_t = lhs_dense.transpose(0, 1).map_err(|e| anyhow::anyhow!("Transpose error: {:?}", e))?;

        let grad_lhs_dense = grad_output_dense.matmul(&rhs_t).map_err(|e| anyhow::anyhow!("Matmul error: {:?}", e))?;
        let grad_rhs_dense = lhs_t.matmul(&grad_output_dense).map_err(|e| anyhow::anyhow!("Matmul error: {:?}", e))?;

        // Convert back to storage type S
        let grad_lhs_data = grad_lhs_dense.storage_ref().as_slice().to_vec();
        let grad_lhs_dims = grad_lhs_dense.shape().dims().to_vec();
        let grad_lhs = Tensor::from_vec_with_backend(grad_lhs_data, &grad_lhs_dims, grad_output.backend().clone())?;

        let grad_rhs_data = grad_rhs_dense.storage_ref().as_slice().to_vec();
        let grad_rhs_dims = grad_rhs_dense.shape().dims().to_vec();
        let grad_rhs = Tensor::from_vec_with_backend(grad_rhs_data, &grad_rhs_dims, grad_output.backend().clone())?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MatMulFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + 'static,
    T: DataType + Clone,
{
    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
}

/// Mul function for element-wise multiplication
#[derive(Debug)]
pub struct MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [lhs, rhs]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Mul function
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        // Convert inputs to dense for gradient computation
        let lhs_dense = lhs.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let rhs_dense = rhs.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        // For C = A * B (element-wise):
        // dC/dA = B, dC/dB = A
        // So gradients are: grad_output * B, grad_output * A
        let grad_lhs_dense = grad_output * &rhs_dense;
        let grad_rhs_dense = grad_output * &lhs_dense;

        // Convert back to storage type S
        let grad_lhs_data = grad_lhs_dense.storage_ref().as_slice().to_vec();
        let grad_lhs_dims = grad_lhs_dense.shape().dims().to_vec();
        let grad_lhs = Tensor::from_vec_with_backend(grad_lhs_data, &grad_lhs_dims, grad_output.backend().clone())?;

        let grad_rhs_data = grad_rhs_dense.storage_ref().as_slice().to_vec();
        let grad_rhs_dims = grad_rhs_dense.shape().dims().to_vec();
        let grad_rhs = Tensor::from_vec_with_backend(grad_rhs_data, &grad_rhs_dims, grad_output.backend().clone())?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Sum function for summation operations
#[derive(Debug)]
pub struct SumFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> SumFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Sum function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for SumFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For sum operations, gradient w.r.t. input is grad_output broadcasted to input shape
        let input = &*self.inputs[0];

        // Convert input to dense for operations
        let input_dense = input.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let grad_input_dense: Tensor<B, DenseStorage<T>, T> = Tensor::ones(input_dense.shape().dims()).map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let grad_input_dense = &grad_input_dense * grad_output;

        // Convert result back to input storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input = Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
            .map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for SumFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "SumBackward"
    }
}



/// Mean function for averaging operations
#[derive(Debug)]
pub struct MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Mean function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For mean operations, gradient w.r.t. input is grad_output / num_elements, broadcasted to input shape
        let input = &*self.inputs[0];
        let num_elements = input.len() as f64;
        let scale_factor = 1.0 / num_elements;

        // Convert input to dense for operations
        let input_dense = input.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let mut grad_input_dense: Tensor<B, DenseStorage<T>, T> = Tensor::ones(input_dense.shape().dims()).map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let scale_tensor: Tensor<B, DenseStorage<T>, T> = Tensor::from_vec(vec![T::from(scale_factor).unwrap()], &[]).map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        grad_input_dense = &grad_input_dense * &scale_tensor;
        grad_input_dense = &grad_input_dense * grad_output;

        // Convert result back to input storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input = Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
            .map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

/// Exp function for exponential operations
#[derive(Debug)]
pub struct ExpFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> ExpFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Exp function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for ExpFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For exp operations: d/dx exp(x) = exp(x)
        // So gradient w.r.t. input is grad_output * exp(input)
        let input = &*self.inputs[0];
        let exp_input = input.exp();

        // Convert exp_input to dense for multiplication with grad_output
        let exp_input_dense = exp_input.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let grad_input_dense = grad_output * &exp_input_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input = Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
            .map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for ExpFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Log function for natural logarithm operations
#[derive(Debug)]
pub struct LogFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> LogFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Log function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for LogFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For log operations: d/dx log(x) = 1/x
        // So gradient w.r.t. input is grad_output / input
        let input = &*self.inputs[0];

        // Convert input to dense for division with grad_output
        let input_dense = input.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let grad_input_dense = grad_output / &input_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input = Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
            .map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for LogFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Sin function for sine operations
#[derive(Debug)]
pub struct SinFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> SinFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Sin function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for SinFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For sin operations: d/dx sin(x) = cos(x)
        // So gradient w.r.t. input is grad_output * cos(input)
        let input = &*self.inputs[0];
        let cos_input = input.cos();

        // Convert cos_input to dense for multiplication with grad_output
        let cos_input_dense = cos_input.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let grad_input_dense = grad_output * &cos_input_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input = Tensor::from_vec_with_backend(grad_data, &grad_dims, grad_output.backend().clone())?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for SinFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

/// Cos function for cosine operations
#[derive(Debug)]
pub struct CosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> CosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Cos function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for CosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For cos operations: d/dx cos(x) = -sin(x)
        // So gradient w.r.t. input is grad_output * (-sin(input))
        let input = &*self.inputs[0];
        let sin_input = input.sin();

        // Convert sin_input to dense and negate it
        let sin_input_dense = sin_input.to_dense_generic().map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let neg_one: Tensor<B, DenseStorage<T>, T> = Tensor::from_vec(vec![T::from(-1.0).unwrap()], &[]).map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let neg_sin_dense = &sin_input_dense * &neg_one;

        let grad_input_dense = grad_output * &neg_sin_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input = Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
            .map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for CosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "CosBackward"
    }
}

/// NLLLoss function for negative log likelihood loss
#[derive(Debug)]
pub struct NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [log_probs, targets]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new NLLLoss function
    #[must_use]
    pub fn new(log_probs: Arc<Tensor<B, S, T>>, targets: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![log_probs, targets],
        }
    }
}

impl<B, S, T> Function<B, S, T> for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // Simplified NLL loss backward - this would need proper implementation
        // For now, return zero gradients to avoid panics
        let log_probs = &*self.inputs[0];
        let targets = &*self.inputs[1];

        let zero_grad_log_probs = Tensor::zeros(log_probs.shape().dims()).map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;
        let zero_grad_targets = Tensor::zeros(targets.shape().dims()).map_err(|e| anyhow::anyhow!("Tensor error: {:?}", e))?;

        Ok(vec![zero_grad_log_probs, zero_grad_targets])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "NLLLossBackward"
    }
}

// RNN function for sequence processing
/// RNN function for automatic differentiation of recurrent operations
#[derive(Debug)]
pub struct RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors that require gradients
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    /// Hidden state tensors
    pub hidden_states: Vec<Arc<Tensor<B, S, T>>>,
    /// Whether batch dimension is first
    pub batch_first: bool,
}

impl<B, S, T> RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new RNN function
    pub fn new(
        inputs: Vec<Arc<Tensor<B, S, T>>>,
        hidden_states: Vec<Arc<Tensor<B, S, T>>>,
        batch_first: bool,
    ) -> Self {
        Self {
            inputs,
            hidden_states,
            batch_first,
        }
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "RNN"
    }
}

impl<B, S, T> tensor::Function<B, S, T> for RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // RNN backward pass - simplified implementation
        // In a real implementation, this would compute gradients for all inputs
        // For now, return zero gradients for all inputs
        let mut gradients = Vec::new();
        for input in &self.inputs {
            let zero_grad = Tensor::<B, S, T>::zeros(input.shape().dims())?;
            gradients.push(zero_grad);
        }
        Ok(gradients)
    }
}

impl<B, S, T> tensor::AsAny for RNNFunction<B, S, T>
where
    B: 'static,
    S: 'static,
    T: 'static,
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
