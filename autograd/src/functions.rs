//! Function objects for automatic differentiation
//!
//! This module defines the backward functions that implement gradient computation
//! for various tensor operations in the computation graph.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use num_traits::FromPrimitive;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
pub use tensor::{DifferentiableFunction, Function, Tensor};

/// Type alias for tensor references used in automatic differentiation
/// Generic over `Backend<B>`, `Storage<S>`, and `DataType<T>`
pub type TensorRef<B, S, T> = alloc::sync::Arc<tensor::Tensor<B, S, T>>;

/// Type-erased function reference for tensor `grad_fn` fields
/// Uses the existing `DifferentiableFunction` trait for compatibility
pub type FunctionRef<B, S, T> = alloc::sync::Arc<dyn DifferentiableFunction<B, S, T>>;

fn canonical_reduce_dims(dim: Option<&[usize]>, input_ndim: usize) -> anyhow::Result<Vec<usize>> {
    let mut dims = if let Some(d) = dim {
        d.to_vec()
    } else {
        (0..input_ndim).collect()
    };

    dims.sort_unstable();
    dims.dedup();

    if dims.iter().any(|&d| d >= input_ndim) {
        return Err(anyhow::anyhow!(
            "reduction dim out of range: dim={dims:?}, input_ndim={input_ndim}"
        ));
    }

    Ok(dims)
}

fn kept_shape_for_reduction(
    input_shape: &[usize],
    reduce_dims: &[usize],
    out_shape: &[usize],
) -> anyhow::Result<Vec<usize>> {
    if input_shape.is_empty() {
        return Ok(Vec::new());
    }

    if reduce_dims.len() == input_shape.len() {
        if out_shape.is_empty() || (out_shape.len() == 1 && out_shape[0] == 1) {
            return Ok(vec![1; input_shape.len()]);
        }
        return Err(anyhow::anyhow!(
            "reduction output shape mismatch: input_shape={input_shape:?}, reduce_dims={reduce_dims:?}, out_shape={out_shape:?}"
        ));
    }

    let mut kept = Vec::with_capacity(input_shape.len());
    let mut out_i = 0usize;
    for axis in 0..input_shape.len() {
        if reduce_dims.binary_search(&axis).is_ok() {
            kept.push(1);
        } else {
            let Some(&d) = out_shape.get(out_i) else {
                return Err(anyhow::anyhow!(
                    "reduction output shape mismatch: input_shape={input_shape:?}, reduce_dims={reduce_dims:?}, out_shape={out_shape:?}"
                ));
            };
            kept.push(d);
            out_i += 1;
        }
    }

    if out_i != out_shape.len() {
        return Err(anyhow::anyhow!(
            "reduction output shape mismatch: input_shape={input_shape:?}, reduce_dims={reduce_dims:?}, out_shape={out_shape:?}"
        ));
    }

    Ok(kept)
}

fn unbroadcast_dense<B, T>(
    grad_output: &Tensor<B, DenseStorage<T>, T>,
    input_shape: &[usize],
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T>,
    T: DataType,
{
    let out_shape = grad_output.shape().dims();

    let out_len = out_shape.len();
    let in_len = input_shape.len();
    let mut input_padded = vec![1; out_len.saturating_sub(in_len)];
    input_padded.extend_from_slice(input_shape);

    let mut reduce_axes = Vec::new();
    for i in 0..out_len {
        let in_dim = input_padded[i];
        let out_dim = out_shape[i];
        if in_dim == 1 && out_dim != 1 {
            reduce_axes.push(i);
        }
    }

    let mut reduced = if reduce_axes.is_empty() {
        grad_output.clone()
    } else {
        grad_output
            .sum_dims(Some(&reduce_axes), true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    };

    if reduced.shape().dims() != input_shape {
        // Use reshape instead of from_vec to preserve the computation graph
        let isize_shape: Vec<isize> = input_shape
            .iter()
            .map(|&x| {
                isize::try_from(x).map_err(|_| {
                    anyhow::anyhow!(
                        "shape dimension {x} exceeds maximum supported dimension {}",
                        isize::MAX
                    )
                })
            })
            .collect::<anyhow::Result<Vec<isize>>>()?;
        reduced = reduced
            .reshape(&isize_shape)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    }

    Ok(reduced)
}

/// Helper to convert a dense tensor to storage S while preserving metadata
fn to_storage_preserving_graph<B, S, T>(
    dense: Tensor<B, DenseStorage<T>, T>,
) -> anyhow::Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + 'static,
    S: Storage<T> + StorageFromVec<T> + 'static,
    T: DataType + Clone + 'static,
{
    dense
        .into_storage_preserving_metadata::<S>()
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

fn to_dense_preserving_graph_identity<B, S, T>(
    tensor: &Tensor<B, S, T>,
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static,
    T: DataType + Clone + 'static,
{
    tensor
        .to_dense_preserving_identity()
        .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))
}

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
    /// Create a new `SubFunction` with the given inputs
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for SubFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let neg_one = T::zero() - T::one();
        let neg_one_tensor: Tensor<B, DenseStorage<T>, T> =
            Tensor::from_vec_with_backend(vec![neg_one], &[], grad_output.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let left_grad_dense = unbroadcast_dense(grad_output, lhs.shape().dims())?;
        let right_grad_dense =
            unbroadcast_dense(&(grad_output * &neg_one_tensor), rhs.shape().dims())?;

        let grad_lhs = Tensor::from_vec_with_backend(
            left_grad_dense.as_slice().to_vec(),
            left_grad_dense.shape().dims(),
            lhs.backend().clone(),
        )?;
        let grad_rhs = Tensor::from_vec_with_backend(
            right_grad_dense.as_slice().to_vec(),
            right_grad_dense.shape().dims(),
            rhs.backend().clone(),
        )?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for SubFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Div function for element-wise division
#[derive(Debug)]
pub struct DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let lhs_dense = lhs
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let rhs_dense = rhs
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let rhs_sq = &rhs_dense * &rhs_dense;
        let neg_one = T::zero() - T::one();
        let neg_one_tensor: Tensor<B, DenseStorage<T>, T> =
            Tensor::from_vec_with_backend(vec![neg_one], &[], grad_output.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let grad_over_rhs_dense = grad_output / &rhs_dense;
        let grad_denominator_dense = {
            let numer = grad_output * &lhs_dense;
            let scaled = &numer / &rhs_sq;
            &scaled * &neg_one_tensor
        };

        let left_grad_dense = unbroadcast_dense(&grad_over_rhs_dense, lhs.shape().dims())?;
        let right_grad_dense = unbroadcast_dense(&grad_denominator_dense, rhs.shape().dims())?;

        let grad_lhs = Tensor::from_vec_with_backend(
            left_grad_dense.as_slice().to_vec(),
            left_grad_dense.shape().dims(),
            lhs.backend().clone(),
        )?;
        let grad_rhs = Tensor::from_vec_with_backend(
            right_grad_dense.as_slice().to_vec(),
            right_grad_dense.shape().dims(),
            rhs.backend().clone(),
        )?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

impl<B, S, T> tensor::AsAny for DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Neg function
#[derive(Debug)]
pub struct NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d(-x)/dx = -1
        let neg_one = -T::one();
        let neg_one_tensor: Tensor<B, DenseStorage<T>, T> =
            Tensor::from_vec_with_backend(vec![neg_one], &[], grad_output.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let grad_input_dense = grad_output * &neg_one_tensor;

        let input = &*self.inputs[0];
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

impl<B, S, T> tensor::AsAny for NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Transpose function
#[derive(Debug)]
pub struct TransposeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub dim0: usize,
    pub dim1: usize,
}

impl<B, S, T> TransposeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, dim0: usize, dim1: usize) -> Self {
        Self {
            inputs: vec![input],
            dim0,
            dim1,
        }
    }
}

impl<B, S, T> Function<B, S, T> for TransposeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // Transpose is its own inverse (if orthogonal, but here it's just swapping dims)
        // d(A^T)/dA -> we need to transpose the gradient back
        let grad_input_dense = grad_output
            .transpose(self.dim0, self.dim1)
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let input = &*self.inputs[0];
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for TransposeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

impl<B, S, T> tensor::AsAny for TransposeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let left_grad_dense = unbroadcast_dense(grad_output, lhs.shape().dims())?;
        let right_grad_dense = unbroadcast_dense(grad_output, rhs.shape().dims())?;

        let grad_lhs = to_storage_preserving_graph(left_grad_dense)?;
        let grad_rhs = to_storage_preserving_graph(right_grad_dense)?;

        Ok(vec![grad_lhs, grad_rhs])
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

/// `MatMul` function for matrix multiplication
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
    /// Create a new `MatMul` function
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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        // Convert to dense for matrix operations
        let lhs_dense = lhs
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let rhs_dense = rhs
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_output_dense = grad_output
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        // For C = A @ B:
        // dC/dA = grad_output @ B^T
        // dC/dB = A^T @ grad_output
        let rhs_t = rhs_dense
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("Transpose error: {e:?}"))?;
        let lhs_t = lhs_dense
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("Transpose error: {e:?}"))?;

        let dense_grads = [
            grad_output_dense
                .matmul(&rhs_t)
                .map_err(|e| anyhow::anyhow!("Matmul error: {e:?}"))?,
            lhs_t
                .matmul(&grad_output_dense)
                .map_err(|e| anyhow::anyhow!("Matmul error: {e:?}"))?,
        ];

        let to_storage = |dense: Tensor<B, DenseStorage<T>, T>| -> anyhow::Result<Tensor<B, S, T>> {
            let data = dense.storage_ref().as_slice().to_vec();
            let dims = dense.shape().dims().to_vec();
            Ok(Tensor::from_vec_with_backend(
                data,
                &dims,
                grad_output.backend().clone(),
            )?)
        };

        let grads = dense_grads
            .into_iter()
            .map(to_storage)
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(grads)
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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let lhs_dense = to_dense_preserving_graph_identity(lhs)?;
        let rhs_dense = to_dense_preserving_graph_identity(rhs)?;

        let left_full_grad_dense = crate::tensor_ops::mul(grad_output, &rhs_dense)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let right_full_grad_dense = crate::tensor_ops::mul(grad_output, &lhs_dense)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let left_grad_dense = unbroadcast_dense(&left_full_grad_dense, lhs.shape().dims())?;
        let right_grad_dense = unbroadcast_dense(&right_full_grad_dense, rhs.shape().dims())?;

        let grad_lhs = to_storage_preserving_graph(left_grad_dense)?;
        let grad_rhs = to_storage_preserving_graph(right_grad_dense)?;

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
    pub dim: Option<Vec<usize>>,
    pub keepdim: bool,
}

impl<B, S, T> SumFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Sum function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, dim: Option<Vec<usize>>, keepdim: bool) -> Self {
        Self {
            inputs: vec![input],
            dim,
            keepdim,
        }
    }
}

impl<B, S, T> Function<B, S, T> for SumFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + Copy + core::ops::Mul<Output = T> + 'static,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let input = &*self.inputs[0];

        let input_shape = input.shape().dims();
        let reduce_dims = canonical_reduce_dims(self.dim.as_deref(), input_shape.len())?;

        let grad_kept = if self.keepdim || input_shape.is_empty() {
            grad_output.clone()
        } else {
            let kept_shape =
                kept_shape_for_reduction(input_shape, &reduce_dims, grad_output.shape().dims())?;
            crate::tensor_ops::reshape(grad_output, &kept_shape)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        };

        let input_dense = to_dense_preserving_graph_identity(input)?;
        let ones: Tensor<B, DenseStorage<T>, T> =
            Tensor::ones_like(&input_dense).map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_broadcast_dense = crate::tensor_ops::mul(&ones, &grad_kept)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let grad_input = to_storage_preserving_graph(grad_broadcast_dense)?;

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
    pub dim: Option<Vec<usize>>,
    pub keepdim: bool,
}

impl<B, S, T> MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Mean function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, dim: Option<Vec<usize>>, keepdim: bool) -> Self {
        Self {
            inputs: vec![input],
            dim,
            keepdim,
        }
    }
}

impl<B, S, T> Function<B, S, T> for MeanFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType
        + Copy
        + num_traits::One
        + core::ops::Div<Output = T>
        + core::ops::Mul<Output = T>
        + 'static,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let input = &*self.inputs[0];
        let input_shape = input.shape().dims();
        let reduce_dims = canonical_reduce_dims(self.dim.as_deref(), input_shape.len())?;

        let mut reduced_numel = 1usize;
        for &d in &reduce_dims {
            reduced_numel = reduced_numel
                .checked_mul(input_shape[d])
                .ok_or_else(|| anyhow::anyhow!("Mean backward: reduced_numel overflow"))?;
        }

        let denom = T::from(reduced_numel)
            .ok_or_else(|| anyhow::anyhow!("Mean backward: reduced_numel not representable"))?;
        let scale = T::one() / denom;

        let grad_kept = if self.keepdim || input_shape.is_empty() {
            grad_output.clone()
        } else {
            let kept_shape =
                kept_shape_for_reduction(input_shape, &reduce_dims, grad_output.shape().dims())?;
            crate::tensor_ops::reshape(grad_output, &kept_shape)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        };

        let input_dense = to_dense_preserving_graph_identity(input)?;
        let ones: Tensor<B, DenseStorage<T>, T> =
            Tensor::ones_like(&input_dense).map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_broadcast_dense = crate::tensor_ops::mul(&ones, &grad_kept)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let scale_tensor: Tensor<B, DenseStorage<T>, T> =
            Tensor::from_vec_with_backend(vec![scale], &[], input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let grad_scaled_dense = crate::tensor_ops::mul(&grad_broadcast_dense, &scale_tensor)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let grad_input = to_storage_preserving_graph(grad_scaled_dense)?;

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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For exp operations: d/dx exp(x) = exp(x)
        // So gradient w.r.t. input is grad_output * exp(input)
        let input = &*self.inputs[0];
        let exp_input = input.exp();

        // Convert exp_input to dense for multiplication with grad_output
        let exp_input_dense = exp_input
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_input_dense = grad_output * &exp_input_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For log operations: d/dx log(x) = 1/x
        // So gradient w.r.t. input is grad_output / input
        let input = &*self.inputs[0];

        // Convert input to dense for division with grad_output
        let input_dense = input
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_input_dense = grad_output / &input_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For sin operations: d/dx sin(x) = cos(x)
        // So gradient w.r.t. input is grad_output * cos(input)
        let input = &*self.inputs[0];
        let cos_input = input.cos();

        // Convert cos_input to dense for multiplication with grad_output
        let cos_input_dense = cos_input
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_input_dense = grad_output * &cos_input_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, grad_output.backend().clone())?;

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

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // For cos operations: d/dx cos(x) = -sin(x)
        // So gradient w.r.t. input is grad_output * (-sin(input))
        let input = &*self.inputs[0];
        let sin_input = input.sin();

        // Convert sin_input to dense and negate it
        let sin_input_dense = sin_input
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let neg_one = T::zero() - T::one();
        let neg_one: Tensor<B, DenseStorage<T>, T> =
            Tensor::from_vec_with_backend(vec![neg_one], &[], sin_input_dense.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let neg_sin_dense = &sin_input_dense * &neg_one;

        let grad_input_dense = grad_output * &neg_sin_dense;

        // Convert back to storage type S
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

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

/// `NLLLoss` function for negative log likelihood loss
#[derive(Debug)]
pub struct NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [`log_probs`, `targets`]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new `NLLLoss` function
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
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + num_traits::FromPrimitive,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let log_probs = &*self.inputs[0];
        let targets = &*self.inputs[1];

        // NLL Loss backward: -1/N * grad_output at target indices
        let dims = log_probs.shape().dims();
        let batch_size = dims[0];
        let num_classes = dims[1];

        // Convert targets to dense to access indices
        let targets_dense = targets
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let targets_data = targets_dense.storage_ref().as_slice();

        let mut grad_data = Vec::with_capacity(batch_size * num_classes);

        let grad_val = if grad_output.numel() == 1 {
            grad_output.storage_ref().as_slice()[0]
        } else {
            return Err(anyhow::anyhow!(
                "NLL loss backward: expected scalar grad_output, got numel={}",
                grad_output.numel()
            ));
        };

        // Scale by 1/batch_size for mean reduction
        let scale = T::from_usize(batch_size)
            .ok_or_else(|| anyhow::anyhow!("NLL loss backward: batch_size not representable"))?;
        let neg_one = T::zero() - T::one();
        let grad_factor = (neg_one / scale) * grad_val;
        let zero = T::zero();

        for (b, target) in targets_data.iter().enumerate().take(batch_size) {
            let target_idx = target.to_usize().ok_or_else(|| {
                anyhow::anyhow!("NLL loss backward: target index at batch {b} not representable")
            })?;
            for c in 0..num_classes {
                if c == target_idx {
                    grad_data.push(grad_factor);
                } else {
                    grad_data.push(zero);
                }
            }
        }

        let grad_log_probs =
            Tensor::from_vec_with_backend(grad_data, dims, log_probs.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let grad_targets = Tensor::zeros(targets.shape().dims())
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        Ok(vec![grad_log_probs, grad_targets])
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
    #[must_use]
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

    fn backward(
        &self,
        _grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
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

/// Sqrt function for square root operations
#[derive(Debug)]
pub struct SqrtFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> SqrtFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Sqrt function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for SqrtFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + FromPrimitive,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d/dx sqrt(x) = 1 / (2 * sqrt(x))
        let input = &*self.inputs[0];

        // Convert to dense
        let input_dense = input
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let two = T::one() + T::one();
        let two_tensor: Tensor<B, DenseStorage<T>, T> =
            Tensor::from_vec_with_backend(vec![two], &[], input_dense.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        // 2 * sqrt(x)
        let sqrt_x = input_dense.sqrt();
        let denom = &sqrt_x * &two_tensor;

        // grad_output / denom
        let grad_input_dense = grad_output / &denom;

        // Convert back
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for SqrtFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

impl<B, S, T> tensor::AsAny for SqrtFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Pow function for power operations
#[derive(Debug)]
pub struct PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [input]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub exponent: f64,
}

impl<B, S, T> PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Pow function
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, exponent: f64) -> Self {
        Self {
            inputs: vec![input],
            exponent,
        }
    }
}

impl<B, S, T> Function<B, S, T> for PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + FromPrimitive,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d/dx x^n = n * x^(n-1)
        let input = &*self.inputs[0];
        let n = self.exponent;

        let input_dense = input
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        // Calculate x^(n-1)
        let input_data = input_dense.storage_ref().as_slice();
        let mut pow_data = Vec::with_capacity(input_data.len());
        for &val in input_data {
            let val_f64 = val.to_f64().ok_or_else(|| {
                anyhow::anyhow!("PowBackward requires elements convertible to f64")
            })?;
            let res = val_f64.powf(n - 1.0);
            let res_t = T::from_f64(res).ok_or_else(|| {
                anyhow::anyhow!(
                    "PowBackward failed to convert computed value {res} to target dtype"
                )
            })?;
            pow_data.push(res_t);
        }
        let pow_tensor =
            Tensor::<B, DenseStorage<T>, T>::from_vec(pow_data, input_dense.shape().dims())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        // n * x^(n-1)
        let n_val = T::from_f64(n).ok_or_else(|| {
            anyhow::anyhow!("PowBackward failed to convert exponent {n} to target dtype")
        })?;
        let n_tensor = Tensor::<B, DenseStorage<T>, T>::from_vec(vec![n_val], &[])
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        let deriv = &pow_tensor * &n_tensor;
        let grad_input_dense = grad_output * &deriv;

        // Convert back
        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "PowBackward"
    }
}

impl<B, S, T> tensor::AsAny for PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Reshape function
#[derive(Debug)]
pub struct ReshapeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub input_shape: Vec<usize>,
}

impl<B, S, T> ReshapeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, input_shape: Vec<usize>) -> Self {
        Self {
            inputs: vec![input],
            input_shape,
        }
    }
}

impl<B, S, T> Function<B, S, T> for ReshapeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let grad_data = grad_output.storage_ref().as_slice().to_vec();
        let input = &*self.inputs[0];

        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &self.input_shape, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for ReshapeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

impl<B, S, T> tensor::AsAny for ReshapeFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Max function
#[derive(Debug)]
pub struct MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub mask: Arc<Tensor<B, S, T>>,
    pub dim_val: usize,
    pub keepdim: bool,
}

impl<B, S, T> MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(
        input: Arc<Tensor<B, S, T>>,
        mask: Arc<Tensor<B, S, T>>,
        dim: usize,
        keepdim: bool,
    ) -> Self {
        Self {
            inputs: vec![input],
            mask,
            dim_val: dim,
            keepdim,
        }
    }
}

impl<B, S, T> Function<B, S, T> for MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let input = &*self.inputs[0];

        let mut grad_output_dense = grad_output
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        if !self.keepdim {
            let mut new_shape = grad_output_dense.shape().dims().to_vec();
            new_shape.insert(self.dim_val, 1);

            let data = grad_output_dense.storage_ref().as_slice().to_vec();
            grad_output_dense =
                Tensor::from_vec_with_backend(data, &new_shape, grad_output.backend().clone())
                    .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        }

        let mask_dense = self
            .mask
            .to_dense_preserving_identity()
            .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;
        let grad_input_dense = &grad_output_dense * &mask_dense;

        let grad_data = grad_input_dense.storage_ref().as_slice().to_vec();
        let grad_dims = grad_input_dense.shape().dims().to_vec();
        let grad_input =
            Tensor::from_vec_with_backend(grad_data, &grad_dims, input.backend().clone())
                .map_err(|e| anyhow::anyhow!("Tensor error: {e:?}"))?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

impl<B, S, T> tensor::AsAny for MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
