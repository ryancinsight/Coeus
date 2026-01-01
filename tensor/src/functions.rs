//! Function objects for automatic differentiation
//!
//! This module defines the backward functions that implement gradient computation
//! for various tensor operations in the computation graph.

extern crate alloc;

use crate::{DifferentiableFunction, Function, Tensor};
use alloc::sync::Arc;
use alloc::vec::Vec;
use anyhow;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec};

fn unbroadcast_dense<B, T>(
    grad_output: &Tensor<B, DenseStorage<T>, T>,
    input_shape: &[usize],
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Clone + Copy,
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
        let data = reduced.as_slice().to_vec();
        let expected = input_shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(anyhow::anyhow!("unbroadcast numel mismatch"));
        }
        reduced = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            data,
            input_shape,
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    }

    Ok(reduced)
}

fn broadcast_to_shape_dense<B, S, T>(
    input: &Tensor<B, S, T>,
    target_shape: &[usize],
) -> anyhow::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + Clone + Copy,
{
    if input.shape().dims() == target_shape {
        return Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            input.as_slice().to_vec(),
            target_shape,
            input.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()));
    }

    let data = crate::ops::arithmetic::broadcast_tensor_data(
        input.as_slice(),
        input.shape().dims(),
        target_shape,
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
        data,
        target_shape,
        input.backend().clone(),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))
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

impl<B, S, T> crate::AsAny for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new AddFunction with the given inputs
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>) -> Self {
        Self { inputs }
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

impl<B, S, T> Function<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> std::result::Result<Vec<Tensor<B, S, T>>, anyhow::Error> {
        let mut result = Vec::with_capacity(self.inputs.len());

        for input in &self.inputs {
            let dense = unbroadcast_dense(grad_output, input.shape().dims())?;
            let grad_tensor = Tensor::from_vec_with_backend(
                dense.as_slice().to_vec(),
                dense.shape().dims(),
                input.backend().clone(),
            )?;
            result.push(grad_tensor);
        }

        Ok(result)
    }
}

#[derive(Debug)]
pub struct SubFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> crate::AsAny for SubFunction<B, S, T>
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
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>) -> Self {
        Self { inputs }
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

impl<B, S, T> Function<B, S, T> for SubFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> std::result::Result<Vec<Tensor<B, S, T>>, anyhow::Error> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let grad_lhs_dense = unbroadcast_dense(grad_output, lhs.shape().dims())?;

        let grad_rhs_out_data: Vec<T> = grad_output
            .as_slice()
            .iter()
            .map(|&g| T::zero() - g)
            .collect();
        let grad_rhs_out = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            grad_rhs_out_data,
            grad_output.shape().dims(),
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let grad_rhs_dense = unbroadcast_dense(&grad_rhs_out, rhs.shape().dims())?;

        let grad_lhs = Tensor::from_vec_with_backend(
            grad_lhs_dense.as_slice().to_vec(),
            grad_lhs_dense.shape().dims(),
            lhs.backend().clone(),
        )?;
        let grad_rhs = Tensor::from_vec_with_backend(
            grad_rhs_dense.as_slice().to_vec(),
            grad_rhs_dense.shape().dims(),
            rhs.backend().clone(),
        )?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

#[derive(Debug)]
pub struct MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> crate::AsAny for MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>) -> Self {
        Self { inputs }
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

impl<B, S, T> Function<B, S, T> for MulFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> std::result::Result<Vec<Tensor<B, S, T>>, anyhow::Error> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let out_shape = grad_output.shape().dims();
        let lhs_b = broadcast_to_shape_dense(lhs, out_shape)?;
        let rhs_b = broadcast_to_shape_dense(rhs, out_shape)?;

        let grad_lhs_out_data: Vec<T> = grad_output
            .as_slice()
            .iter()
            .zip(rhs_b.as_slice())
            .map(|(&g, &b)| g * b)
            .collect();
        let grad_rhs_out_data: Vec<T> = grad_output
            .as_slice()
            .iter()
            .zip(lhs_b.as_slice())
            .map(|(&g, &a)| g * a)
            .collect();

        let grad_lhs_out = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            grad_lhs_out_data,
            out_shape,
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let grad_rhs_out = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            grad_rhs_out_data,
            out_shape,
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let grad_lhs_dense = unbroadcast_dense(&grad_lhs_out, lhs.shape().dims())?;
        let grad_rhs_dense = unbroadcast_dense(&grad_rhs_out, rhs.shape().dims())?;

        let grad_lhs = Tensor::from_vec_with_backend(
            grad_lhs_dense.as_slice().to_vec(),
            grad_lhs_dense.shape().dims(),
            lhs.backend().clone(),
        )?;
        let grad_rhs = Tensor::from_vec_with_backend(
            grad_rhs_dense.as_slice().to_vec(),
            grad_rhs_dense.shape().dims(),
            rhs.backend().clone(),
        )?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

#[derive(Debug)]
pub struct DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> crate::AsAny for DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> DivFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>) -> Self {
        Self { inputs }
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

impl<B, S, T> Function<B, S, T> for DivFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> std::result::Result<Vec<Tensor<B, S, T>>, anyhow::Error> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let out_shape = grad_output.shape().dims();
        let lhs_b = broadcast_to_shape_dense(lhs, out_shape)?;
        let rhs_b = broadcast_to_shape_dense(rhs, out_shape)?;

        let grad_lhs_out_data: Vec<T> = grad_output
            .as_slice()
            .iter()
            .zip(rhs_b.as_slice())
            .map(|(&g, &b)| g / b)
            .collect();

        let grad_rhs_out_data: Vec<T> = grad_output
            .as_slice()
            .iter()
            .zip(lhs_b.as_slice())
            .zip(rhs_b.as_slice())
            .map(|((&g, &a), &b)| {
                let numerator = g * a;
                let denom = b * b;
                (T::zero() - numerator) / denom
            })
            .collect();

        let grad_lhs_out = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            grad_lhs_out_data,
            out_shape,
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let grad_rhs_out = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            grad_rhs_out_data,
            out_shape,
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let grad_lhs_dense = unbroadcast_dense(&grad_lhs_out, lhs.shape().dims())?;
        let grad_rhs_dense = unbroadcast_dense(&grad_rhs_out, rhs.shape().dims())?;

        let grad_lhs = Tensor::from_vec_with_backend(
            grad_lhs_dense.as_slice().to_vec(),
            grad_lhs_dense.shape().dims(),
            lhs.backend().clone(),
        )?;
        let grad_rhs = Tensor::from_vec_with_backend(
            grad_rhs_dense.as_slice().to_vec(),
            grad_rhs_dense.shape().dims(),
            rhs.backend().clone(),
        )?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

#[derive(Debug)]
pub struct NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> crate::AsAny for NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<B, S, T> NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>) -> Self {
        Self { inputs }
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

impl<B, S, T> Function<B, S, T> for NegFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> std::result::Result<Vec<Tensor<B, S, T>>, anyhow::Error> {
        let input = &*self.inputs[0];
        let out_data: Vec<T> = grad_output
            .as_slice()
            .iter()
            .map(|&g| T::zero() - g)
            .collect();
        let out = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            out_data,
            grad_output.shape().dims(),
            grad_output.backend().clone(),
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let grad_dense = unbroadcast_dense(&out, input.shape().dims())?;
        let grad = Tensor::from_vec_with_backend(
            grad_dense.as_slice().to_vec(),
            grad_dense.shape().dims(),
            input.backend().clone(),
        )?;
        Ok(vec![grad])
    }
}
