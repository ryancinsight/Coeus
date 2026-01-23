//! GRUCell (Gated Recurrent Unit Cell) implementation.

use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::add, ops::arithmetic::mul, ops::arithmetic::sub, Tensor};

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// GRU (Gated Recurrent Unit) Cell.
///
/// Implements a single step of the GRU architecture.
#[derive(Debug, Clone)]
pub struct GRUCell<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Input-to-hidden weights [3 * hidden_size, input_size]
    pub weight_ih: Parameter<B, S, T>,
    /// Hidden-to-hidden weights [3 * hidden_size, hidden_size]
    pub weight_hh: Parameter<B, S, T>,
    /// Input-to-hidden biases [3 * hidden_size]
    pub bias_ih: Option<Parameter<B, S, T>>,
    /// Hidden-to-hidden biases [3 * hidden_size]
    pub bias_hh: Option<Parameter<B, S, T>>,
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> GRUCell<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new GRUCell.
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self> {
        Self::new_with_backend(B::default(), input_size, hidden_size, bias)
    }

    /// Create a new GRUCell with a specific backend.
    pub fn new_with_backend(
        backend: B,
        input_size: usize,
        hidden_size: usize,
        bias: bool,
    ) -> Result<Self> {
        let gate_size = 3 * hidden_size;

        // Initialize weights in dense storage first for initialization logic
        let w_ih_size = gate_size.checked_mul(input_size).ok_or_else(|| {
            crate::core::error::NNError::InvalidInput {
                message: "GRUCell weight_ih size overflows usize".to_string(),
            }
        })?;
        let mut w_ih_dense = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            vec![T::zero(); w_ih_size],
            &[gate_size, input_size],
            backend.clone(),
        )?;
        crate::init::xavier_uniform_(&mut w_ih_dense, 1.0)?;
        let w_ih_storage = S::from_vec(w_ih_dense.as_slice().to_vec(), &[gate_size, input_size])?;
        let weight_ih = Parameter::new(
            Tensor::from_storage(w_ih_storage, backend.clone()).requires_grad_(true),
            "weight_ih".to_string(),
        );

        let w_hh_size = gate_size.checked_mul(hidden_size).ok_or_else(|| {
            crate::core::error::NNError::InvalidInput {
                message: "GRUCell weight_hh size overflows usize".to_string(),
            }
        })?;
        let mut w_hh_dense = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            vec![T::zero(); w_hh_size],
            &[gate_size, hidden_size],
            backend.clone(),
        )?;
        crate::init::xavier_uniform_(&mut w_hh_dense, 1.0)?;
        let w_hh_storage = S::from_vec(w_hh_dense.as_slice().to_vec(), &[gate_size, hidden_size])?;
        let weight_hh = Parameter::new(
            Tensor::from_storage(w_hh_storage, backend.clone()).requires_grad_(true),
            "weight_hh".to_string(),
        );

        let mut bias_ih = None;
        let mut bias_hh = None;

        if bias {
            let b_ih_data = vec![T::zero(); gate_size];
            let b_ih_storage = S::from_vec(b_ih_data, &[gate_size])?;
            bias_ih = Some(Parameter::new(
                Tensor::from_storage(b_ih_storage, backend.clone()).requires_grad_(true),
                "bias_ih".to_string(),
            ));

            let b_hh_data = vec![T::zero(); gate_size];
            let b_hh_storage = S::from_vec(b_hh_data, &[gate_size])?;
            bias_hh = Some(Parameter::new(
                Tensor::from_storage(b_hh_storage, backend).requires_grad_(true),
                "bias_hh".to_string(),
            ));
        }

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            _phantom: PhantomData,
        })
    }

    /// Forward pass for a single step.
    pub fn forward_step(
        &self,
        x: &Tensor<B, S, T>,
        h: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let weight_ih = self.weight_ih.data().to_dense_generic()?;
        let weight_hh = self.weight_hh.data().to_dense_generic()?;
        let x_dense = x.to_dense_generic()?;
        let h_dense = h.to_dense_generic()?;

        let mut ih_gates = tensor::ops::matmul(&x_dense, &weight_ih.transpose(1, 0)?)?;
        let mut hh_gates = tensor::ops::matmul(&h_dense, &weight_hh.transpose(1, 0)?)?;

        if let (Some(b_ih_p), Some(b_hh_p)) = (&self.bias_ih, &self.bias_hh) {
            let b_ih = b_ih_p.data().to_dense_generic()?;
            let b_hh = b_hh_p.data().to_dense_generic()?;
            ih_gates = add(&ih_gates, &b_ih)?;
            hh_gates = add(&hh_gates, &b_hh)?;
        }

        let hs = self.hidden_size as i32;
        let r_ih = ih_gates.advanced_slice(&[(None, None, 1), (Some(0), Some(hs), 1)])?;
        let z_ih = ih_gates.advanced_slice(&[(None, None, 1), (Some(hs), Some(2 * hs), 1)])?;
        let n_ih = ih_gates.advanced_slice(&[(None, None, 1), (Some(2 * hs), Some(3 * hs), 1)])?;

        let r_hh = hh_gates.advanced_slice(&[(None, None, 1), (Some(0), Some(hs), 1)])?;
        let z_hh = hh_gates.advanced_slice(&[(None, None, 1), (Some(hs), Some(2 * hs), 1)])?;
        let n_hh = hh_gates.advanced_slice(&[(None, None, 1), (Some(2 * hs), Some(3 * hs), 1)])?;

        let r = crate::functional_api::sigmoid(&add(&r_ih, &r_hh)?)?;
        let z = crate::functional_api::sigmoid(&add(&z_ih, &z_hh)?)?;
        let n = crate::functional_api::tanh(&add(&n_ih, &mul(&r, &n_hh)?)?)?;

        let ones = Tensor::<B, DenseStorage<T>, T>::ones_with_backend(
            h_dense.shape().dims(),
            x.backend().clone(),
        )?;
        let one_minus_z = sub(&ones, &z)?;
        let n_part = mul(&one_minus_z, &n)?;
        let h_part = mul(&z, &h_dense)?;

        let next_h_dense = add(&n_part, &h_part)?;

        // Convert back to original storage type if needed
        let result_data = next_h_dense.as_slice().to_vec();
        let result_shape = next_h_dense.shape().dims();
        let result_storage = S::from_vec(result_data, result_shape)?;

        Ok(Tensor::from_storage(result_storage, x.backend().clone()))
    }
}

impl<B, S, T> Module<B, S, T> for GRUCell<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // For a single cell, we assume input is [batch_size, input_size]
        // and hidden state is zeros if not provided. This matches PyTorch GRUCell.
        let batch_size = input.shape().dims()[0];
        let h_storage = S::zeros(&[batch_size, self.hidden_size])?;
        let h = Tensor::from_storage(h_storage, input.backend().clone());
        self.forward_step(input, &h)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(b_ih) = &self.bias_ih {
            params.push(b_ih.clone());
        }
        if let Some(b_hh) = &self.bias_hh {
            params.push(b_hh.clone());
        }
        params
    }

    fn name(&self) -> &str {
        "GRUCell"
    }

    fn train(&mut self, _mode: bool) {}

    fn zero_grad(&mut self) {
        self.weight_ih.zero_grad();
        self.weight_hh.zero_grad();
        if let Some(b_ih) = &mut self.bias_ih {
            b_ih.zero_grad();
        }
        if let Some(b_hh) = &mut self.bias_hh {
            b_hh.zero_grad();
        }
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
