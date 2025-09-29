//! GRU (Gated Recurrent Unit) layers
//!
//! This module provides GRU and GRUCell implementations for sequence processing.
//!
//! ## Mathematical Foundation
//!
//! ### GRU Cell Update
//! ```math
//! r_t = σ(W_ir * x_t + W_hr * h_{t-1} + b_ir + b_hr)  // reset gate
//! z_t = σ(W_iz * x_t + W_hz * h_{t-1} + b_iz + b_hz)  // update gate
//! n_t = tanh(W_in * x_t + W_hn * (r_t * h_{t-1}) + b_in + b_hn) // new gate
//! h_t = (1 - z_t) * n_t + z_t * h_{t-1}  // hidden state
//! ```
//!
//! ## References
//!
//! - [Cho et al., 2014 - Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)
//! - [PyTorch GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor};
use rand::{distributions::uniform::SampleUniform, Rng};

#[derive(Debug, Clone)]
pub struct Gru<T: FloatDtype> {
    /// Input-to-hidden weights for reset gate, shape (hidden_size, input_size)
    pub weight_ih_r: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for reset gate, shape (hidden_size, hidden_size)
    pub weight_hh_r: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for update gate, shape (hidden_size, input_size)
    pub weight_ih_z: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for update gate, shape (hidden_size, hidden_size)
    pub weight_hh_z: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for new gate, shape (hidden_size, input_size)
    pub weight_ih_n: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for new gate, shape (hidden_size, hidden_size)
    pub weight_hh_n: Tensor<T, CpuBackend>,
    /// Input-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_ih_r: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_hh_r: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_ih_z: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_hh_z: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_ih_n: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_hh_n: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + SampleUniform> Gru<T> {
    /// Create a new GRU layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for GRU weights
        let bound = (6.0 / (input_size + hidden_size) as f64).sqrt();

        // Create weights sequentially to avoid borrowing conflicts
        let weight_ih_r_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_r = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_hh_r_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_r = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_ih_z_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_z = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_hh_z_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_z = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_ih_n_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_n = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_hh_n_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_n = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        // Create biases sequentially
        let bias_ih_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_r = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_hh_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_r = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_ih_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_z = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_hh_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_z = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_ih_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_n = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_hh_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_n = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        Self {
            weight_ih_r,
            weight_hh_r,
            weight_ih_z,
            weight_hh_z,
            weight_ih_n,
            weight_hh_n,
            bias_ih_r,
            bias_hh_r,
            bias_ih_z,
            bias_hh_z,
            bias_ih_n,
            bias_hh_n,
            input_size,
            hidden_size,
        }
    }
}


#[derive(Debug, Clone)]
pub struct GruCell<T: FloatDtype> {
    /// Input-to-hidden weights for reset gate, shape (hidden_size, input_size)
    pub weight_ih_r: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for reset gate, shape (hidden_size, hidden_size)
    pub weight_hh_r: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for update gate, shape (hidden_size, input_size)
    pub weight_ih_z: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for update gate, shape (hidden_size, hidden_size)
    pub weight_hh_z: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for new gate, shape (hidden_size, input_size)
    pub weight_ih_n: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for new gate, shape (hidden_size, hidden_size)
    pub weight_hh_n: Tensor<T, CpuBackend>,
    /// Input-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_ih_r: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_hh_r: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_ih_z: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_hh_z: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_ih_n: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_hh_n: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + SampleUniform + std::ops::AddAssign> GruCell<T> {
    /// Create a new GRUCell
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for GRU weights
        let bound = (6.0 / (input_size + hidden_size) as f64).sqrt();

        // Create weights sequentially to avoid borrowing conflicts
        let weight_ih_r_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_r = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_hh_r_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_r = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_ih_z_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_z = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_hh_z_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_z = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_ih_n_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_n = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let weight_hh_n_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_n = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        // Create biases sequentially
        let bias_ih_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_r = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_hh_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_r = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_ih_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_z = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_hh_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_z = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_ih_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_n = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        let bias_hh_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_n = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        Self {
            weight_ih_r,
            weight_hh_r,
            weight_ih_z,
            weight_hh_z,
            weight_ih_n,
            weight_hh_n,
            bias_ih_r,
            bias_hh_r,
            bias_ih_z,
            bias_hh_z,
            bias_ih_n,
            bias_hh_n,
            input_size,
            hidden_size,
        }
    }
}



