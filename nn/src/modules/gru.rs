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

use crate::Result;
use coeus_tensor::FloatDtype;
use rand::prelude::*;

/// GRU (Gated Recurrent Unit) layer
///
/// Implements a GRU cell with reset and update gates.
/// Compatible with PyTorch's `torch.nn.GRU`.
#[derive(Debug, Clone)]
pub struct Gru<T: FloatDtype> {
    /// Input-to-hidden weights for reset gate, shape (hidden_size, input_size)
    pub weight_ih_r: coeus_tensor::Tensor<T>,
    /// Hidden-to-hidden weights for reset gate, shape (hidden_size, hidden_size)
    pub weight_hh_r: coeus_tensor::Tensor<T>,
    /// Input-to-hidden weights for update gate, shape (hidden_size, input_size)
    pub weight_ih_z: coeus_tensor::Tensor<T>,
    /// Hidden-to-hidden weights for update gate, shape (hidden_size, hidden_size)
    pub weight_hh_z: coeus_tensor::Tensor<T>,
    /// Input-to-hidden weights for new gate, shape (hidden_size, input_size)
    pub weight_ih_n: coeus_tensor::Tensor<T>,
    /// Hidden-to-hidden weights for new gate, shape (hidden_size, hidden_size)
    pub weight_hh_n: coeus_tensor::Tensor<T>,
    /// Input-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_ih_r: Option<coeus_tensor::Tensor<T>>,
    /// Hidden-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_hh_r: Option<coeus_tensor::Tensor<T>>,
    /// Input-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_ih_z: Option<coeus_tensor::Tensor<T>>,
    /// Hidden-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_hh_z: Option<coeus_tensor::Tensor<T>>,
    /// Input-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_ih_n: Option<coeus_tensor::Tensor<T>>,
    /// Hidden-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_hh_n: Option<coeus_tensor::Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Gru<T> {
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
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_r = coeus_tensor::Tensor::from_vec(weight_ih_r_data, vec![hidden_size, input_size]);

        let weight_hh_r_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_r = coeus_tensor::Tensor::from_vec(weight_hh_r_data, vec![hidden_size, hidden_size]);

        let weight_ih_z_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_z = coeus_tensor::Tensor::from_vec(weight_ih_z_data, vec![hidden_size, input_size]);

        let weight_hh_z_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_z = coeus_tensor::Tensor::from_vec(weight_hh_z_data, vec![hidden_size, hidden_size]);

        let weight_ih_n_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_n = coeus_tensor::Tensor::from_vec(weight_ih_n_data, vec![hidden_size, input_size]);

        let weight_hh_n_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_n = coeus_tensor::Tensor::from_vec(weight_hh_n_data, vec![hidden_size, hidden_size]);

        // Create biases sequentially
        let bias_ih_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_r = Some(coeus_tensor::Tensor::from_vec(bias_ih_r_data, vec![hidden_size]));

        let bias_hh_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_r = Some(coeus_tensor::Tensor::from_vec(bias_hh_r_data, vec![hidden_size]));

        let bias_ih_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_z = Some(coeus_tensor::Tensor::from_vec(bias_ih_z_data, vec![hidden_size]));

        let bias_hh_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_z = Some(coeus_tensor::Tensor::from_vec(bias_hh_z_data, vec![hidden_size]));

        let bias_ih_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_n = Some(coeus_tensor::Tensor::from_vec(bias_ih_n_data, vec![hidden_size]));

        let bias_hh_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_n = Some(coeus_tensor::Tensor::from_vec(bias_hh_n_data, vec![hidden_size]));

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

    /// Forward pass through the GRU (placeholder implementation)
    pub fn forward(
        &self,
        _input: &coeus_tensor::Tensor<T>,
        _h_0: Option<&coeus_tensor::Tensor<T>>,
    ) -> Result<(coeus_tensor::Tensor<T>, coeus_tensor::Tensor<T>)> {
        // Placeholder implementation - needs proper GRU forward pass
        Err(crate::NNError::InvalidInput {
            message: "GRU forward pass not yet implemented".to_string(),
        })
    }

    pub fn parameters(&self) -> Vec<&coeus_tensor::Tensor<T>> {
        let mut params = vec![
            &self.weight_ih_r,
            &self.weight_hh_r,
            &self.weight_ih_z,
            &self.weight_hh_z,
            &self.weight_ih_n,
            &self.weight_hh_n,
        ];

        // Add biases if present
        if let Some(ref bias) = self.bias_ih_r {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_r {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_z {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_z {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_n {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_n {
            params.push(bias);
        }

        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut coeus_tensor::Tensor<T>> {
        let mut params = vec![
            &mut self.weight_ih_r,
            &mut self.weight_hh_r,
            &mut self.weight_ih_z,
            &mut self.weight_hh_z,
            &mut self.weight_ih_n,
            &mut self.weight_hh_n,
        ];

        // Add biases if present
        if let Some(ref mut bias) = self.bias_ih_r {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_r {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_z {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_z {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_n {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_n {
            params.push(bias);
        }

        params
    }
}

/// GRUCell (Gated Recurrent Unit Cell)
///
/// A single GRU cell that processes one timestep at a time.
/// Compatible with PyTorch's `torch.nn.GRUCell`.
#[derive(Debug, Clone)]
pub struct GruCell<T: FloatDtype> {
    /// Input-to-hidden weights for reset gate, shape (hidden_size, input_size)
    pub weight_ih_r: coeus_tensor::Tensor<T>,
    /// Hidden-to-hidden weights for reset gate, shape (hidden_size, hidden_size)
    pub weight_hh_r: coeus_tensor::Tensor<T>,
    /// Input-to-hidden weights for update gate, shape (hidden_size, input_size)
    pub weight_ih_z: coeus_tensor::Tensor<T>,
    /// Hidden-to-hidden weights for update gate, shape (hidden_size, hidden_size)
    pub weight_hh_z: coeus_tensor::Tensor<T>,
    /// Input-to-hidden weights for new gate, shape (hidden_size, input_size)
    pub weight_ih_n: coeus_tensor::Tensor<T>,
    /// Hidden-to-hidden weights for new gate, shape (hidden_size, hidden_size)
    pub weight_hh_n: coeus_tensor::Tensor<T>,
    /// Input-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_ih_r: Option<coeus_tensor::Tensor<T>>,
    /// Hidden-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_hh_r: Option<coeus_tensor::Tensor<T>>,
    /// Input-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_ih_z: Option<coeus_tensor::Tensor<T>>,
    /// Hidden-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_hh_z: Option<coeus_tensor::Tensor<T>>,
    /// Input-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_ih_n: Option<coeus_tensor::Tensor<T>>,
    /// Hidden-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_hh_n: Option<coeus_tensor::Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> GruCell<T> {
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
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_r = coeus_tensor::Tensor::from_vec(weight_ih_r_data, vec![hidden_size, input_size]);

        let weight_hh_r_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_r = coeus_tensor::Tensor::from_vec(weight_hh_r_data, vec![hidden_size, hidden_size]);

        let weight_ih_z_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_z = coeus_tensor::Tensor::from_vec(weight_ih_z_data, vec![hidden_size, input_size]);

        let weight_hh_z_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_z = coeus_tensor::Tensor::from_vec(weight_hh_z_data, vec![hidden_size, hidden_size]);

        let weight_ih_n_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_ih_n = coeus_tensor::Tensor::from_vec(weight_ih_n_data, vec![hidden_size, input_size]);

        let weight_hh_n_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_n = coeus_tensor::Tensor::from_vec(weight_hh_n_data, vec![hidden_size, hidden_size]);

        // Create biases sequentially
        let bias_ih_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_r = Some(coeus_tensor::Tensor::from_vec(bias_ih_r_data, vec![hidden_size]));

        let bias_hh_r_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_r = Some(coeus_tensor::Tensor::from_vec(bias_hh_r_data, vec![hidden_size]));

        let bias_ih_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_z = Some(coeus_tensor::Tensor::from_vec(bias_ih_z_data, vec![hidden_size]));

        let bias_hh_z_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_z = Some(coeus_tensor::Tensor::from_vec(bias_hh_z_data, vec![hidden_size]));

        let bias_ih_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_ih_n = Some(coeus_tensor::Tensor::from_vec(bias_ih_n_data, vec![hidden_size]));

        let bias_hh_n_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let bias_hh_n = Some(coeus_tensor::Tensor::from_vec(bias_hh_n_data, vec![hidden_size]));

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

    /// Forward pass through a single GRU cell (placeholder implementation)
    pub fn forward(
        &self,
        _input: &coeus_tensor::Tensor<T>,
        _hx: Option<&coeus_tensor::Tensor<T>>,
    ) -> Result<coeus_tensor::Tensor<T>> {
        // Placeholder implementation - needs proper GRU cell forward pass
        Err(crate::NNError::InvalidInput {
            message: "GRUCell forward pass not yet implemented".to_string(),
        })
    }

    pub fn parameters(&self) -> Vec<&coeus_tensor::Tensor<T>> {
        let mut params = vec![
            &self.weight_ih_r,
            &self.weight_hh_r,
            &self.weight_ih_z,
            &self.weight_hh_z,
            &self.weight_ih_n,
            &self.weight_hh_n,
        ];

        // Add biases if present
        if let Some(ref bias) = self.bias_ih_r {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_r {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_z {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_z {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_n {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_n {
            params.push(bias);
        }

        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut coeus_tensor::Tensor<T>> {
        let mut params = vec![
            &mut self.weight_ih_r,
            &mut self.weight_hh_r,
            &mut self.weight_ih_z,
            &mut self.weight_hh_z,
            &mut self.weight_ih_n,
            &mut self.weight_hh_n,
        ];

        // Add biases if present
        if let Some(ref mut bias) = self.bias_ih_r {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_r {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_z {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_z {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_n {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_n {
            params.push(bias);
        }

        params
    }
}
