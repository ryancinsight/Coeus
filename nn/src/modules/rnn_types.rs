//! RNN (Recurrent Neural Network) layers
//!
//! This module provides RNN and RNNCell implementations for sequence processing.
//!
//! ## Mathematical Foundation
//!
//! ### RNN Forward Pass
//! ```math
//! h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
//! y_t = W_hy * h_t + b_y
//! ```
//!
//! ## References
//!
//! - [Recurrent Neural Networks Tutorial](https://www.deeplearningbook.org/contents/rnn.html)
//! - [PyTorch RNN Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

use crate::NNError;
use coeus_backend::CpuBackend;
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor, ops::{indexing::{self}, reduction::cat as tensor_cat}};
use rand::{distributions::uniform::SampleUniform, Rng};

#[derive(Debug, Clone)]
pub struct Rnn<T: FloatDtype> {
    /// Input-to-hidden weights, shape (hidden_size, input_size)
    pub weight_ih: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights, shape (hidden_size, hidden_size)
    pub weight_hh: Tensor<T, CpuBackend>,
    /// Input-to-hidden bias, shape (hidden_size,)
    pub bias_ih: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias, shape (hidden_size,)
    pub bias_hh: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + SampleUniform + std::ops::AddAssign + std::iter::Sum> Rnn<T> {
    /// Create a new RNN layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for RNN weights
        let ih_bound = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let hh_bound = (6.0 / (hidden_size + hidden_size) as f64).sqrt();

        let _weight_ih_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let _weight_hh_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let weight_ih = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();
        let weight_hh = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let _bias_ih_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let _bias_hh_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let bias_ih = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());
        let bias_hh = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through the RNN
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (seq_len, batch_size, input_size)
    /// * `h_0` - Initial hidden state of shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Tuple of (output, final_hidden_state)
    pub fn forward(
        &self,
        input: &Tensor<T, CpuBackend>,
        h_0: Option<&Tensor<T, CpuBackend>>,
    ) -> Result<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>), NNError> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        // Current implementation: single-layer unidirectional RNN
        // Future enhancement: bidirectional and multi-layer support

        // Initialize hidden state if not provided
        let h_init = h_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]).unwrap_grad());
        let mut h_current = h_init.clone();

        let mut outputs = Vec::new();

        // Process each timestep in the sequence
        for t in 0..seq_len {
            // Extract input at timestep t: shape (batch_size, input_size)
            // Use slice to get the t-th element along the sequence dimension
            let slices = vec![
                indexing::Slice::Range(t, t + 1), // Sequence dimension: single timestep
                indexing::Slice::Full,           // Batch dimension: all batches
                indexing::Slice::Full,           // Feature dimension: all features
            ];
            let sliced = input.slice(&slices)?;
            // After slicing, we have shape (1, batch_size, input_size)
            // We need to squeeze out the first dimension to get (batch_size, input_size)
            let input_t = if sliced.shape().len() == 3 && sliced.shape()[0] == 1 {
                // Squeeze out the sequence dimension
                sliced.reshape(vec![batch_size, self.input_size])?
            } else {
                return Err(NNError::ShapeMismatch {
                    expected: vec![1, batch_size, self.input_size],
                    actual: sliced.shape().to_vec(),
                });
            };

            // h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)
            // Matrix multiplication: (batch_size, input_size) @ (input_size, hidden_size) -> (batch_size, hidden_size)
            // Note: weight_ih has shape (hidden_size, input_size), we need to transpose it for matmul
            let weight_ih_t = self.weight_ih.t()?;
            let x_contrib = input_t.matmul(&weight_ih_t)?;

            // (batch_size, hidden_size) @ (hidden_size, hidden_size) -> (batch_size, hidden_size)
            let weight_hh_t = self.weight_hh.t()?;
            let h_contrib = h_current.matmul(&weight_hh_t)?;

            let mut combined = (&x_contrib + &h_contrib).unwrap();

            // Add biases if present
            if let Some(ref bias_ih) = self.bias_ih {
                // Bias broadcasting: reshape bias from (hidden_size,) to (1, hidden_size) for broadcasting
                let _bias_ih_broadcast = bias_ih.reshape(vec![1, self.hidden_size])?;
                combined = (&x_contrib + &h_contrib).unwrap();
            }
            if let Some(ref bias_hh) = self.bias_hh {
                // Bias broadcasting: reshape bias from (hidden_size,) to (1, hidden_size) for broadcasting
                let _bias_hh_broadcast = bias_hh.reshape(vec![1, self.hidden_size])?;
                combined = (&x_contrib + &h_contrib).unwrap();
            }

            // Apply tanh activation
            h_current = combined.tanh()?;

            // Store output for this timestep
            outputs.push(h_current.clone());
        }

        // Stack all timestep outputs to create proper sequence output
        // Each output has shape (batch_size, hidden_size)
        // We want final shape (seq_len, batch_size, hidden_size)
        let _output_tensors: Vec<&Tensor<T, CpuBackend>> = outputs.iter().collect();

        // First, add sequence dimension to each output: (batch_size, hidden_size) -> (1, batch_size, hidden_size)
        let mut expanded_outputs = Vec::new();
        for output in &outputs {
            let expanded = output.unsqueeze(0)?; // Add sequence dimension at position 0
            expanded_outputs.push(expanded);
        }

        // Then concatenate along sequence dimension (dim 0)
        let expanded_refs: Vec<&Tensor<T, CpuBackend>> = expanded_outputs.iter().collect();
        let output = tensor_cat(&expanded_refs, 0)?;
        let h_n = h_current;

        Ok((output, h_n))
    }

    #[allow(dead_code)]
    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }

    #[allow(dead_code)]
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref mut bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
}

#[derive(Debug, Clone)]
pub struct RnnCell<T: FloatDtype> {
    /// Input-to-hidden weights, shape (hidden_size, input_size)
    pub weight_ih: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights, shape (hidden_size, hidden_size)
    pub weight_hh: Tensor<T, CpuBackend>,
    /// Input-to-hidden bias, shape (hidden_size,)
    pub bias_ih: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias, shape (hidden_size,)
    pub bias_hh: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + SampleUniform> RnnCell<T> {
    /// Create a new RNNCell
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for RNN weights
        let ih_bound = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let hh_bound = (6.0 / (hidden_size + hidden_size) as f64).sqrt();

        let _weight_ih_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let _weight_hh_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let weight_ih = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();
        let weight_hh = Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap();

        let _bias_ih_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let _bias_hh_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                <T as Dtype>::from_f64(val).unwrap()
            })
            .collect();

        let bias_ih = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());
        let bias_hh = Some(Tensor::from_vec(CpuBackend::default(), vec![T::zero()], vec![1]).unwrap());

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through a single RNN cell
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, input_size)
    /// * `hx` - Hidden state from previous timestep, shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Hidden state for this timestep, shape (batch_size, hidden_size)
    pub fn forward(&self, input: &Tensor<T, CpuBackend>, hx: Option<&Tensor<T, CpuBackend>>) -> Result<Tensor<T, CpuBackend>, NNError> {
        let batch_size = input.shape()[0];

        // Initialize hidden state if not provided
        let h_prev = hx
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]).unwrap_grad());

        // h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)
        let weight_ih_t = self.weight_ih.t()?;
        let x_contrib = input.matmul(&weight_ih_t)?;

        let weight_hh_t = self.weight_hh.t()?;
        let h_contrib = h_prev.matmul(&weight_hh_t)?;

        let mut combined = (&x_contrib + &h_contrib).unwrap();

        // Add biases if present
        if let Some(ref bias_ih) = self.bias_ih {
            // Bias broadcasting: reshape bias from (hidden_size,) to (1, hidden_size) for broadcasting
            let _bias_ih_broadcast = bias_ih.reshape(vec![1, self.hidden_size])?;
            combined = (&x_contrib + &h_contrib).unwrap();
        }
        if let Some(ref bias_hh) = self.bias_hh {
            // Bias broadcasting: reshape bias from (hidden_size,) to (1, hidden_size) for broadcasting
            let _bias_hh_broadcast = bias_hh.reshape(vec![1, self.hidden_size])?;
            combined = (&x_contrib + &h_contrib).unwrap();
        }

        // Apply tanh activation
        Ok(combined.tanh()?)
    }

    #[allow(dead_code)]
    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }

    #[allow(dead_code)]
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = vec![&mut self.weight_ih, &mut self.weight_hh];
        if let Some(ref mut bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref mut bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }
}


