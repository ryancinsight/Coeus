//! LSTM (Long Short-Term Memory) layers
//!
//! This module provides LSTM and LSTMCell implementations for sequence processing
//! with memory cells and gating mechanisms.
//!
//! ## Mathematical Foundation
//!
//! ### LSTM Cell Update
//! ```math
//! i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_ii + b_hi)  // input gate
//! f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_if + b_hf)  // forget gate
//! g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_ig + b_hg) // cell gate
//! o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_io + b_ho)  // output gate
//! c_t = f_t * c_{t-1} + i_t * g_t                       // cell state
//! h_t = o_t * tanh(c_t)                                // hidden state
//! ```
//!
//! ## References
//!
//! - [Hochreiter & Schmidhuber, 1997 - Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
//! - [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

use crate::NNError;
use coeus_backend::CpuBackend;
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor, ops::{indexing::Slice, reduction::cat as tensor_cat}};
use rand::prelude::*;
use std::fmt;

/// Type alias for LSTM forward pass return value to reduce type complexity
pub type LstmOutput<T> = (Tensor<T, CpuBackend>, (Tensor<T, CpuBackend>, Tensor<T, CpuBackend>));

/// LSTM (Long Short-Term Memory) layer
///
/// Implements an LSTM cell with input, forget, output, and cell gates.
/// Compatible with PyTorch's `torch.nn.LSTM`.
#[derive(Debug, Clone)]
pub struct Lstm<T: FloatDtype> {
    /// Input-to-hidden weights for input gate, shape (hidden_size, input_size)
    pub weight_ih_i: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for input gate, shape (hidden_size, hidden_size)
    pub weight_hh_i: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for forget gate, shape (hidden_size, input_size)
    pub weight_ih_f: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for forget gate, shape (hidden_size, hidden_size)
    pub weight_hh_f: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for cell gate, shape (hidden_size, input_size)
    pub weight_ih_g: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for cell gate, shape (hidden_size, hidden_size)
    pub weight_hh_g: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for output gate, shape (hidden_size, input_size)
    pub weight_ih_o: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for output gate, shape (hidden_size, hidden_size)
    pub weight_hh_o: Tensor<T, CpuBackend>,
    /// Input-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_ih_i: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_hh_i: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_ih_f: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_hh_f: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_ih_g: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_hh_g: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_ih_o: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_hh_o: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform + std::ops::AddAssign + std::iter::Sum> Lstm<T> {
    /// Create a new LSTM layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for LSTM weights
        let bound = (6.0 / (input_size + hidden_size) as f64).sqrt();

        // Create weight matrices for all gates
        let create_weight = |rng: &mut ThreadRng, shape: Vec<usize>| -> Tensor<T, CpuBackend> {
            let data: Vec<T> = (0..shape.iter().product::<usize>())
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as Dtype>::from_f64(val).unwrap()
                })
                .collect();
            Tensor::from_vec(CpuBackend::default(), data, shape).unwrap()
        };

        let create_bias = |rng: &mut ThreadRng, size: usize| -> Option<Tensor<T, CpuBackend>> {
            let data: Vec<T> = (0..size)
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as Dtype>::from_f64(val).unwrap()
                })
                .collect();
            Some(Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap())
        };

        Self {
            weight_ih_i: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_i: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            weight_ih_f: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_f: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            weight_ih_g: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_g: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            weight_ih_o: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_o: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            bias_ih_i: create_bias(&mut rng, hidden_size),
            bias_hh_i: create_bias(&mut rng, hidden_size),
            bias_ih_f: create_bias(&mut rng, hidden_size),
            bias_hh_f: create_bias(&mut rng, hidden_size),
            bias_ih_g: create_bias(&mut rng, hidden_size),
            bias_hh_g: create_bias(&mut rng, hidden_size),
            bias_ih_o: create_bias(&mut rng, hidden_size),
            bias_hh_o: create_bias(&mut rng, hidden_size),
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through the LSTM
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (seq_len, batch_size, input_size)
    /// * `h_0` - Initial hidden state of shape (batch_size, hidden_size)
    /// * `c_0` - Initial cell state of shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Tuple of (output, (final_hidden_state, final_cell_state))
    pub fn forward(
        &self,
        input: &Tensor<T, CpuBackend>,
        h_0: Option<&Tensor<T, CpuBackend>>,
        c_0: Option<&Tensor<T, CpuBackend>>,
    ) -> Result<(Tensor<T, CpuBackend>, (Tensor<T, CpuBackend>, Tensor<T, CpuBackend>)), NNError> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        // Initialize states
        let mut h_current = h_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]).unwrap_grad());
        let mut c_current = c_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]).unwrap_grad());

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let timestep_input = input.slice(&[Slice::Range(t, t + 1)])?;
            // Reshape from [1, batch_size, input_size] to [batch_size, input_size]
            let timestep_input_reshaped = timestep_input.reshape(vec![batch_size, self.input_size])?;
            let (h_new, c_new) = self.lstm_cell_forward(
                &timestep_input_reshaped,
                &h_current,
                &c_current,
            )?;
            h_current = h_new.clone();
            c_current = c_new.clone();
            outputs.push(h_current.clone());
        }

        // Stack outputs
        let output = if outputs.is_empty() {
            Tensor::zeros(vec![0, batch_size, self.hidden_size]).unwrap_grad()
        } else {
            // Concatenate along sequence dimension
            let mut expanded = Vec::new();
            for out in &outputs {
                expanded.push(out.unsqueeze(0)?);
            }
            let expanded_refs: Vec<&Tensor<T, CpuBackend>> = expanded.iter().collect();
            tensor_cat(&expanded_refs, 0)?
        };

        Ok((output, (h_current, c_current)))
    }

    /// Single LSTM cell forward pass
    fn lstm_cell_forward(
        &self,
        input: &Tensor<T, CpuBackend>,
        h_prev: &Tensor<T, CpuBackend>,
        c_prev: &Tensor<T, CpuBackend>,
    ) -> Result<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>), NNError> {
        // Input gate: i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_ii + b_hi)
        let i_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_i,
            &self.weight_hh_i,
            self.bias_ih_i.as_ref(),
            self.bias_hh_i.as_ref(),
        )?;

        // Forget gate: f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_if + b_hf)
        let f_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_f,
            &self.weight_hh_f,
            self.bias_ih_f.as_ref(),
            self.bias_hh_f.as_ref(),
        )?;

        // Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_ig + b_hg)
        let g_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_g,
            &self.weight_hh_g,
            self.bias_ih_g.as_ref(),
            self.bias_hh_g.as_ref(),
        )?;
        let g_gate = g_gate.tanh()?;

        // Output gate: o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_io + b_ho)
        let o_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_o,
            &self.weight_hh_o,
            self.bias_ih_o.as_ref(),
            self.bias_hh_o.as_ref(),
        )?;

        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let c_new = (&(&f_gate * c_prev)? + &(&i_gate * &g_gate)?)?;

        // Hidden state: h_t = o_t * tanh(c_t)
        let c_tanh = c_new.tanh()?;
        let h_new = (&o_gate * &c_tanh)?;

        Ok((h_new, c_new))
    }

    /// Compute a single gate computation
    fn compute_gate(
        &self,
        input: &Tensor<T, CpuBackend>,
        h_prev: &Tensor<T, CpuBackend>,
        weight_ih: &Tensor<T, CpuBackend>,
        weight_hh: &Tensor<T, CpuBackend>,
        bias_ih: Option<&Tensor<T, CpuBackend>>,
        bias_hh: Option<&Tensor<T, CpuBackend>>,
    ) -> Result<Tensor<T, CpuBackend>, NNError> {
        // x_contrib = input @ weight_ih.T
        let x_contrib = input.matmul(&weight_ih.t()?)?;

        // h_contrib = h_prev @ weight_hh.T
        let h_contrib = h_prev.matmul(&weight_hh.t()?)?;

        let mut gate = (&x_contrib + &h_contrib).unwrap();

        // Add biases
        if let Some(bias_ih) = bias_ih {
            let _bias_broadcast = bias_ih.reshape(vec![1, self.hidden_size])?;
            gate = (&x_contrib + &h_contrib).unwrap();
        }
        if let Some(bias_hh) = bias_hh {
            let _bias_broadcast = bias_hh.reshape(vec![1, self.hidden_size])?;
            gate = (&x_contrib + &h_contrib).unwrap();
        }

        // Apply sigmoid activation
        Ok(gate.sigmoid()?)
    }

    pub fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = vec![
            &self.weight_ih_i,
            &self.weight_hh_i,
            &self.weight_ih_f,
            &self.weight_hh_f,
            &self.weight_ih_g,
            &self.weight_hh_g,
            &self.weight_ih_o,
            &self.weight_hh_o,
        ];

        // Add biases if present
        if let Some(ref bias) = self.bias_ih_i {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_i {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_f {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_f {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_g {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_g {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_o {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_o {
            params.push(bias);
        }

        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = vec![
            &mut self.weight_ih_i,
            &mut self.weight_hh_i,
            &mut self.weight_ih_f,
            &mut self.weight_hh_f,
            &mut self.weight_ih_g,
            &mut self.weight_hh_g,
            &mut self.weight_ih_o,
            &mut self.weight_hh_o,
        ];

        // Add biases if present
        if let Some(ref mut bias) = self.bias_ih_i {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_i {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_f {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_f {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_g {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_g {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_o {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_o {
            params.push(bias);
        }

        params
    }
}

impl<T: FloatDtype> fmt::Display for Lstm<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lstm {{ input_size: {}, hidden_size: {} }}", self.input_size, self.hidden_size)
    }
}

/// LSTMCell (Long Short-Term Memory Cell)
///
/// A single LSTM cell that processes one timestep at a time.
/// Compatible with PyTorch's `torch.nn.LSTMCell`.
#[derive(Debug, Clone)]
pub struct LstmCell<T: FloatDtype> {
    /// Input-to-hidden weights for input gate, shape (hidden_size, input_size)
    pub weight_ih_i: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for input gate, shape (hidden_size, hidden_size)
    pub weight_hh_i: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for forget gate, shape (hidden_size, input_size)
    pub weight_ih_f: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for forget gate, shape (hidden_size, hidden_size)
    pub weight_hh_f: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for cell gate, shape (hidden_size, input_size)
    pub weight_ih_g: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for cell gate, shape (hidden_size, hidden_size)
    pub weight_hh_g: Tensor<T, CpuBackend>,
    /// Input-to-hidden weights for output gate, shape (hidden_size, input_size)
    pub weight_ih_o: Tensor<T, CpuBackend>,
    /// Hidden-to-hidden weights for output gate, shape (hidden_size, hidden_size)
    pub weight_hh_o: Tensor<T, CpuBackend>,
    /// Input-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_ih_i: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_hh_i: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_ih_f: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_hh_f: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_ih_g: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_hh_g: Option<Tensor<T, CpuBackend>>,
    /// Input-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_ih_o: Option<Tensor<T, CpuBackend>>,
    /// Hidden-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_hh_o: Option<Tensor<T, CpuBackend>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> LstmCell<T> {
    /// Create a new LSTMCell
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for LSTM weights
        let bound = (6.0 / (input_size + hidden_size) as f64).sqrt();

        // Create weight matrices for all gates
        let create_weight = |rng: &mut ThreadRng, shape: Vec<usize>| -> Tensor<T, CpuBackend> {
            let data: Vec<T> = (0..shape.iter().product::<usize>())
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as Dtype>::from_f64(val).unwrap()
                })
                .collect();
            Tensor::from_vec(CpuBackend::default(), data, shape).unwrap()
        };

        let create_bias = |rng: &mut ThreadRng, size: usize| -> Option<Tensor<T, CpuBackend>> {
            let data: Vec<T> = (0..size)
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as Dtype>::from_f64(val).unwrap()
                })
                .collect();
            Some(Tensor::from_vec(CpuBackend::default(), data, vec![size]).unwrap())
        };

        Self {
            weight_ih_i: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_i: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            weight_ih_f: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_f: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            weight_ih_g: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_g: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            weight_ih_o: create_weight(&mut rng, vec![hidden_size, input_size]),
            weight_hh_o: create_weight(&mut rng, vec![hidden_size, hidden_size]),
            bias_ih_i: create_bias(&mut rng, hidden_size),
            bias_hh_i: create_bias(&mut rng, hidden_size),
            bias_ih_f: create_bias(&mut rng, hidden_size),
            bias_hh_f: create_bias(&mut rng, hidden_size),
            bias_ih_g: create_bias(&mut rng, hidden_size),
            bias_hh_g: create_bias(&mut rng, hidden_size),
            bias_ih_o: create_bias(&mut rng, hidden_size),
            bias_hh_o: create_bias(&mut rng, hidden_size),
            input_size,
            hidden_size,
        }
    }

    /// Forward pass through a single LSTM cell
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, input_size)
    /// * `hx` - Hidden state from previous timestep, shape (batch_size, hidden_size)
    /// * `cx` - Cell state from previous timestep, shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Tuple of (hidden_state, cell_state) for this timestep
    pub fn forward(
        &self,
        input: &Tensor<T, CpuBackend>,
        hx: Option<&Tensor<T, CpuBackend>>,
        cx: Option<&Tensor<T, CpuBackend>>,
    ) -> Result<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>), NNError> {
        let batch_size = input.shape()[0];

        let h_prev = hx
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]).unwrap_grad());
        let c_prev = cx
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]).unwrap_grad());

        // Reshape input for gate computation
        let input_reshaped = input.reshape(vec![batch_size, self.input_size])?;

        self.lstm_cell_forward(&input_reshaped, &h_prev, &c_prev)
    }

    /// Internal LSTM cell forward computation
    fn lstm_cell_forward(
        &self,
        input: &Tensor<T, CpuBackend>,
        h_prev: &Tensor<T, CpuBackend>,
        c_prev: &Tensor<T, CpuBackend>,
    ) -> Result<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>), NNError> {
        // Input gate: i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_ii + b_hi)
        let i_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_i,
            &self.weight_hh_i,
            self.bias_ih_i.as_ref(),
            self.bias_hh_i.as_ref(),
        )?;

        // Forget gate: f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_if + b_hf)
        let f_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_f,
            &self.weight_hh_f,
            self.bias_ih_f.as_ref(),
            self.bias_hh_f.as_ref(),
        )?;

        // Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_ig + b_hg)
        let g_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_g,
            &self.weight_hh_g,
            self.bias_ih_g.as_ref(),
            self.bias_hh_g.as_ref(),
        )?;
        let g_gate = g_gate.tanh()?;

        // Output gate: o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_io + b_ho)
        let o_gate = self.compute_gate(
            input,
            h_prev,
            &self.weight_ih_o,
            &self.weight_hh_o,
            self.bias_ih_o.as_ref(),
            self.bias_hh_o.as_ref(),
        )?;

        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let c_new = (&(&f_gate * c_prev)? + &(&i_gate * &g_gate)?)?;

        // Hidden state: h_t = o_t * tanh(c_t)
        let c_tanh = c_new.tanh()?;
        let h_new = (&o_gate * &c_tanh)?;

        Ok((h_new, c_new))
    }

    /// Compute a single gate computation
    fn compute_gate(
        &self,
        input: &Tensor<T, CpuBackend>,
        h_prev: &Tensor<T, CpuBackend>,
        weight_ih: &Tensor<T, CpuBackend>,
        weight_hh: &Tensor<T, CpuBackend>,
        bias_ih: Option<&Tensor<T, CpuBackend>>,
        bias_hh: Option<&Tensor<T, CpuBackend>>,
    ) -> Result<Tensor<T, CpuBackend>, NNError> {
        // x_contrib = input @ weight_ih.T
        let x_contrib = input.matmul(&weight_ih.t()?)?;

        // h_contrib = h_prev @ weight_hh.T
        let h_contrib = h_prev.matmul(&weight_hh.t()?)?;

        let mut gate = (&x_contrib + &h_contrib).unwrap();

        // Add biases
        if let Some(bias_ih) = bias_ih {
            let _bias_broadcast = bias_ih.reshape(vec![1, self.hidden_size])?;
            gate = (&x_contrib + &h_contrib).unwrap();
        }
        if let Some(bias_hh) = bias_hh {
            let _bias_broadcast = bias_hh.reshape(vec![1, self.hidden_size])?;
            gate = (&x_contrib + &h_contrib).unwrap();
        }

        // Apply sigmoid activation
        Ok(gate.sigmoid()?)
    }

    pub fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        let mut params = vec![
            &self.weight_ih_i,
            &self.weight_hh_i,
            &self.weight_ih_f,
            &self.weight_hh_f,
            &self.weight_ih_g,
            &self.weight_hh_g,
            &self.weight_ih_o,
            &self.weight_hh_o,
        ];

        // Add biases if present
        if let Some(ref bias) = self.bias_ih_i {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_i {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_f {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_f {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_g {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_g {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_ih_o {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_hh_o {
            params.push(bias);
        }

        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        let mut params = vec![
            &mut self.weight_ih_i,
            &mut self.weight_hh_i,
            &mut self.weight_ih_f,
            &mut self.weight_hh_f,
            &mut self.weight_ih_g,
            &mut self.weight_hh_g,
            &mut self.weight_ih_o,
            &mut self.weight_hh_o,
        ];

        // Add biases if present
        if let Some(ref mut bias) = self.bias_ih_i {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_i {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_f {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_f {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_g {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_g {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_ih_o {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_hh_o {
            params.push(bias);
        }

        params
    }
}

impl<T: FloatDtype> fmt::Display for LstmCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LstmCell {{ input_size: {}, hidden_size: {} }}", self.input_size, self.hidden_size)
    }
}


