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

use crate::{functional, Result};
use coeus_backend::{Backend, CpuBackend};
use coeus_tensor::{indexing::Indexing, ops::reduction, FloatDtype, Tensor};
use rand::prelude::*;

/// Type alias for LSTM forward pass return value to reduce type complexity
pub type LstmOutput<T, B> = (Tensor<T, B>, (Tensor<T, B>, Tensor<T, B>));

/// LSTM (Long Short-Term Memory) layer
///
/// Implements an LSTM cell with input, forget, output, and cell gates.
/// Compatible with PyTorch's `torch.nn.LSTM`.
#[derive(Debug, Clone)]
pub struct Lstm<T: FloatDtype, B: Backend<T> + Clone = CpuBackend> {
    /// Input-to-hidden weights for input gate, shape (hidden_size, input_size)
    pub weight_ih_i: Tensor<T, B>,
    /// Hidden-to-hidden weights for input gate, shape (hidden_size, hidden_size)
    pub weight_hh_i: Tensor<T, B>,
    /// Input-to-hidden weights for forget gate, shape (hidden_size, input_size)
    pub weight_ih_f: Tensor<T, B>,
    /// Hidden-to-hidden weights for forget gate, shape (hidden_size, hidden_size)
    pub weight_hh_f: Tensor<T, B>,
    /// Input-to-hidden weights for cell gate, shape (hidden_size, input_size)
    pub weight_ih_g: Tensor<T, B>,
    /// Hidden-to-hidden weights for cell gate, shape (hidden_size, hidden_size)
    pub weight_hh_g: Tensor<T, B>,
    /// Input-to-hidden weights for output gate, shape (hidden_size, input_size)
    pub weight_ih_o: Tensor<T, B>,
    /// Hidden-to-hidden weights for output gate, shape (hidden_size, hidden_size)
    pub weight_hh_o: Tensor<T, B>,
    /// Input-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_ih_i: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_hh_i: Option<Tensor<T, B>>,
    /// Input-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_ih_f: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_hh_f: Option<Tensor<T, B>>,
    /// Input-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_ih_g: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_hh_g: Option<Tensor<T, B>>,
    /// Input-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_ih_o: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_hh_o: Option<Tensor<T, B>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Default + Send + Sync> Lstm<T, B> {
    /// Create a new LSTM layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Lstm;
    ///
    /// let lstm = Lstm::<f32>::new(10, 20);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for LSTM weights
        let bound = (6.0 / (input_size + hidden_size) as f64).sqrt();

        // Create weight matrices for all gates
        let create_weight = |rng: &mut ThreadRng, shape: Vec<usize>| -> Tensor<T, B> {
            let data: Vec<T> = (0..shape.iter().product::<usize>())
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as coeus_tensor::Dtype>::from_f64(val).expect("f64 should convert to T")
                })
                .collect();
            Tensor::from_vec(B::default(), data, shape).unwrap()
        };

        let create_bias = |rng: &mut ThreadRng, size: usize| -> Option<Tensor<T, B>> {
            let data: Vec<T> = (0..size)
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as coeus_tensor::Dtype>::from_f64(val).expect("f64 should convert to T")
                })
                .collect();
            Some(Tensor::from_vec(B::default(), data, vec![size]).expect("tensor creation should not fail"))
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
        input: &Tensor<T, B>,
        hidden: Option<(&Tensor<T, B>, &Tensor<T, B>)>,
    ) -> Result<(Tensor<T, B>, (Tensor<T, B>, Tensor<T, B>)), crate::NNError> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        // Initialize states
        let (mut h_current, mut c_current) = if let Some((h_0, c_0)) = hidden {
            (h_0.clone(), c_0.clone())
        } else {
            (
                Tensor::<T, coeus_backend::CpuBackend>::zeros(B::default(), vec![batch_size, self.hidden_size]).expect("zeros should not fail"),
                Tensor::<T, coeus_backend::CpuBackend>::zeros(B::default(), vec![batch_size, self.hidden_size]).expect("zeros should not fail"),
            )
        };

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let timestep_input = input.slice(&[coeus_tensor::ops::indexing::Slice::range(t, t + 1)])?;
            // Reshape from [1, batch_size, input_size] to [batch_size, input_size]
            // TODO: Implement reshape method
            // let timestep_input_reshaped = timestep_input.reshape(vec![batch_size, self.input_size])?;
            let timestep_input_reshaped = timestep_input;
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
        let _output_refs: Vec<&Tensor<T, B>> = outputs.iter().collect();
        let output = if outputs.is_empty() {
            Tensor::<T, CpuBackend>::zeros(B::default(), vec![0, batch_size, self.hidden_size])
        } else {
            // TODO: Implement unsqueeze and cat operations
            // // Concatenate along sequence dimension
            // let mut expanded = Vec::new();
            // for out in &outputs {
            //     expanded.push(out.unsqueeze(0)?);
            // }
            // let expanded_refs: Vec<&Tensor<T, B>> = expanded.iter().collect();
            // reduction::cat(&expanded_refs, 0)?
            return Err(crate::NNError::InvalidInput {
                message: "unsqueeze and cat operations not yet implemented".to_string(),
            });
        };

        Ok((output?, (h_current, c_current)))
    }


    pub fn parameters(&self) -> Vec<&Tensor<T, B>> {
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

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
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

    /// Compute a single gate (input, forget, cell, or output)
    fn compute_gate(
        &self,
        input: &Tensor<T, B>,
        h_prev: &Tensor<T, B>,
        weight_ih: &Tensor<T, B>,
        weight_hh: &Tensor<T, B>,
        bias_ih: Option<&Tensor<T, B>>,
        bias_hh: Option<&Tensor<T, B>>,
    ) -> Result<Tensor<T, B>> {
        // Input contribution: W_ih * x
        let ih_contrib = weight_ih.matmul(input)?;
        // Hidden contribution: W_hh * h_prev
        let hh_contrib = weight_hh.matmul(h_prev)?;
        // Combine: W_ih * x + W_hh * h_prev
        let mut gate = (&ih_contrib + &hh_contrib)?;

        // Add biases if present
        if let Some(bias_ih) = bias_ih {
            gate = (&gate + bias_ih)?;
        }
        if let Some(bias_hh) = bias_hh {
            gate = (&gate + bias_hh)?;
        }

        // Apply sigmoid activation (for input, forget, output gates)
        functional::sigmoid(&gate)
    }

    /// Internal LSTM cell forward computation
    fn lstm_cell_forward(
        &self,
        input: &Tensor<T, B>,
        h_prev: &Tensor<T, B>,
        c_prev: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        // Input gate: i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_ii + b_hi)
        let i_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_i,
            &self.weight_hh_i,
            self.bias_ih_i.as_ref(),
            self.bias_hh_i.as_ref(),
        )?;

        // Forget gate: f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_if + b_hf)
        let f_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_f,
            &self.weight_hh_f,
            self.bias_ih_f.as_ref(),
            self.bias_hh_f.as_ref(),
        )?;

        // Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_ig + b_hg)
        let g_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_g,
            &self.weight_hh_g,
            self.bias_ih_g.as_ref(),
            self.bias_hh_g.as_ref(),
        )?;
        let g_gate = functional::tanh(&g_gate)?;

        // Output gate: o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_io + b_ho)
        let o_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_o,
            &self.weight_hh_o,
            self.bias_ih_o.as_ref(),
            self.bias_hh_o.as_ref(),
        )?;

        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let f_c = (&f_gate * c_prev)?;
        let i_g = (&i_gate * &g_gate)?;
        let c_new = (&f_c + &i_g)?;

        // Hidden state: h_t = o_t * tanh(c_t)
        let c_tanh = functional::tanh(&c_new)?;
        let h_new = (&o_gate * &c_tanh)?;

        Ok((h_new, c_new))
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Default + Send + Sync> crate::Module<T, B> for Lstm<T, B> {
    fn forward(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // For sequence processing, expect input of shape (seq_len, batch_size, input_size)
        // For now, return input unchanged (simplified implementation)
        // Full LSTM would process the sequence timestep by timestep
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        Lstm::parameters(self)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        Lstm::parameters_mut(self)
    }
}

/// LSTMCell (Long Short-Term Memory Cell)
///
/// A single LSTM cell that processes one timestep at a time.
/// Compatible with PyTorch's `torch.nn.LSTMCell`.
#[derive(Debug, Clone)]
pub struct LstmCell<T: FloatDtype, B: Backend<T> + Clone = CpuBackend> {
    /// Input-to-hidden weights for input gate, shape (hidden_size, input_size)
    pub weight_ih_i: Tensor<T, B>,
    /// Hidden-to-hidden weights for input gate, shape (hidden_size, hidden_size)
    pub weight_hh_i: Tensor<T, B>,
    /// Input-to-hidden weights for forget gate, shape (hidden_size, input_size)
    pub weight_ih_f: Tensor<T, B>,
    /// Hidden-to-hidden weights for forget gate, shape (hidden_size, hidden_size)
    pub weight_hh_f: Tensor<T, B>,
    /// Input-to-hidden weights for cell gate, shape (hidden_size, input_size)
    pub weight_ih_g: Tensor<T, B>,
    /// Hidden-to-hidden weights for cell gate, shape (hidden_size, hidden_size)
    pub weight_hh_g: Tensor<T, B>,
    /// Input-to-hidden weights for output gate, shape (hidden_size, input_size)
    pub weight_ih_o: Tensor<T, B>,
    /// Hidden-to-hidden weights for output gate, shape (hidden_size, hidden_size)
    pub weight_hh_o: Tensor<T, B>,
    /// Input-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_ih_i: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_hh_i: Option<Tensor<T, B>>,
    /// Input-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_ih_f: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_hh_f: Option<Tensor<T, B>>,
    /// Input-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_ih_g: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_hh_g: Option<Tensor<T, B>>,
    /// Input-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_ih_o: Option<Tensor<T, B>>,
    /// Hidden-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_hh_o: Option<Tensor<T, B>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Default + Send + Sync> LstmCell<T, B> {
    /// Create a new LSTMCell
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::LstmCell;
    ///
    /// let lstm_cell = LstmCell::<f32>::new(10, 20);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for LSTM weights
        let bound = (6.0 / (input_size + hidden_size) as f64).sqrt();

        // Create weight matrices for all gates
        let create_weight = |rng: &mut ThreadRng, shape: Vec<usize>| -> Tensor<T, B> {
            let data: Vec<T> = (0..shape.iter().product::<usize>())
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as coeus_tensor::Dtype>::from_f64(val).expect("f64 should convert to T")
                })
                .collect();
            Tensor::from_vec(B::default(), data, shape).unwrap()
        };

        let create_bias = |rng: &mut ThreadRng, size: usize| -> Option<Tensor<T, B>> {
            let data: Vec<T> = (0..size)
                .map(|_| {
                    let val: f64 = rng.gen_range(-bound..bound);
                    <T as coeus_tensor::Dtype>::from_f64(val).expect("f64 should convert to T")
                })
                .collect();
            Some(Tensor::from_vec(B::default(), data, vec![size]).expect("tensor creation should not fail"))
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
        input: &Tensor<T, B>,
        hx: Option<&Tensor<T, B>>,
        cx: Option<&Tensor<T, B>>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let batch_size = input.shape()[0];

        let h_prev = hx
            .cloned()
            .unwrap_or_else(|| Tensor::<T, coeus_backend::CpuBackend>::zeros(B::default(), vec![batch_size, self.hidden_size]).expect("zeros should not fail"));
        let c_prev = cx
            .cloned()
            .unwrap_or_else(|| Tensor::<T, coeus_backend::CpuBackend>::zeros(B::default(), vec![batch_size, self.hidden_size]).expect("zeros should not fail"));

        // Reshape input for gate computation
        // TODO: Implement reshape method
        // let input_reshaped = input.reshape(vec![batch_size, self.input_size])?;
        let input_reshaped = input;

        self.lstm_cell_forward_impl(&input_reshaped, &h_prev, &c_prev)
    }

    /// Compute a single gate (input, forget, cell, or output)
    fn compute_gate_cell(
        &self,
        input: &Tensor<T, B>,
        h_prev: &Tensor<T, B>,
        weight_ih: &Tensor<T, B>,
        weight_hh: &Tensor<T, B>,
        bias_ih: Option<&Tensor<T, B>>,
        bias_hh: Option<&Tensor<T, B>>,
    ) -> Result<Tensor<T, B>> {
        // Compute: gate = W_ih * input + W_hh * h_prev + bias_ih + bias_hh
        let ih_part = weight_ih.matmul(input)?;
        let hh_part = weight_hh.matmul(h_prev)?;

        let mut gate = (&ih_part + &hh_part)?;

        // Add biases if present
        if let Some(bias_ih) = bias_ih {
            gate = (&gate + bias_ih)?;
        }
        if let Some(bias_hh) = bias_hh {
            gate = (&gate + bias_hh)?;
        }

        // Apply sigmoid activation (for input, forget, output gates)
        functional::sigmoid(&gate)
    }

    /// Internal LSTM cell forward computation

    /// Internal LSTM cell forward computation
    fn lstm_cell_forward(
        &self,
        input: &Tensor<T, B>,
        h_prev: &Tensor<T, B>,
        c_prev: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        // Input gate: i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_ii + b_hi)
        let i_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_i,
            &self.weight_hh_i,
            self.bias_ih_i.as_ref(),
            self.bias_hh_i.as_ref(),
        )?;

        // Forget gate: f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_if + b_hf)
        let f_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_f,
            &self.weight_hh_f,
            self.bias_ih_f.as_ref(),
            self.bias_hh_f.as_ref(),
        )?;

        // Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_ig + b_hg)
        let g_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_g,
            &self.weight_hh_g,
            self.bias_ih_g.as_ref(),
            self.bias_hh_g.as_ref(),
        )?;
        let g_gate = functional::tanh(&g_gate)?;

        // Output gate: o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_io + b_ho)
        let o_gate = self.compute_gate_cell(
            input,
            h_prev,
            &self.weight_ih_o,
            &self.weight_hh_o,
            self.bias_ih_o.as_ref(),
            self.bias_hh_o.as_ref(),
        )?;

        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let f_c = (&f_gate * c_prev)?;
        let i_g = (&i_gate * &g_gate)?;
        let c_new = (&f_c + &i_g)?;

        // Hidden state: h_t = o_t * tanh(c_t)
        let c_tanh = functional::tanh(&c_new)?;
        let h_new = (&o_gate * &c_tanh)?;
        Ok((h_new, c_new))
    }
}
