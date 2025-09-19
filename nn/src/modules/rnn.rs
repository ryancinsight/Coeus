//! Recurrent neural network layers
//!
//! This module provides RNN, LSTM, and GRU layers for sequence processing.
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

use crate::Result;
use coeus_tensor::{
    ops::indexing::Slice,
    ops::reduction::{self},
    FloatDtype, Tensor,
};
use rand::prelude::*;

/// Type alias for LSTM forward pass return value to reduce type complexity
pub type LstmOutput<T> = (Tensor<T>, (Tensor<T>, Tensor<T>));

/// RNN (Recurrent Neural Network) layer
///
/// Implements a basic RNN cell with configurable hidden size.
/// Compatible with PyTorch's `torch.nn.RNN`.
#[derive(Debug, Clone)]
pub struct Rnn<T: FloatDtype> {
    /// Input-to-hidden weights, shape (hidden_size, input_size)
    pub weight_ih: Tensor<T>,
    /// Hidden-to-hidden weights, shape (hidden_size, hidden_size)
    pub weight_hh: Tensor<T>,
    /// Input-to-hidden bias, shape (hidden_size,)
    pub bias_ih: Option<Tensor<T>>,
    /// Hidden-to-hidden bias, shape (hidden_size,)
    pub bias_hh: Option<Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Rnn<T> {
    /// Create a new RNN layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    /// * `num_layers` - Number of RNN layers (default: 1)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Rnn;
    ///
    /// let rnn = Rnn::<f32>::new(10, 20); // Single layer unidirectional RNN
    /// ```
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for RNN weights
        let ih_bound = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let hh_bound = (6.0 / (hidden_size + hidden_size) as f64).sqrt();

        let weight_ih_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        let weight_hh_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        let weight_ih = Tensor::from_vec(weight_ih_data, vec![hidden_size, input_size]);
        let weight_hh = Tensor::from_vec(weight_hh_data, vec![hidden_size, hidden_size]);

        let bias_ih_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        let bias_hh_data: Vec<T> = (0..hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        let bias_ih = Some(Tensor::from_vec(bias_ih_data, vec![hidden_size]));
        let bias_hh = Some(Tensor::from_vec(bias_hh_data, vec![hidden_size]));

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
    /// * `h_0` - Initial hidden state of shape (batch_size, hidden_size) or (2, batch_size, hidden_size) for bidirectional
    ///
    /// # Returns
    /// Tuple of (output, final_hidden_state)
    /// - output: shape (seq_len, batch_size, hidden_size) or (seq_len, batch_size, 2*hidden_size) for bidirectional
    /// - h_n: shape (batch_size, hidden_size) or (2, batch_size, hidden_size) for bidirectional
    pub fn forward(
        &self,
        input: &Tensor<T>,
        h_0: Option<&Tensor<T>>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        // Current implementation: single-layer unidirectional RNN
        // Future enhancement: bidirectional and multi-layer support

        // Initialize hidden state if not provided
        let h_init = h_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]));

        let mut h_current = h_init.clone();
        let mut outputs = Vec::new();

        // Process each timestep in the sequence
        for t in 0..seq_len {
            // Extract input at timestep t: shape (batch_size, input_size)
            // Use slice to get the t-th element along the sequence dimension
            let slices = vec![
                Slice::range(t, t + 1), // Sequence dimension: single timestep
                Slice::all(),           // Batch dimension: all batches
                Slice::all(),           // Feature dimension: all features
            ];
            let sliced = input.slice(&slices)?;
            // After slicing, we have shape (1, batch_size, input_size)
            // We need to squeeze out the first dimension to get (batch_size, input_size)
            let input_t = if sliced.shape().len() == 3 && sliced.shape()[0] == 1 {
                // Squeeze out the sequence dimension
                sliced.reshape(vec![batch_size, self.input_size])?
            } else {
                return Err(crate::NNError::ShapeMismatch {
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

            let mut combined = (&x_contrib + &h_contrib)?;

            // Add biases if present
            if let Some(ref bias_ih) = self.bias_ih {
                // Bias broadcasting: reshape bias from (hidden_size,) to (1, hidden_size) for broadcasting
                let bias_ih_broadcast = bias_ih.reshape(vec![1, self.hidden_size])?;
                combined = (&combined + &bias_ih_broadcast)?;
            }
            if let Some(ref bias_hh) = self.bias_hh {
                // Bias broadcasting: reshape bias from (hidden_size,) to (1, hidden_size) for broadcasting
                let bias_hh_broadcast = bias_hh.reshape(vec![1, self.hidden_size])?;
                combined = (&combined + &bias_hh_broadcast)?;
            }

            // Apply tanh activation
            h_current = combined.tanh();

            // Store output for this timestep
            outputs.push(h_current.clone());
        }

        // Stack all timestep outputs to create proper sequence output
        // Each output has shape (batch_size, hidden_size)
        // We want final shape (seq_len, batch_size, hidden_size)
        let _output_tensors: Vec<&Tensor<T>> = outputs.iter().collect();

        // First, add sequence dimension to each output: (batch_size, hidden_size) -> (1, batch_size, hidden_size)
        let mut expanded_outputs = Vec::new();
        for output in &outputs {
            let expanded = output.unsqueeze(0)?; // Add sequence dimension at position 0
            expanded_outputs.push(expanded);
        }

        // Then concatenate along sequence dimension (dim 0)
        let expanded_refs: Vec<&Tensor<T>> = expanded_outputs.iter().collect();
        let output = reduction::cat(&expanded_refs, 0)?;
        let h_n = h_current;

        Ok((output, h_n))
    }

    pub fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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

/// LSTM (Long Short-Term Memory) layer
///
/// Implements an LSTM cell with input, forget, output, and cell gates.
/// Compatible with PyTorch's `torch.nn.LSTM`.
#[derive(Debug, Clone)]
pub struct Lstm<T: FloatDtype> {
    /// Input-to-hidden weights for input gate, shape (hidden_size, input_size)
    pub weight_ih_i: Tensor<T>,
    /// Hidden-to-hidden weights for input gate, shape (hidden_size, hidden_size)
    pub weight_hh_i: Tensor<T>,
    /// Input-to-hidden weights for forget gate, shape (hidden_size, input_size)
    pub weight_ih_f: Tensor<T>,
    /// Hidden-to-hidden weights for forget gate, shape (hidden_size, hidden_size)
    pub weight_hh_f: Tensor<T>,
    /// Input-to-hidden weights for cell gate, shape (hidden_size, input_size)
    pub weight_ih_g: Tensor<T>,
    /// Hidden-to-hidden weights for cell gate, shape (hidden_size, hidden_size)
    pub weight_hh_g: Tensor<T>,
    /// Input-to-hidden weights for output gate, shape (hidden_size, input_size)
    pub weight_ih_o: Tensor<T>,
    /// Hidden-to-hidden weights for output gate, shape (hidden_size, hidden_size)
    pub weight_hh_o: Tensor<T>,
    /// Input-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_ih_i: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_hh_i: Option<Tensor<T>>,
    /// Input-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_ih_f: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_hh_f: Option<Tensor<T>>,
    /// Input-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_ih_g: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_hh_g: Option<Tensor<T>>,
    /// Input-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_ih_o: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_hh_o: Option<Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
    /// Number of LSTM layers
    pub num_layers: usize,
    /// Whether the LSTM is bidirectional
    pub bidirectional: bool,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Lstm<T> {
    /// Create a new LSTM layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    /// * `num_layers` - Number of LSTM layers (default: 1)
    /// * `bidirectional` - Whether to use bidirectional LSTM (default: false)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Lstm;
    ///
    /// let lstm = Lstm::<f32>::new(10, 20, 1, false);
    /// ```
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
    ) -> Self {
        // Current implementation: single-layer unidirectional LSTM
        // Future enhancement: multi-layer and bidirectional support
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for LSTM weights
        let ih_bound = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let hh_bound = (6.0 / (hidden_size + hidden_size) as f64).sqrt();

        // Input gate weights
        let weight_ih_i_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_i_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Forget gate weights
        let weight_ih_f_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_f_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Cell gate weights
        let weight_ih_g_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_g_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Output gate weights
        let weight_ih_o_data: Vec<T> = (0..hidden_size * input_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-ih_bound..ih_bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        let weight_hh_o_data: Vec<T> = (0..hidden_size * hidden_size)
            .map(|_| {
                let val: f64 = rng.gen_range(-hh_bound..hh_bound);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Create weight tensors
        let weight_ih_i = Tensor::from_vec(weight_ih_i_data, vec![hidden_size, input_size]);
        let weight_hh_i = Tensor::from_vec(weight_hh_i_data, vec![hidden_size, hidden_size]);
        let weight_ih_f = Tensor::from_vec(weight_ih_f_data, vec![hidden_size, input_size]);
        let weight_hh_f = Tensor::from_vec(weight_hh_f_data, vec![hidden_size, hidden_size]);
        let weight_ih_g = Tensor::from_vec(weight_ih_g_data, vec![hidden_size, input_size]);
        let weight_hh_g = Tensor::from_vec(weight_hh_g_data, vec![hidden_size, hidden_size]);
        let weight_ih_o = Tensor::from_vec(weight_ih_o_data, vec![hidden_size, input_size]);
        let weight_hh_o = Tensor::from_vec(weight_hh_o_data, vec![hidden_size, hidden_size]);

        // Initialize biases (LSTM typically has forget gate bias initialized to 1)
        let bias_ih_i_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();
        let bias_hh_i_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();

        let bias_ih_f_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(1.0).unwrap()) // Forget gate bias = 1.0
            .collect();
        let bias_hh_f_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();

        let bias_ih_g_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();
        let bias_hh_g_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();

        let bias_ih_o_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();
        let bias_hh_o_data: Vec<T> = (0..hidden_size)
            .map(|_| T::from_f64(0.0).unwrap())
            .collect();

        Self {
            weight_ih_i,
            weight_hh_i,
            weight_ih_f,
            weight_hh_f,
            weight_ih_g,
            weight_hh_g,
            weight_ih_o,
            weight_hh_o,
            bias_ih_i: Some(Tensor::from_vec(bias_ih_i_data, vec![hidden_size])),
            bias_hh_i: Some(Tensor::from_vec(bias_hh_i_data, vec![hidden_size])),
            bias_ih_f: Some(Tensor::from_vec(bias_ih_f_data, vec![hidden_size])),
            bias_hh_f: Some(Tensor::from_vec(bias_hh_f_data, vec![hidden_size])),
            bias_ih_g: Some(Tensor::from_vec(bias_ih_g_data, vec![hidden_size])),
            bias_hh_g: Some(Tensor::from_vec(bias_hh_g_data, vec![hidden_size])),
            bias_ih_o: Some(Tensor::from_vec(bias_ih_o_data, vec![hidden_size])),
            bias_hh_o: Some(Tensor::from_vec(bias_hh_o_data, vec![hidden_size])),
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
        }
    }

    /// Forward pass through the LSTM
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (seq_len, batch_size, input_size)
    /// * `h_0` - Initial hidden state of shape (batch_size, hidden_size) or (num_layers, batch_size, hidden_size) for multi-layer
    /// * `c_0` - Initial cell state of shape (batch_size, hidden_size) or (num_layers, batch_size, hidden_size) for multi-layer
    ///
    /// # Returns
    /// Tuple of (output, (final_hidden_state, final_cell_state))
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        input: &Tensor<T>,
        h_0: Option<&Tensor<T>>,
        c_0: Option<&Tensor<T>>,
    ) -> Result<(Tensor<T>, (Tensor<T>, Tensor<T>))> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        // Current implementation: single-layer unidirectional LSTM
        // Future enhancement: multi-layer and bidirectional support

        // Initialize hidden and cell states
        let h_init = h_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]));
        let c_init = c_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]));

        let mut h_current = h_init.clone();
        let mut c_current = c_init.clone();
        let mut outputs = Vec::new();

        // Process each timestep in the sequence
        for t in 0..seq_len {
            let slices = vec![Slice::range(t, t + 1), Slice::all(), Slice::all()];
            let sliced = input.slice(&slices)?;
            let x_t = if sliced.shape().len() == 3 && sliced.shape()[0] == 1 {
                sliced.reshape(vec![batch_size, self.input_size])?
            } else {
                return Err(crate::NNError::ShapeMismatch {
                    expected: vec![1, batch_size, self.input_size],
                    actual: sliced.shape().to_vec(),
                });
            };

            // Input gate: i_t = σ(W_ih_i @ x_t + W_hh_i @ h_{t-1} + b_ih_i + b_hh_i)
            let i_t = self.compute_gate(
                &x_t,
                &h_current,
                &self.weight_ih_i,
                &self.weight_hh_i,
                &self.bias_ih_i,
                &self.bias_hh_i,
            )?;

            // Forget gate: f_t = σ(W_ih_f @ x_t + W_hh_f @ h_{t-1} + b_ih_f + b_hh_f)
            let f_t = self.compute_gate(
                &x_t,
                &h_current,
                &self.weight_ih_f,
                &self.weight_hh_f,
                &self.bias_ih_f,
                &self.bias_hh_f,
            )?;

            // Cell gate: g_t = tanh(W_ih_g @ x_t + W_hh_g @ h_{t-1} + b_ih_g + b_hh_g)
            let g_t = self
                .compute_gate(
                    &x_t,
                    &h_current,
                    &self.weight_ih_g,
                    &self.weight_hh_g,
                    &self.bias_ih_g,
                    &self.bias_hh_g,
                )?
                .tanh();

            // Output gate: o_t = σ(W_ih_o @ x_t + W_hh_o @ h_{t-1} + b_ih_o + b_hh_o)
            let o_t = self.compute_gate(
                &x_t,
                &h_current,
                &self.weight_ih_o,
                &self.weight_hh_o,
                &self.bias_ih_o,
                &self.bias_hh_o,
            )?;

            // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
            let f_c = (&f_t * &c_current)?;
            let i_g = (&i_t * &g_t)?;
            c_current = (&f_c + &i_g)?;

            // Hidden state: h_t = o_t * tanh(c_t)
            h_current = (&o_t * &c_current.tanh())?;

            outputs.push(h_current.clone());
        }

        // Stack outputs
        let mut expanded_outputs = Vec::new();
        for output in &outputs {
            let expanded = output.unsqueeze(0)?;
            expanded_outputs.push(expanded);
        }
        let expanded_refs: Vec<&Tensor<T>> = expanded_outputs.iter().collect();
        let output = reduction::cat(&expanded_refs, 0)?;

        Ok((output, (h_current, c_current)))
    }

    /// Compute a gate value (input, forget, or output gate)
    fn compute_gate(
        &self,
        x_t: &Tensor<T>,
        h_prev: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: &Option<Tensor<T>>,
        bias_hh: &Option<Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let weight_ih_t = weight_ih.t()?;
        let x_contrib = x_t.matmul(&weight_ih_t)?;
        let weight_hh_t = weight_hh.t()?;
        let h_contrib = h_prev.matmul(&weight_hh_t)?;
        let mut combined = (&x_contrib + &h_contrib)?;

        if let Some(ref bias_ih) = bias_ih {
            let bias_ih_broadcast = bias_ih.reshape(vec![1, self.hidden_size])?;
            combined = (&combined + &bias_ih_broadcast)?;
        }
        if let Some(ref bias_hh) = bias_hh {
            let bias_hh_broadcast = bias_hh.reshape(vec![1, self.hidden_size])?;
            combined = (&combined + &bias_hh_broadcast)?;
        }

        // Apply sigmoid activation for gates
        Ok(combined.sigmoid())
    }
}

/// GRU (Gated Recurrent Unit) layer
///
/// Implements a GRU cell with reset and update gates.
/// Compatible with PyTorch's `torch.nn.GRU`.
#[derive(Debug, Clone)]
pub struct Gru<T: FloatDtype> {
    /// Input-to-hidden weights for reset gate, shape (hidden_size, input_size)
    pub weight_ih_r: Tensor<T>,
    /// Hidden-to-hidden weights for reset gate, shape (hidden_size, hidden_size)
    pub weight_hh_r: Tensor<T>,
    /// Input-to-hidden weights for update gate, shape (hidden_size, input_size)
    pub weight_ih_z: Tensor<T>,
    /// Hidden-to-hidden weights for update gate, shape (hidden_size, hidden_size)
    pub weight_hh_z: Tensor<T>,
    /// Input-to-hidden weights for candidate hidden state, shape (hidden_size, input_size)
    pub weight_ih_n: Tensor<T>,
    /// Hidden-to-hidden weights for candidate hidden state, shape (hidden_size, hidden_size)
    pub weight_hh_n: Tensor<T>,
    /// Input-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_ih_r: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_hh_r: Option<Tensor<T>>,
    /// Input-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_ih_z: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_hh_z: Option<Tensor<T>>,
    /// Input-to-hidden bias for candidate hidden state, shape (hidden_size,)
    pub bias_ih_n: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for candidate hidden state, shape (hidden_size,)
    pub bias_hh_n: Option<Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
    /// Number of GRU layers
    pub num_layers: usize,
    /// Whether the GRU is bidirectional
    pub bidirectional: bool,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> Gru<T> {
    /// Create a new GRU layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    /// * `num_layers` - Number of GRU layers (default: 1)
    /// * `bidirectional` - Whether to use bidirectional GRU (default: false)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Gru;
    ///
    /// let gru = Gru::<f32>::new(10, 20, 1, false);
    /// ```
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization for GRU weights
        let ih_bound = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let hh_bound = (6.0 / (hidden_size + hidden_size) as f64).sqrt();

        // Create all weight matrices with proper initialization
        let weight_ih_r = Self::create_weight_matrix(hidden_size, input_size, ih_bound, &mut rng);
        let weight_hh_r = Self::create_weight_matrix(hidden_size, hidden_size, hh_bound, &mut rng);
        let weight_ih_z = Self::create_weight_matrix(hidden_size, input_size, ih_bound, &mut rng);
        let weight_hh_z = Self::create_weight_matrix(hidden_size, hidden_size, hh_bound, &mut rng);
        let weight_ih_n = Self::create_weight_matrix(hidden_size, input_size, ih_bound, &mut rng);
        let weight_hh_n = Self::create_weight_matrix(hidden_size, hidden_size, hh_bound, &mut rng);

        // Initialize biases
        let bias_ih_r = Some(Self::create_bias_vector(hidden_size, ih_bound, &mut rng));
        let bias_hh_r = Some(Self::create_bias_vector(hidden_size, hh_bound, &mut rng));
        let bias_ih_z = Some(Self::create_bias_vector(hidden_size, ih_bound, &mut rng));
        let bias_hh_z = Some(Self::create_bias_vector(hidden_size, hh_bound, &mut rng));
        let bias_ih_n = Some(Self::create_bias_vector(hidden_size, ih_bound, &mut rng));
        let bias_hh_n = Some(Self::create_bias_vector(hidden_size, hh_bound, &mut rng));

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
            num_layers,
            bidirectional,
        }
    }

    /// Helper function to create weight matrices with Xavier initialization
    fn create_weight_matrix(
        rows: usize,
        cols: usize,
        bound: f64,
        rng: &mut impl rand::Rng,
    ) -> Tensor<T> {
        let data: Vec<T> = (0..rows * cols)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        Tensor::from_vec(data, vec![rows, cols])
    }

    /// Helper function to create bias vectors
    fn create_bias_vector(size: usize, bound: f64, rng: &mut impl rand::Rng) -> Tensor<T> {
        let data: Vec<T> = (0..size)
            .map(|_| {
                let val: f64 = rng.gen_range(-bound..bound);
                T::from_f64(val).unwrap()
            })
            .collect();
        Tensor::from_vec(data, vec![size])
    }

    /// Forward pass through the GRU
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (seq_len, batch_size, input_size)
    /// * `h_0` - Initial hidden state of shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Tuple of (output, final_hidden_state)
    pub fn forward(
        &self,
        input: &Tensor<T>,
        h_0: Option<&Tensor<T>>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        // Initialize hidden state if not provided
        let h_init = h_0
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(vec![batch_size, self.hidden_size]));

        let mut h_current = h_init.clone();
        let mut outputs = Vec::new();

        // Process each timestep in the sequence
        for t in 0..seq_len {
            // Extract input at timestep t: shape (batch_size, input_size)
            // Use slice to get the t-th element along the sequence dimension
            let slices = vec![
                Slice::range(t, t + 1), // Sequence dimension: single timestep
                Slice::all(),           // Batch dimension: all batches
                Slice::all(),           // Feature dimension: all features
            ];
            let sliced = input.slice(&slices)?;
            // After slicing, we have shape (1, batch_size, input_size)
            // We need to squeeze out the first dimension to get (batch_size, input_size)
            let x_t = if sliced.shape().len() == 3 && sliced.shape()[0] == 1 {
                // Squeeze out the sequence dimension
                sliced.reshape(vec![batch_size, self.input_size])?
            } else {
                return Err(crate::NNError::ShapeMismatch {
                    expected: vec![1, batch_size, self.input_size],
                    actual: sliced.shape().to_vec(),
                });
            };

            // Reset gate: r_t = σ(W_xr @ x_t + W_hr @ h_{t-1} + b_r)
            let r_gate = self.compute_gate(
                &x_t,
                &h_current,
                &self.weight_ih_r,
                &self.weight_hh_r,
                &self.bias_ih_r,
                &self.bias_hh_r,
            )?;

            // Update gate: z_t = σ(W_xz @ x_t + W_hz @ h_{t-1} + b_z)
            let z_gate = self.compute_gate(
                &x_t,
                &h_current,
                &self.weight_ih_z,
                &self.weight_hh_z,
                &self.bias_ih_r,
                &self.bias_hh_r,
            )?;

            // New gate: n_t = tanh(W_xn @ x_t + r_t * (W_hn @ h_{t-1}) + b_n)
            let weight_ih_n_t = self.weight_ih_n.t()?;
            let weight_hh_n_t = self.weight_hh_n.t()?;
            let x_contrib = x_t.matmul(&weight_ih_n_t)?;
            let h_contrib = (&r_gate * &h_current)?.matmul(&weight_hh_n_t)?;
            let mut n_gate = (&x_contrib + &h_contrib)?;

            // Add biases for new gate
            if let Some(ref bias_ih_n) = self.bias_ih_n {
                // Bias broadcasting: reshape bias for proper broadcasting
                let bias_ih_n_broadcast = bias_ih_n.reshape(vec![1, bias_ih_n.shape()[0]])?;
                n_gate = (&n_gate + &bias_ih_n_broadcast)?;
            }
            if let Some(ref bias_hh_n) = self.bias_hh_n {
                // Bias broadcasting: reshape bias for proper broadcasting
                let bias_hh_n_broadcast = bias_hh_n.reshape(vec![1, bias_hh_n.shape()[0]])?;
                n_gate = (&n_gate + &bias_hh_n_broadcast)?;
            }

            n_gate = n_gate.tanh();

            // Hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
            let ones = Tensor::ones(vec![batch_size, self.hidden_size]);
            let one_minus_z = (&ones - &z_gate)?;
            let z_h_prev = (&z_gate * &h_current)?;
            let one_minus_z_n = (&one_minus_z * &n_gate)?;
            h_current = (&one_minus_z_n + &z_h_prev)?;

            // Store output for this timestep
            outputs.push(h_current.clone());
        }

        // Stack all timestep outputs to create proper sequence output
        // Each output has shape (batch_size, hidden_size)
        // We want final shape (seq_len, batch_size, hidden_size)
        let _output_tensors: Vec<&Tensor<T>> = outputs.iter().collect();

        // First, add sequence dimension to each output: (batch_size, hidden_size) -> (1, batch_size, hidden_size)
        let mut expanded_outputs = Vec::new();
        for output in &outputs {
            let expanded = output.unsqueeze(0)?; // Add sequence dimension at position 0
            expanded_outputs.push(expanded);
        }

        // Then concatenate along sequence dimension (dim 0)
        let expanded_refs: Vec<&Tensor<T>> = expanded_outputs.iter().collect();
        let output = reduction::cat(&expanded_refs, 0)?;
        Ok((output, h_current))
    }

    /// Helper function to compute reset and update gates
    fn compute_gate(
        &self,
        x_t: &Tensor<T>,
        h_prev: &Tensor<T>,
        w_ih: &Tensor<T>,
        w_hh: &Tensor<T>,
        bias_ih: &Option<Tensor<T>>,
        bias_hh: &Option<Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let w_ih_t = w_ih.t()?;
        let w_hh_t = w_hh.t()?;
        let x_contrib = x_t.matmul(&w_ih_t)?;
        let h_contrib = h_prev.matmul(&w_hh_t)?;
        let mut gate = (&x_contrib + &h_contrib)?;

        // Add biases if present
        if let Some(ref bias_ih) = bias_ih {
            // Bias broadcasting: reshape bias for proper broadcasting
            let bias_ih_broadcast = bias_ih.reshape(vec![1, bias_ih.shape()[0]])?;
            gate = (&gate + &bias_ih_broadcast)?;
        }
        if let Some(ref bias_hh) = bias_hh {
            // Bias broadcasting: reshape bias for proper broadcasting
            let bias_hh_broadcast = bias_hh.reshape(vec![1, bias_hh.shape()[0]])?;
            gate = (&gate + &bias_hh_broadcast)?;
        }

        // Apply sigmoid activation
        Ok(gate.sigmoid())
    }
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform> crate::Module<T> for Gru<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Check input dimensionality
        match input.ndim() {
            2 => {
                // Single timestep: (batch_size, input_size)
                // Add sequence dimension to make it (1, batch_size, input_size)
                let input_3d = input.unsqueeze(0)?;
                let (output, _) = self.forward(&input_3d, None)?;
                // Remove sequence dimension to return (batch_size, hidden_size)
                // output shape is (1, batch_size, hidden_size), we want (batch_size, hidden_size)
                let batch_size = output.shape()[1];
                let hidden_size = output.shape()[2];
                Ok(output.reshape(vec![batch_size, hidden_size])?)
            }
            3 => {
                // Sequence: (seq_len, batch_size, input_size)
                let (output, _) = self.forward(input, None)?;
                Ok(output)
            }
            _ => Err(crate::NNError::InvalidInput {
                message: format!("GRU input must be 2D or 3D, got {}D", input.ndim()),
            }),
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![
            &self.weight_ih_r,
            &self.weight_hh_r,
            &self.weight_ih_z,
            &self.weight_hh_z,
            &self.weight_ih_n,
            &self.weight_hh_n,
        ];

        if let Some(ref bias_ih_r) = self.bias_ih_r {
            params.push(bias_ih_r);
        }
        if let Some(ref bias_hh_r) = self.bias_hh_r {
            params.push(bias_hh_r);
        }
        if let Some(ref bias_ih_z) = self.bias_ih_z {
            params.push(bias_ih_z);
        }
        if let Some(ref bias_hh_z) = self.bias_hh_z {
            params.push(bias_hh_z);
        }
        if let Some(ref bias_ih_n) = self.bias_ih_n {
            params.push(bias_ih_n);
        }
        if let Some(ref bias_hh_n) = self.bias_hh_n {
            params.push(bias_hh_n);
        }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![
            &mut self.weight_ih_r,
            &mut self.weight_hh_r,
            &mut self.weight_ih_z,
            &mut self.weight_hh_z,
            &mut self.weight_ih_n,
            &mut self.weight_hh_n,
        ];

        if let Some(ref mut bias_ih_r) = self.bias_ih_r {
            params.push(bias_ih_r);
        }
        if let Some(ref mut bias_hh_r) = self.bias_hh_r {
            params.push(bias_hh_r);
        }
        if let Some(ref mut bias_ih_z) = self.bias_ih_z {
            params.push(bias_ih_z);
        }
        if let Some(ref mut bias_hh_z) = self.bias_hh_z {
            params.push(bias_hh_z);
        }
        if let Some(ref mut bias_ih_n) = self.bias_ih_n {
            params.push(bias_ih_n);
        }
        if let Some(ref mut bias_hh_n) = self.bias_hh_n {
            params.push(bias_hh_n);
        }

        params
    }
}

/// RNNCell (Recurrent Neural Network Cell)
///
/// A single RNN cell that processes one timestep at a time.
/// Compatible with PyTorch's `torch.nn.RNNCell`.
#[derive(Debug, Clone)]
pub struct RnnCell<T: FloatDtype> {
    /// Input-to-hidden weights, shape (hidden_size, input_size)
    pub weight_ih: Tensor<T>,
    /// Hidden-to-hidden weights, shape (hidden_size, hidden_size)
    pub weight_hh: Tensor<T>,
    /// Input-to-hidden bias, shape (hidden_size,)
    pub bias_ih: Option<Tensor<T>>,
    /// Hidden-to-hidden bias, shape (hidden_size,)
    pub bias_hh: Option<Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + num_traits::FromPrimitive> RnnCell<T> {
    /// Create a new RNNCell layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::RnnCell;
    ///
    /// let rnn_cell = RnnCell::<f32>::new(10, 20);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        use crate::init::Xavier;

        let init = Xavier::new();

        Self {
            weight_ih: init.initialize(&[hidden_size, input_size]).unwrap(),
            weight_hh: init.initialize(&[hidden_size, hidden_size]).unwrap(),
            bias_ih: Some(Tensor::zeros(vec![hidden_size])),
            bias_hh: Some(Tensor::zeros(vec![hidden_size])),
            input_size,
            hidden_size,
        }
    }
}

impl<T: FloatDtype> RnnCell<T> {
    /// Forward pass for a single timestep
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, input_size)
    /// * `hx` - Hidden state from previous timestep, shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Next hidden state of shape (batch_size, hidden_size)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::RnnCell;
    /// use coeus_tensor::Tensor;
    ///
    /// let rnn_cell = RnnCell::<f32>::new(10, 20);
    /// let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 0.5, 1.2, 0.8, 1.5, 0.3, 2.1, 1.8, 0.9], vec![1, 10]);
    /// let hx = Tensor::zeros(vec![1, 20]);
    ///
    /// let next_hx = rnn_cell.forward(&input, &hx).unwrap();
    /// ```
    pub fn forward(&self, input: &Tensor<T>, hx: &Tensor<T>) -> Result<Tensor<T>> {
        // h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)
        let ih_part = input.matmul(&self.weight_ih.t()?)?;
        let hh_part = hx.matmul(&self.weight_hh.t()?)?;

        let mut result = (&ih_part + &hh_part)?;

        if let Some(ref bias_ih) = self.bias_ih {
            result = (&result + &bias_ih.unsqueeze(0)?)?;
        }
        if let Some(ref bias_hh) = self.bias_hh {
            result = (&result + &bias_hh.unsqueeze(0)?)?;
        }

        Ok(result.tanh())
    }
}

impl<T: FloatDtype> crate::Module<T> for RnnCell<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For Module trait, expect concatenated [input, hx]
        if input.shape().len() != 2 || input.shape()[1] != self.input_size + self.hidden_size {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "RNNCell forward expects shape (batch_size, {}) for concatenated [input, hx], got {:?}",
                    self.input_size + self.hidden_size,
                    input.shape()
                ),
            });
        }

        let batch_size = input.shape()[0];
        use coeus_tensor::ops::indexing::Slice;
        let input_part = input.slice(&[Slice::range(0, self.input_size), Slice::range(0, batch_size)])?;
        let hx_part = input.slice(&[Slice::range(self.input_size, self.input_size + self.hidden_size), Slice::range(0, batch_size)])?;

        self.forward(&input_part, &hx_part)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight_ih, &self.weight_hh];
        if let Some(ref bias_ih) = self.bias_ih {
            params.push(bias_ih);
        }
        if let Some(ref bias_hh) = self.bias_hh {
            params.push(bias_hh);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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

/// LSTMCell (Long Short-Term Memory Cell)
///
/// A single LSTM cell that processes one timestep at a time.
/// Compatible with PyTorch's `torch.nn.LSTMCell`.
#[derive(Debug, Clone)]
pub struct LstmCell<T: FloatDtype> {
    /// Input-to-hidden weights for input gate, shape (hidden_size, input_size)
    pub weight_ih_i: Tensor<T>,
    /// Input-to-hidden weights for forget gate, shape (hidden_size, input_size)
    pub weight_ih_f: Tensor<T>,
    /// Input-to-hidden weights for cell gate, shape (hidden_size, input_size)
    pub weight_ih_g: Tensor<T>,
    /// Input-to-hidden weights for output gate, shape (hidden_size, input_size)
    pub weight_ih_o: Tensor<T>,
    /// Hidden-to-hidden weights for input gate, shape (hidden_size, hidden_size)
    pub weight_hh_i: Tensor<T>,
    /// Hidden-to-hidden weights for forget gate, shape (hidden_size, hidden_size)
    pub weight_hh_f: Tensor<T>,
    /// Hidden-to-hidden weights for cell gate, shape (hidden_size, hidden_size)
    pub weight_hh_g: Tensor<T>,
    /// Hidden-to-hidden weights for output gate, shape (hidden_size, hidden_size)
    pub weight_hh_o: Tensor<T>,
    /// Input-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_ih_i: Option<Tensor<T>>,
    /// Input-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_ih_f: Option<Tensor<T>>,
    /// Input-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_ih_g: Option<Tensor<T>>,
    /// Input-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_ih_o: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for input gate, shape (hidden_size,)
    pub bias_hh_i: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for forget gate, shape (hidden_size,)
    pub bias_hh_f: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for cell gate, shape (hidden_size,)
    pub bias_hh_g: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for output gate, shape (hidden_size,)
    pub bias_hh_o: Option<Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + num_traits::FromPrimitive> LstmCell<T> {
    /// Create a new LSTMCell layer
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
        use crate::init::Xavier;
        let init = Xavier::new();

        let weight_ih_i = init.initialize(&[hidden_size, input_size]).unwrap();
        let weight_ih_f = init.initialize(&[hidden_size, input_size]).unwrap();
        let weight_ih_g = init.initialize(&[hidden_size, input_size]).unwrap();
        let weight_ih_o = init.initialize(&[hidden_size, input_size]).unwrap();

        let weight_hh_i = init.initialize(&[hidden_size, hidden_size]).unwrap();
        let weight_hh_f = init.initialize(&[hidden_size, hidden_size]).unwrap();
        let weight_hh_g = init.initialize(&[hidden_size, hidden_size]).unwrap();
        let weight_hh_o = init.initialize(&[hidden_size, hidden_size]).unwrap();

        // Initialize biases
        let bias_ih_i = Some(Tensor::zeros(vec![hidden_size]));
        let bias_ih_f = Some(Tensor::ones(vec![hidden_size])); // Forget gate bias = 1
        let bias_ih_g = Some(Tensor::zeros(vec![hidden_size]));
        let bias_ih_o = Some(Tensor::zeros(vec![hidden_size]));

        let bias_hh_i = Some(Tensor::zeros(vec![hidden_size]));
        let bias_hh_f = Some(Tensor::zeros(vec![hidden_size]));
        let bias_hh_g = Some(Tensor::zeros(vec![hidden_size]));
        let bias_hh_o = Some(Tensor::zeros(vec![hidden_size]));

        Self {
            weight_ih_i,
            weight_ih_f,
            weight_ih_g,
            weight_ih_o,
            weight_hh_i,
            weight_hh_f,
            weight_hh_g,
            weight_hh_o,
            bias_ih_i,
            bias_ih_f,
            bias_ih_g,
            bias_ih_o,
            bias_hh_i,
            bias_hh_f,
            bias_hh_g,
            bias_hh_o,
            input_size,
            hidden_size,
        }
    }
}

impl<T: FloatDtype> LstmCell<T> {
    /// Forward pass for a single timestep
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, input_size)
    /// * `hx` - Hidden state from previous timestep, shape (batch_size, hidden_size)
    /// * `cx` - Cell state from previous timestep, shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Tuple of (next_hidden_state, next_cell_state)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::LstmCell;
    /// use coeus_tensor::Tensor;
    ///
    /// let lstm_cell = LstmCell::<f32>::new(10, 20);
    /// let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 0.5, 1.2, 0.8, 1.5, 0.3, 2.1, 1.8, 0.9], vec![1, 10]);
    /// let hx = Tensor::zeros(vec![1, 20]);
    /// let cx = Tensor::zeros(vec![1, 20]);
    ///
    /// let (next_hx, next_cx) = lstm_cell.forward(&input, &hx, &cx).unwrap();
    /// ```
    pub fn forward(&self, input: &Tensor<T>, hx: &Tensor<T>, cx: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)> {
        // Input gate: i_t = σ(W_ii * x + b_ii + W_hi * h + b_hi)
        let i_ih = input.matmul(&self.weight_ih_i.t()?)?;
        let i_hh = hx.matmul(&self.weight_hh_i.t()?)?;
        let mut i_gate = (&i_ih + &i_hh)?;
        if let Some(ref b) = self.bias_ih_i { i_gate = (&i_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_i { i_gate = (&i_gate + &b.unsqueeze(0)?)?; }
        let i_gate = i_gate.sigmoid();

        // Forget gate: f_t = σ(W_if * x + b_if + W_hf * h + b_hf)
        let f_ih = input.matmul(&self.weight_ih_f.t()?)?;
        let f_hh = hx.matmul(&self.weight_hh_f.t()?)?;
        let mut f_gate = (&f_ih + &f_hh)?;
        if let Some(ref b) = self.bias_ih_f { f_gate = (&f_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_f { f_gate = (&f_gate + &b.unsqueeze(0)?)?; }
        let f_gate = f_gate.sigmoid();

        // Cell gate: g_t = tanh(W_ig * x + b_ig + W_hg * h + b_hg)
        let g_ih = input.matmul(&self.weight_ih_g.t()?)?;
        let g_hh = hx.matmul(&self.weight_hh_g.t()?)?;
        let mut g_gate = (&g_ih + &g_hh)?;
        if let Some(ref b) = self.bias_ih_g { g_gate = (&g_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_g { g_gate = (&g_gate + &b.unsqueeze(0)?)?; }
        let g_gate = g_gate.tanh();

        // Output gate: o_t = σ(W_io * x + b_io + W_ho * h + b_ho)
        let o_ih = input.matmul(&self.weight_ih_o.t()?)?;
        let o_hh = hx.matmul(&self.weight_hh_o.t()?)?;
        let mut o_gate = (&o_ih + &o_hh)?;
        if let Some(ref b) = self.bias_ih_o { o_gate = (&o_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_o { o_gate = (&o_gate + &b.unsqueeze(0)?)?; }
        let o_gate = o_gate.sigmoid();

        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let c_new = (&(&f_gate * cx)? + &(&i_gate * &g_gate)?)?;

        // Hidden state: h_t = o_t * tanh(c_t)
        let h_new = (&o_gate * &c_new.tanh())?;

        Ok((h_new, c_new))
    }
}

impl<T: FloatDtype> crate::Module<T> for LstmCell<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For Module trait, expect concatenated [input, hx, cx]
        if input.shape().len() != 2 || input.shape()[1] != self.input_size + self.hidden_size + self.hidden_size {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "LSTMCell forward expects shape (batch_size, {}) for concatenated [input, hx, cx], got {:?}",
                    self.input_size + self.hidden_size + self.hidden_size,
                    input.shape()
                ),
            });
        }

        let batch_size = input.shape()[0];
        use coeus_tensor::ops::indexing::Slice;
        let input_part = input.slice(&[Slice::range(0, self.input_size), Slice::range(0, batch_size)])?;
        let hx_part = input.slice(&[Slice::range(self.input_size, self.input_size + self.hidden_size), Slice::range(0, batch_size)])?;
        let cx_part = input.slice(&[Slice::range(self.input_size + self.hidden_size, self.input_size + 2 * self.hidden_size), Slice::range(0, batch_size)])?;

        let (h_new, c_new) = self.forward(&input_part, &hx_part, &cx_part)?;

        // Concatenate h_new and c_new for return
        let mut result_data = Vec::with_capacity(h_new.numel() + c_new.numel());
        result_data.extend_from_slice(h_new.data());
        result_data.extend_from_slice(c_new.data());

        Ok(Tensor::from_vec(result_data, vec![batch_size, 2 * self.hidden_size]))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![
            &self.weight_ih_i, &self.weight_ih_f, &self.weight_ih_g, &self.weight_ih_o,
            &self.weight_hh_i, &self.weight_hh_f, &self.weight_hh_g, &self.weight_hh_o,
        ];

        if let Some(ref b) = self.bias_ih_i { params.push(b); }
        if let Some(ref b) = self.bias_ih_f { params.push(b); }
        if let Some(ref b) = self.bias_ih_g { params.push(b); }
        if let Some(ref b) = self.bias_ih_o { params.push(b); }
        if let Some(ref b) = self.bias_hh_i { params.push(b); }
        if let Some(ref b) = self.bias_hh_f { params.push(b); }
        if let Some(ref b) = self.bias_hh_g { params.push(b); }
        if let Some(ref b) = self.bias_hh_o { params.push(b); }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![
            &mut self.weight_ih_i, &mut self.weight_ih_f, &mut self.weight_ih_g, &mut self.weight_ih_o,
            &mut self.weight_hh_i, &mut self.weight_hh_f, &mut self.weight_hh_g, &mut self.weight_hh_o,
        ];

        if let Some(ref mut b) = self.bias_ih_i { params.push(b); }
        if let Some(ref mut b) = self.bias_ih_f { params.push(b); }
        if let Some(ref mut b) = self.bias_ih_g { params.push(b); }
        if let Some(ref mut b) = self.bias_ih_o { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_i { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_f { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_g { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_o { params.push(b); }

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
    pub weight_ih_r: Tensor<T>,
    /// Input-to-hidden weights for update gate, shape (hidden_size, input_size)
    pub weight_ih_z: Tensor<T>,
    /// Input-to-hidden weights for new gate, shape (hidden_size, input_size)
    pub weight_ih_n: Tensor<T>,
    /// Hidden-to-hidden weights for reset gate, shape (hidden_size, hidden_size)
    pub weight_hh_r: Tensor<T>,
    /// Hidden-to-hidden weights for update gate, shape (hidden_size, hidden_size)
    pub weight_hh_z: Tensor<T>,
    /// Hidden-to-hidden weights for new gate, shape (hidden_size, hidden_size)
    pub weight_hh_n: Tensor<T>,
    /// Input-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_ih_r: Option<Tensor<T>>,
    /// Input-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_ih_z: Option<Tensor<T>>,
    /// Input-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_ih_n: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for reset gate, shape (hidden_size,)
    pub bias_hh_r: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for update gate, shape (hidden_size,)
    pub bias_hh_z: Option<Tensor<T>>,
    /// Hidden-to-hidden bias for new gate, shape (hidden_size,)
    pub bias_hh_n: Option<Tensor<T>>,
    /// Number of input features
    pub input_size: usize,
    /// Number of hidden features
    pub hidden_size: usize,
}

impl<T: FloatDtype + num_traits::FromPrimitive> GruCell<T> {
    /// Create a new GRUCell layer
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden features
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::GruCell;
    ///
    /// let gru_cell = GruCell::<f32>::new(10, 20);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        use crate::init::Xavier;
        let init = Xavier::new();

        let weight_ih_r = init.initialize(&[hidden_size, input_size]).unwrap();
        let weight_ih_z = init.initialize(&[hidden_size, input_size]).unwrap();
        let weight_ih_n = init.initialize(&[hidden_size, input_size]).unwrap();

        let weight_hh_r = init.initialize(&[hidden_size, hidden_size]).unwrap();
        let weight_hh_z = init.initialize(&[hidden_size, hidden_size]).unwrap();
        let weight_hh_n = init.initialize(&[hidden_size, hidden_size]).unwrap();

        let bias_ih_r = Some(Tensor::zeros(vec![hidden_size]));
        let bias_ih_z = Some(Tensor::zeros(vec![hidden_size]));
        let bias_ih_n = Some(Tensor::zeros(vec![hidden_size]));
        let bias_hh_r = Some(Tensor::zeros(vec![hidden_size]));
        let bias_hh_z = Some(Tensor::zeros(vec![hidden_size]));
        let bias_hh_n = Some(Tensor::zeros(vec![hidden_size]));

        Self {
            weight_ih_r,
            weight_ih_z,
            weight_ih_n,
            weight_hh_r,
            weight_hh_z,
            weight_hh_n,
            bias_ih_r,
            bias_ih_z,
            bias_ih_n,
            bias_hh_r,
            bias_hh_z,
            bias_hh_n,
            input_size,
            hidden_size,
        }
    }
}

impl<T: FloatDtype> GruCell<T> {
    /// Forward pass for a single timestep
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, input_size)
    /// * `hx` - Hidden state from previous timestep, shape (batch_size, hidden_size)
    ///
    /// # Returns
    /// Next hidden state of shape (batch_size, hidden_size)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::GruCell;
    /// use coeus_tensor::Tensor;
    ///
    /// let gru_cell = GruCell::<f32>::new(10, 20);
    /// let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 0.5, 1.2, 0.8, 1.5, 0.3, 2.1, 1.8, 0.9], vec![1, 10]);
    /// let hx = Tensor::zeros(vec![1, 20]);
    ///
    /// let next_hx = gru_cell.forward(&input, &hx).unwrap();
    /// ```
    pub fn forward(&self, input: &Tensor<T>, hx: &Tensor<T>) -> Result<Tensor<T>> {
        // Reset gate: r_t = σ(W_ir * x + b_ir + W_hr * h + b_hr)
        let r_ih = input.matmul(&self.weight_ih_r.t()?)?;
        let r_hh = hx.matmul(&self.weight_hh_r.t()?)?;
        let mut r_gate = (&r_ih + &r_hh)?;
        if let Some(ref b) = self.bias_ih_r { r_gate = (&r_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_r { r_gate = (&r_gate + &b.unsqueeze(0)?)?; }
        let r_gate = r_gate.sigmoid();

        // Update gate: z_t = σ(W_iz * x + b_iz + W_hz * h + b_hz)
        let z_ih = input.matmul(&self.weight_ih_z.t()?)?;
        let z_hh = hx.matmul(&self.weight_hh_z.t()?)?;
        let mut z_gate = (&z_ih + &z_hh)?;
        if let Some(ref b) = self.bias_ih_z { z_gate = (&z_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_z { z_gate = (&z_gate + &b.unsqueeze(0)?)?; }
        let z_gate = z_gate.sigmoid();

        // New gate: n_t = tanh(W_in * x + b_in + r_t * (W_hn * h + b_hn))
        let n_ih = input.matmul(&self.weight_ih_n.t()?)?;
        let n_hh = hx.matmul(&self.weight_hh_n.t()?)?;
        let mut n_gate = (&n_ih + &(&r_gate * &n_hh)?)?;
        if let Some(ref b) = self.bias_ih_n { n_gate = (&n_gate + &b.unsqueeze(0)?)?; }
        if let Some(ref b) = self.bias_hh_n { n_gate = (&n_gate + &b.unsqueeze(0)?)?; }
        let n_gate = n_gate.tanh();

        // Hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        let one_minus_z = (&Tensor::ones(z_gate.shape().to_vec()) - &z_gate)?;
        let h_new = (&(&one_minus_z * &n_gate)? + &(&z_gate * hx)?)?;

        Ok(h_new)
    }
}

impl<T: FloatDtype> crate::Module<T> for GruCell<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // For Module trait, expect concatenated [input, hx]
        if input.shape().len() != 2 || input.shape()[1] != self.input_size + self.hidden_size {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "GRUCell forward expects shape (batch_size, {}) for concatenated [input, hx], got {:?}",
                    self.input_size + self.hidden_size,
                    input.shape()
                ),
            });
        }

        let batch_size = input.shape()[0];
        use coeus_tensor::ops::indexing::Slice;
        let input_part = input.slice(&[Slice::range(0, self.input_size), Slice::range(0, batch_size)])?;
        let hx_part = input.slice(&[Slice::range(self.input_size, self.input_size + self.hidden_size), Slice::range(0, batch_size)])?;

        self.forward(&input_part, &hx_part)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![
            &self.weight_ih_r, &self.weight_ih_z, &self.weight_ih_n,
            &self.weight_hh_r, &self.weight_hh_z, &self.weight_hh_n,
        ];

        if let Some(ref b) = self.bias_ih_r { params.push(b); }
        if let Some(ref b) = self.bias_ih_z { params.push(b); }
        if let Some(ref b) = self.bias_ih_n { params.push(b); }
        if let Some(ref b) = self.bias_hh_r { params.push(b); }
        if let Some(ref b) = self.bias_hh_z { params.push(b); }
        if let Some(ref b) = self.bias_hh_n { params.push(b); }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![
            &mut self.weight_ih_r, &mut self.weight_ih_z, &mut self.weight_ih_n,
            &mut self.weight_hh_r, &mut self.weight_hh_z, &mut self.weight_hh_n,
        ];

        if let Some(ref mut b) = self.bias_ih_r { params.push(b); }
        if let Some(ref mut b) = self.bias_ih_z { params.push(b); }
        if let Some(ref mut b) = self.bias_ih_n { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_r { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_z { params.push(b); }
        if let Some(ref mut b) = self.bias_hh_n { params.push(b); }

        params
    }
}
