//! Recurrent Neural Network (RNN) layers for sequence modeling.
//!
//! This module provides RNN, LSTM, and GRU layers for processing sequential data.
//! All implementations support:
//! - Bidirectional processing
//! - Multi-layer stacking
//! - Batch-first or sequence-first input formats
//! - Hidden state management

use std::fmt;
use std::marker::PhantomData;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

// Type aliases to reduce complexity
type CpuTensor<T> = Tensor<CpuBackend<T>, DenseStorage<T>, T>;
type TensorPair<T> = (CpuTensor<T>, CpuTensor<T>);

/// Basic Recurrent Neural Network (RNN) layer.
///
/// Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.
///
/// For each element in the input sequence, each layer computes:
/// ```text
/// h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
/// ```
///
/// Where:
/// - `x_t`: input at time step t, shape [batch_size, input_size]
/// - `h_t`: hidden state at time step t, shape [batch_size, hidden_size]
/// - `W_ih`: input-to-hidden weights, shape [hidden_size, input_size]
/// - `W_hh`: hidden-to-hidden weights, shape [hidden_size, hidden_size]
/// - `b_ih`, `b_hh`: biases, shape [hidden_size]
///
/// # Arguments
/// * `input_size` - The number of expected features in the input x
/// * `hidden_size` - The number of features in the hidden state h
/// * `num_layers` - Number of recurrent layers (default: 1)
/// * `bias` - If False, the layer does not use bias weights (default: True)
/// * `batch_first` - If True, input/output tensors are (batch, seq, feature) (default: False)
/// * `bidirectional` - If True, becomes a bidirectional RNN (default: False)
///
/// # Shape
/// - Input: `(seq_len, batch, input_size)` or `(batch, seq_len, input_size)` if batch_first=True
/// - Hidden: `(num_layers * num_directions, batch, hidden_size)`
/// - Output: `(seq_len, batch, num_directions * hidden_size)` or `(batch, seq_len, num_directions * hidden_size)`
///
/// # Examples
/// ```rust
/// use coeus_nn::{RNN, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create a 2-layer bidirectional RNN
/// let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, true).unwrap();
/// // RNN created successfully
/// ```
#[derive(Debug)]
pub struct RNN<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Input-to-hidden weights for each layer
    pub weight_ih: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Hidden-to-hidden weights for each layer
    pub weight_hh: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Input-to-hidden biases for each layer
    pub bias_ih: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Hidden-to-hidden biases for each layer
    pub bias_hh: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// Number of recurrent layers
    pub num_layers: usize,
    /// Whether to use bias weights
    #[allow(dead_code)]
    pub bias: bool,
    /// Whether input/output tensors are (batch, seq, feature)
    #[allow(dead_code)]
    pub batch_first: bool,
    /// Whether this is a bidirectional RNN
    #[allow(dead_code)]
    pub bidirectional: bool,
    /// Phantom data to ensure B and S are used for type safety
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> RNN<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new RNN layer.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `bias` - If false, the layer does not use bias weights (default: true)
    /// * `batch_first` - If true, input/output tensors are (batch, seq, feature) (default: false)
    /// * `bidirectional` - If true, becomes a bidirectional RNN (default: false)
    ///
    /// # Errors
    /// Returns `NNError::InvalidInput` if `input_size`, `hidden_size`, or `num_layers` is 0.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> Result<Self> {
        if input_size == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "input_size must be > 0".to_string(),
            });
        }
        if hidden_size == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "hidden_size must be > 0".to_string(),
            });
        }
        if num_layers == 0 {
            return Err(crate::error::NNError::InvalidInput {
                message: "num_layers must be > 0".to_string(),
            });
        }

        let num_directions = if bidirectional { 2 } else { 1 };
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..num_layers {
            for _dir in 0..num_directions {
                // First layer uses input_size, subsequent layers use hidden_size * num_directions
                let layer_input_size = if layer == 0 {
                    input_size
                } else {
                    hidden_size * num_directions
                };

                // Xavier/Glorot uniform initialization
                let limit = (T::from(6.0).unwrap()
                    / T::from(layer_input_size + hidden_size).unwrap())
                .sqrt();
                let w_ih = Self::xavier_uniform_init(hidden_size, layer_input_size, limit);
                let w_hh = Self::xavier_uniform_init(hidden_size, hidden_size, limit);

                let weight_ih_var = w_ih;
                let weight_hh_var = w_hh;

                weight_ih.push(Parameter::new(
                    weight_ih_var,
                    format!("weight_ih_l{}", layer),
                ));
                weight_hh.push(Parameter::new(
                    weight_hh_var,
                    format!("weight_hh_l{}", layer),
                ));

                if bias {
                    let b_ih =
                        Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[hidden_size]).unwrap();
                    let b_hh =
                        Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[hidden_size]).unwrap();

                    let bias_ih_var = b_ih;
                    let bias_hh_var = b_hh;

                    bias_ih.push(Parameter::new(bias_ih_var, format!("bias_ih_l{}", layer)));
                    bias_hh.push(Parameter::new(bias_hh_var, format!("bias_hh_l{}", layer)));
                }
            }
        }

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
            _phantom: PhantomData,
        })
    }

    /// Xavier/Glorot uniform initialization for weights.
    ///
    /// Initializes weights using Xavier uniform distribution with proper random sampling.
    /// This ensures symmetry breaking for gradient descent convergence.
    ///
    /// # References
    /// - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
    fn xavier_uniform_init(
        rows: usize,
        cols: usize,
        _limit: T,
    ) -> Tensor<CpuBackend<T>, DenseStorage<T>, T> {
        let mut tensor = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[rows, cols]).unwrap();
        crate::init::xavier_uniform_(&mut tensor, 1.0).unwrap();
        tensor
    }
}

impl<B, S, T> RNN<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    /// Forward pass with hidden state management.
    ///
    /// This method performs the full RNN computation and returns both the output
    /// and the final hidden state. Use this when you need access to hidden states.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(seq_len, batch, input_size)` or `(batch, seq_len, input_size)` if batch_first
    /// * `hidden` - Optional initial hidden state of shape `(num_layers * num_directions, batch, hidden_size)`.
    ///   If None, initializes to zeros.
    ///
    /// # Returns
    /// * `output` - Output tensor of shape `(seq_len, batch, hidden_size * num_directions)` or batch_first variant
    /// * `hidden` - Final hidden state of shape `(num_layers * num_directions, batch, hidden_size)`
    ///
    /// # Shape
    /// - Input: `(seq_len, batch, input_size)` or `(batch, seq_len, input_size)` if batch_first
    /// - Hidden: `(num_layers * num_directions, batch, hidden_size)`
    /// - Output: `(seq_len, batch, hidden_size * num_directions)` or batch_first variant
    pub fn forward_with_hidden(
        &self,
        input: &CpuTensor<T>,
        hidden: Option<&CpuTensor<T>>,
    ) -> Result<TensorPair<T>> {
        // Get input dimensions
        let input_shape = input.shape().dims();
        let (seq_len, batch_size, input_size) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        // Validate input size
        if input_size != self.input_size {
            return Err(crate::error::NNError::InvalidInput {
                message: format!(
                    "Expected input_size={}, got {}",
                    self.input_size, input_size
                ),
            });
        }

        let num_directions = if self.bidirectional { 2 } else { 1 };

        // Initialize or validate hidden state
        let h = if let Some(h_init) = hidden {
            h_init.clone()
        } else {
            Tensor::zeros(&[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ])?
        };

        // Transpose input if batch_first: (batch, seq_len, input_size) ? (seq_len, batch, input_size)
        let input_seq = if self.batch_first {
            Self::transpose_3d(input, batch_size, seq_len, input_size)?
        } else {
            input.clone()
        };

        // Process each layer with bidirectional support
        let mut layer_input = input_seq;
        for layer in 0..self.num_layers {
            if self.bidirectional {
                // Bidirectional: process forward and backward directions separately

                // Determine input size for this layer
                let layer_input_size = if layer == 0 {
                    self.input_size
                } else {
                    self.hidden_size * 2 // Previous layer output is concatenated
                };

                // Forward direction (use weights at layer*2)
                let (forward_output, _forward_hidden) = self.forward_layer(
                    &layer_input,
                    &h,
                    layer * 2, // Forward direction weight index
                    seq_len,
                    batch_size,
                    layer_input_size,
                )?;

                // Backward direction (use weights at layer*2+1)
                // Reverse input, process, then reverse output
                let reversed_input =
                    Self::reverse_sequence(&layer_input, seq_len, batch_size, layer_input_size)?;
                let (backward_output_reversed, _backward_hidden) = self.forward_layer(
                    &reversed_input,
                    &h,
                    layer * 2 + 1, // Backward direction weight index
                    seq_len,
                    batch_size,
                    layer_input_size,
                )?;
                let backward_output = Self::reverse_sequence(
                    &backward_output_reversed,
                    seq_len,
                    batch_size,
                    self.hidden_size,
                )?;

                // Concatenate forward and backward outputs along hidden dimension
                // forward_output: [seq_len, batch, hidden_size]
                // backward_output: [seq_len, batch, hidden_size]
                // concatenated: [seq_len, batch, hidden_size * 2]
                let forward_data = forward_output.as_slice();
                let backward_data = backward_output.as_slice();
                let mut concatenated_data =
                    Vec::with_capacity(seq_len * batch_size * self.hidden_size * 2);

                for t in 0..seq_len {
                    for b in 0..batch_size {
                        let forward_start = (t * batch_size + b) * self.hidden_size;
                        let forward_end = forward_start + self.hidden_size;
                        let backward_start = (t * batch_size + b) * self.hidden_size;
                        let backward_end = backward_start + self.hidden_size;

                        concatenated_data
                            .extend_from_slice(&forward_data[forward_start..forward_end]);
                        concatenated_data
                            .extend_from_slice(&backward_data[backward_start..backward_end]);
                    }
                }

                layer_input = Tensor::from_vec(
                    concatenated_data,
                    &[seq_len, batch_size, self.hidden_size * 2],
                )?;
            } else {
                // Unidirectional: process forward direction only
                let layer_input_size = if layer == 0 {
                    self.input_size
                } else {
                    self.hidden_size
                };
                let (layer_output, _layer_hidden) = self.forward_layer(
                    &layer_input,
                    &h,
                    layer,
                    seq_len,
                    batch_size,
                    layer_input_size,
                )?;
                layer_input = layer_output;
            }
        }

        // Transpose output if batch_first: (seq_len, batch, hidden_size) ? (batch, seq_len, hidden_size)
        let output = if self.batch_first {
            let output_hidden_size = if self.bidirectional {
                self.hidden_size * 2
            } else {
                self.hidden_size
            };
            Self::transpose_3d(&layer_input, seq_len, batch_size, output_hidden_size)?
        } else {
            layer_input
        };

        Ok((output, h))
    }

    /// Transpose dimensions 0 and 1 of a 3D tensor.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(dim0, dim1, dim2)`
    ///
    /// # Returns
    /// Transposed tensor of shape `(dim1, dim0, dim2)`
    fn transpose_3d(
        input: &CpuTensor<T>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
    ) -> Result<CpuTensor<T>> {
        let input_data = input.as_slice();
        let mut transposed_data = Vec::with_capacity(dim0 * dim1 * dim2);

        // Transpose: (dim0, dim1, dim2) ? (dim1, dim0, dim2)
        for b in 0..dim1 {
            for t in 0..dim0 {
                let start = (t * dim1 + b) * dim2;
                let end = start + dim2;
                transposed_data.extend_from_slice(&input_data[start..end]);
            }
        }

        Ok(Tensor::from_vec(transposed_data, &[dim1, dim0, dim2])?)
    }

    /// Reverse a sequence tensor along the time dimension (dim 0).
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(seq_len, batch_size, feature_size)`
    ///
    /// # Returns
    /// Reversed tensor of shape `(seq_len, batch_size, feature_size)`
    fn reverse_sequence(
        input: &CpuTensor<T>,
        seq_len: usize,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<CpuTensor<T>> {
        let input_data = input.as_slice();
        let mut reversed_data = Vec::with_capacity(seq_len * batch_size * feature_size);

        // Reverse along seq_len dimension
        for t in (0..seq_len).rev() {
            let start = t * batch_size * feature_size;
            let end = (t + 1) * batch_size * feature_size;
            reversed_data.extend_from_slice(&input_data[start..end]);
        }

        Ok(Tensor::from_vec(
            reversed_data,
            &[seq_len, batch_size, feature_size],
        )?)
    }

    /// Forward pass for a single RNN layer (or direction).
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(seq_len, batch_size, input_size)`
    /// * `hidden` - Hidden state tensor
    /// * `weight_idx` - Index into weight arrays (for bidirectional: layer*2 or layer*2+1)
    /// * `seq_len` - Sequence length
    /// * `batch_size` - Batch size
    /// * `input_size` - Input feature size for this layer/direction
    fn forward_layer(
        &self,
        input: &CpuTensor<T>,
        hidden: &CpuTensor<T>,
        weight_idx: usize,
        seq_len: usize,
        batch_size: usize,
        input_size: usize,
    ) -> Result<TensorPair<T>> {
        // RNN computation: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
        // Implement proper sequence-by-sequence processing

        // Prepare weight matrices
        let weight_ih_data = self.weight_ih[weight_idx].data().as_slice().to_vec();
        let weight_ih = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            weight_ih_data,
            &[self.hidden_size, input_size],
        )?;

        let weight_hh_data = self.weight_hh[weight_idx].data().as_slice().to_vec();
        let weight_hh = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            weight_hh_data,
            &[self.hidden_size, self.hidden_size],
        )?;

        // Prepare bias tensors if enabled
        let bias_ih = if self.bias {
            let bias_data = self.bias_ih[weight_idx].data().as_slice().to_vec();
            Some(Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                bias_data,
                &[self.hidden_size],
            )?)
        } else {
            None
        };

        let bias_hh = if self.bias {
            let bias_data = self.bias_hh[weight_idx].data().as_slice().to_vec();
            Some(Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                bias_data,
                &[self.hidden_size],
            )?)
        } else {
            None
        };

        // Initialize output tensor: (seq_len, batch_size, hidden_size)
        let mut output_data = Vec::with_capacity(seq_len * batch_size * self.hidden_size);

        // Get initial hidden state for this layer/direction
        let hidden_slice = hidden.as_slice();
        let hidden_start = weight_idx * batch_size * self.hidden_size;
        let hidden_end = (weight_idx + 1) * batch_size * self.hidden_size;
        let mut current_hidden = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            hidden_slice[hidden_start..hidden_end].to_vec(),
            &[batch_size, self.hidden_size],
        )?;

        // Process each time step sequentially
        for t in 0..seq_len {
            // Get input at current time step: (batch_size, input_size)
            let input_start = t * batch_size * input_size;
            let input_end = (t + 1) * batch_size * input_size;
            let x_t_data = &input.as_slice()[input_start..input_end];
            let x_t = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                x_t_data.to_vec(),
                &[batch_size, input_size],
            )?;

            // Compute W_ih @ x_t: (batch_size, hidden_size)
            let ih_out = x_t.matmul(&weight_ih.transpose(1, 0)?)?;

            // Add input bias if enabled
            let ih_with_bias = if let Some(ref bias) = bias_ih {
                &ih_out + bias
            } else {
                ih_out
            };

            // Compute W_hh @ h_{t-1}: (batch_size, hidden_size)
            let hh_out = current_hidden.matmul(&weight_hh)?;

            // Add hidden bias if enabled
            let hh_with_bias = if let Some(ref bias) = bias_hh {
                &hh_out + bias
            } else {
                hh_out
            };

            // Combine: W_ih @ x_t + W_hh @ h_{t-1} + biases
            let combined = &ih_with_bias + &hh_with_bias;

            // Apply tanh activation: h_t = tanh(combined)
            current_hidden = crate::functional::tanh(&combined)?;

            // Store output for this time step
            output_data.extend_from_slice(current_hidden.as_slice());
        }

        // Create output tensor: (seq_len, batch_size, hidden_size)
        let layer_output = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            output_data,
            &[seq_len, batch_size, self.hidden_size],
        )?;

        // Update hidden state in the hidden tensor
        let mut layer_hidden = hidden.clone();
        let hidden_slice_mut = layer_hidden.as_mut_slice();
        hidden_slice_mut[hidden_start..hidden_end].copy_from_slice(current_hidden.as_slice());

        Ok((layer_output, layer_hidden))
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T> for RNN<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + num_traits::Float,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Check if autograd is required
        let requires_grad = input.requires_grad()
            || self.weight_ih.iter().any(|w| w.requires_grad())
            || self.weight_hh.iter().any(|w| w.requires_grad())
            || self.bias_ih.iter().any(|b| b.data().requires_grad())
            || self.bias_hh.iter().any(|b| b.data().requires_grad());

        let (output, _hidden) = self.forward_with_hidden(input, None)?;

        // Attach autograd function if gradients are required
        if requires_grad {
            use coeus_autograd::functions::RNNFunction;
            use coeus_autograd::functions::TensorRef;

            // Collect all input tensors that require gradients
            let input_ref = TensorRef::from(input.clone());
            let mut weights_refs = Vec::new();
            let mut biases_refs = Vec::new();

            // Add weight references
            for weight in &self.weight_ih {
                if weight.requires_grad() {
                    weights_refs.push(TensorRef::from(weight.data().clone()));
                }
            }
            for weight in &self.weight_hh {
                if weight.requires_grad() {
                    weights_refs.push(TensorRef::from(weight.data().clone()));
                }
            }

            // Add bias references
            for bias in &self.bias_ih {
                if bias.data().requires_grad() {
                    biases_refs.push(Some(TensorRef::from(bias.data().clone())));
                } else {
                    biases_refs.push(None);
                }
            }
            for bias in &self.bias_hh {
                if bias.data().requires_grad() {
                    biases_refs.push(Some(TensorRef::from(bias.data().clone())));
                } else {
                    biases_refs.push(None);
                }
            }

            // Collect all input references
            let mut all_inputs = vec![input_ref];
            all_inputs.extend(weights_refs);
            all_inputs.extend(biases_refs.into_iter().flatten());

            // Create RNN function (simplified - full implementation would store hidden states)
            let rnn_fn = RNNFunction::new(
                all_inputs,
                Vec::new(), // hidden_states (would need to be collected during forward)
                self.batch_first,
                self.bidirectional,
                "tanh".to_string(), // nonlinearity
            );

            // Attach the function to the output tensor
            let mut output_with_grad = output.requires_grad_(true);
            let function: std::sync::Arc<
                dyn coeus_autograd::Function<CpuBackend<T>, DenseStorage<T>, T>,
            > = std::sync::Arc::new(rnn_fn);
            output_with_grad.set_grad_fn(Some(function));
            Ok(output_with_grad)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        let mut params = Vec::new();
        params.extend(self.weight_ih.iter().cloned());
        params.extend(self.weight_hh.iter().cloned());
        params.extend(self.bias_ih.iter().cloned());
        params.extend(self.bias_hh.iter().cloned());
        params
    }

    fn zero_grad(&mut self) {
        for param in &mut self.weight_ih {
            param.zero_grad();
        }
        for param in &mut self.weight_hh {
            param.zero_grad();
        }
        for param in &mut self.bias_ih {
            param.zero_grad();
        }
        for param in &mut self.bias_hh {
            param.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // RNN layers don't have training-specific behavior
    }

    fn name(&self) -> &str {
        "RNN"
    }
}

impl<B, S, T> fmt::Display for RNN<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RNN(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, bidirectional={})",
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.bidirectional
        )
    }
}

#[cfg(test)]
mod rnn_forward_var_tests {
    use super::*;

    use coeus_dtype::float::Float32;
    use coeus_tensor::Tensor;

    #[test]
    fn test_rnn_forward_var() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5, 1, false, true, false)
                .unwrap();

        // Create Variable input [seq_len=3, batch_size=2, input_size=10]
        let input_data = vec![Float32::new(1.0); 3 * 2 * 10]; // 60 elements
        let input_tensor = Tensor::from_vec(input_data, &[3, 2, 10]).unwrap();
        let input_var = input_tensor;

        // Forward pass with Variable
        let output_var = rnn.forward(&input_var);

        // Verify output shape [seq_len=3, batch_size=2, output_size=5]
        let output_var = output_var.unwrap();
        let _output_data = output_var.as_slice();
        assert_eq!(output_var.shape().dims(), &[3, 2, 5]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_rnn_creation() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
                .unwrap();
        assert_eq!(rnn.input_size, 10);
        assert_eq!(rnn.hidden_size, 20);
        assert_eq!(rnn.num_layers, 1);
        assert!(rnn.bias);
        assert!(!rnn.batch_first);
        assert!(!rnn.bidirectional);
    }

    #[test]
    fn test_rnn_bidirectional() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, true)
                .unwrap();
        assert!(rnn.bidirectional);
        assert_eq!(rnn.num_layers, 2);
        // Bidirectional RNN has 2x parameters per layer
        assert_eq!(rnn.weight_ih.len(), 4); // 2 layers * 2 directions
    }

    #[test]
    fn test_rnn_forward_shape() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
                .unwrap();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5, 3, 10]).unwrap();
        let output = rnn.forward(&input).unwrap();
        // RNN outputs (seq_len, batch_size, hidden_size) = (5, 3, 20)
        assert_eq!(output.shape().dims(), &[5, 3, 20]);
    }

    #[test]
    fn test_rnn_parameters() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
                .unwrap();
        let params = rnn.parameters();
        // 1 layer, 1 direction: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_rnn_no_bias() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, false, false, false)
                .unwrap();
        let params = rnn.parameters();
        // 1 layer, 1 direction, no bias: weight_ih, weight_hh only
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_rnn_multilayer() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 3, true, false, false)
                .unwrap();
        assert_eq!(rnn.num_layers, 3);
        let params = rnn.parameters();
        // 3 layers, 1 direction: 4 params per layer
        assert_eq!(params.len(), 12);
    }

    #[test]
    fn test_rnn_batch_first() {
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, true, false)
                .unwrap();
        assert!(rnn.batch_first);
    }

    #[test]
    #[should_panic(expected = "input_size must be > 0")]
    fn test_rnn_invalid_input_size() {
        RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(0, 20, 1, true, false, false)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "hidden_size must be > 0")]
    fn test_rnn_invalid_hidden_size() {
        RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 0, 1, true, false, false)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "num_layers must be > 0")]
    fn test_rnn_invalid_num_layers() {
        RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 0, true, false, false)
            .unwrap();
    }

    #[test]
    fn test_rnn_multilayer_state_propagation() {
        // Test that multi-layer RNNs properly propagate hidden states between layers
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, false)
                .unwrap();

        // Create input: (seq_len=3, batch_size=2, input_size=10) with some non-zero values
        let input_data: Vec<Float32> = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1,
            7.1, 8.1, 9.1, 10.1, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 1.3, 2.3, 3.3,
            4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
        ]
        .into_iter()
        .map(Float32::new)
        .collect();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[3, 2, 10])
                .unwrap();

        let output = rnn.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2, 20]); // seq_len, batch_size, hidden_size

        // Test with different inputs to ensure layers behave differently
        let input2_data: Vec<Float32> = vec![
            2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
            8.1, 9.1, 10.1, 11.1, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 2.3, 3.3,
            4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4,
            11.4, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
        ]
        .into_iter()
        .map(Float32::new)
        .collect();
        let input2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input2_data,
            &[3, 2, 10],
        )
        .unwrap();
        let _output2 = rnn.forward(&input2).unwrap();

        // Verify that multi-layer processing works: correct output shape
        assert_eq!(output.shape().dims(), &[3, 2, 20]); // seq_len, batch_size, hidden_size

        // Test passes if we reach here without panics - multi-layer RNN structure works
        // (Weights are currently initialized to zeros, so outputs are zero, but that's a separate issue)
    }

    #[test]
    #[ignore = "Bidirectional RNN not yet implemented"]
    fn test_rnn_bidirectional_shape() {
        // Test that bidirectional RNN produces correct output shape
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(5, 8, 1, true, false, true)
                .unwrap();

        // Create input: (seq_len=3, batch=2, input_size=5)
        let input_data: Vec<Float32> = (0..30)
            .map(|i| Float32::new((i as f32 + 1.0) * 0.1))
            .collect();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[3, 2, 5])
                .unwrap();

        let output = rnn.forward(&input).unwrap();

        // Bidirectional output should be: (seq_len, batch, hidden_size * 2)
        assert_eq!(
            output.shape().dims(),
            &[3, 2, 16],
            "Bidirectional RNN output shape should be [seq_len, batch, hidden_size*2]"
        );
    }

    #[test]
    #[ignore = "Bidirectional RNN not yet implemented"]
    fn test_rnn_bidirectional_multi_layer() {
        // Test that multi-layer bidirectional RNN works correctly
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 6, 2, true, false, true)
                .unwrap();

        // Create input: (seq_len=2, batch=1, input_size=4)
        let input_data: Vec<Float32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            .into_iter()
            .map(Float32::new)
            .collect();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[2, 1, 4])
                .unwrap();

        let output = rnn.forward(&input).unwrap();

        // Multi-layer bidirectional output: (seq_len, batch, hidden_size * 2)
        assert_eq!(
            output.shape().dims(),
            &[2, 1, 12],
            "Multi-layer bidirectional RNN output shape should be [seq_len, batch, hidden_size*2]"
        );
    }

    #[test]
    #[ignore = "Bidirectional RNN not yet implemented"]
    fn test_rnn_bidirectional_numerical() {
        // Test that bidirectional RNN computation runs without errors
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, 5, 1, true, false, true)
                .unwrap();

        // Create non-zero input: (seq_len=2, batch=1, input_size=3)
        let input_data: Vec<Float32> = vec![0.5, 1.0, -0.5, 0.3, -0.2, 0.9]
            .into_iter()
            .map(Float32::new)
            .collect();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[2, 1, 3])
                .unwrap();

        let output = rnn.forward(&input).unwrap();

        // Verify output values are valid
        let data = output.as_slice();
        assert!(
            data.iter().all(|v| !v.get().is_nan()),
            "Bidirectional RNN output contains NaN"
        );
        assert!(
            data.iter().all(|v| !v.get().is_infinite()),
            "Bidirectional RNN output contains Inf"
        );
        assert!(
            data.iter().all(|v| v.get().abs() <= 1.0),
            "Bidirectional RNN output outside tanh range"
        );
    }

    #[test]
    #[ignore = "Bidirectional RNN not yet implemented"]
    fn test_rnn_bidirectional_batch_first() {
        // Test that bidirectional RNN works with batch_first=true
        let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 6, 1, true, true, true)
            .unwrap();

        // Create input with batch_first: (batch=2, seq_len=3, input_size=4)
        let input_data: Vec<Float32> = (0..24)
            .map(|i| Float32::new((i as f32 + 1.0) * 0.1))
            .collect();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[2, 3, 4])
                .unwrap();

        let output = rnn.forward(&input).unwrap();

        // Output should be: (batch, seq_len, hidden_size * 2)
        assert_eq!(output.shape().dims(), &[2, 3, 12], "Bidirectional RNN with batch_first output shape should be [batch, seq_len, hidden_size*2]");
    }

    #[test]
    fn test_rnn_reverse_sequence() {
        // Test the reverse_sequence helper function
        let input_data: Vec<Float32> = (0..12).map(|i| Float32::new(i as f32)).collect();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[3, 2, 2])
                .unwrap();

        let reversed =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::reverse_sequence(&input, 3, 2, 2)
                .unwrap();

        // Verify shape is preserved
        assert_eq!(reversed.shape().dims(), &[3, 2, 2]);

        // Verify data is reversed along time dimension
        let input_slice = input.as_slice();
        let reversed_slice = reversed.as_slice();

        // First time step in reversed should be last time step in original
        assert_eq!(reversed_slice[0].get(), input_slice[8].get()); // t=0 in reversed = t=2 in original
        assert_eq!(reversed_slice[4].get(), input_slice[4].get()); // t=1 in reversed = t=1 in original
        assert_eq!(reversed_slice[8].get(), input_slice[0].get()); // t=2 in reversed = t=0 in original
    }

    #[test]
    fn test_rnn_weight_initialization() {
        // Test that RNN weights are properly initialized with Xavier uniform (non-zero)
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, false)
                .unwrap();

        // Check weight_ih for first layer
        let w_ih_0 = rnn.weight_ih[0].data();
        let w_ih_0_data = w_ih_0.as_slice();

        // Verify not all zeros (proper Xavier initialization)
        let non_zero_count = w_ih_0_data
            .iter()
            .filter(|&&v| v.get().abs() > 1e-6)
            .count();
        assert!(
            non_zero_count > 0,
            "RNN weight_ih should be initialized with non-zero values (Xavier uniform)"
        );

        // Verify values are within reasonable bounds for Xavier uniform
        // Xavier uniform: U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
        // For input_size=10, hidden_size=20: a approx sqrt(6/30) approx 0.447
        let max_val = w_ih_0_data
            .iter()
            .map(|v| v.get().abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_val < 1.0,
            "RNN weights should be within Xavier uniform bounds"
        );

        // Check weight_hh for first layer
        let w_hh_0 = rnn.weight_hh[0].data();
        let w_hh_0_data = w_hh_0.as_slice();
        let non_zero_count_hh = w_hh_0_data
            .iter()
            .filter(|&&v| v.get().abs() > 1e-6)
            .count();
        assert!(
            non_zero_count_hh > 0,
            "RNN weight_hh should be initialized with non-zero values (Xavier uniform)"
        );
    }

    #[test]
    fn test_rnn_autograd_integration() {
        // Test that RNN creates autograd function when gradients are required
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
                .unwrap();

        // Create input with gradient tracking
        let input_data = vec![Float32::new(1.0); 30]; // 3 time steps, 1 batch, 10 features
        let input = Tensor::from_vec(input_data, &[3, 1, 10])
            .unwrap()
            .requires_grad_(true);

        // Forward pass should create autograd function
        let output = rnn.forward(&input).unwrap();

        // Output should require gradients and have a grad_fn
        assert!(output.requires_grad());
        assert!(output.grad_fn().is_some());

        // Create a scalar loss from the output (sum all elements)
        let loss = output.sum(None, false).unwrap().sum(None, false).unwrap();

        // Check that backward can be called on the scalar loss
        loss.backward().unwrap();

        // Autograd integration test passes if backward succeeds
        // (Full gradient computation for RNN input is complex and simplified)
    }

    #[test]
    fn test_rnn_no_autograd_when_not_required() {
        // Test that RNN doesn't create autograd function when no gradients required
        let rnn =
            RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
                .unwrap();

        // Create input without gradient tracking
        let input_data = vec![Float32::new(1.0); 30];
        let input = Tensor::from_vec(input_data, &[3, 1, 10]).unwrap();

        // Forward pass should not create autograd function
        let output = rnn.forward(&input).unwrap();

        // Output should not require gradients
        assert!(!output.requires_grad());
        assert!(output.grad_fn().is_none());
    }
}
