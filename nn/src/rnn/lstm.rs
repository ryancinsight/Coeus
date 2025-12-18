//! LSTM (Long Short-Term Memory) layer for sequence modeling.
//!
//! Implements the LSTM architecture with forget, input, output, and candidate gates.
//! Provides better gradient flow than basic RNNs for long sequences.

use std::fmt;
use std::marker::PhantomData;

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

#[derive(Debug)]
pub struct LSTM<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Input-to-hidden weights for each gate (i, f, g, o) and layer
    pub weight_ih: Vec<Parameter<B, S, T>>,
    /// Hidden-to-hidden weights for each gate (i, f, g, o) and layer
    pub weight_hh: Vec<Parameter<B, S, T>>,
    /// Input-to-hidden biases for each gate (i, f, g, o) and layer
    pub bias_ih: Vec<Parameter<B, S, T>>,
    /// Hidden-to-hidden biases for each gate (i, f, g, o) and layer
    pub bias_hh: Vec<Parameter<B, S, T>>,
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
    /// Whether this is a bidirectional LSTM
    #[allow(dead_code)]
    pub bidirectional: bool,
    /// Projection size (if set, projects hidden state to this size)
    pub proj_size: Option<usize>,
    /// Phantom data to ensure B and S are used for type safety
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> LSTM<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new LSTM layer.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `bias` - If false, the layer does not use bias weights (default: true)
    /// * `batch_first` - If true, input/output tensors are (batch, seq, feature) (default: false)
    /// * `bidirectional` - If true, becomes a bidirectional LSTM (default: false)
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

                // LSTM has 4 gates (i, f, g, o), so weights are 4x larger
                let gate_size = 4 * hidden_size;

                // Xavier/Glorot uniform initialization
                let limit = (T::from(6.0).unwrap()
                    / T::from(layer_input_size + hidden_size).unwrap())
                .sqrt();
                let w_ih = Self::xavier_uniform_init(gate_size, layer_input_size, limit);
                let w_hh = Self::xavier_uniform_init(gate_size, hidden_size, limit);

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
                        Tensor::<B, S, T>::zeros(&[gate_size])?;
                    let b_hh =
                        Tensor::<B, S, T>::zeros(&[gate_size])?;

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
            proj_size: None,
            _phantom: PhantomData,
        })
    }

    /// Set projection size.
    pub fn with_proj_size(mut self, proj_size: usize) -> Self {
        self.proj_size = Some(proj_size);
        self
    }

    /// Set batch_first.
    pub fn with_batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
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
    ) -> Tensor<B, S, T> {
        let mut tensor = Tensor::<B, S, T>::zeros(&[rows, cols]).unwrap();
        crate::init::xavier_uniform_(&mut tensor, 1.0).unwrap();
        tensor
    }

    /// Transpose dimensions 0 and 1 of a 3D tensor.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(dim0, dim1, dim2)`
    ///
    /// # Returns
    /// Transposed tensor of shape `(dim1, dim0, dim2)`
    fn transpose_3d(
        input: &Tensor<B, S, T>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
    ) -> Result<Tensor<B, S, T>> {
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
        input: &Tensor<B, S, T>,
        seq_len: usize,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<Tensor<B, S, T>> {
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
}

impl<B, S, T> LSTM<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Forward pass for a single LSTM layer (or direction).
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `(seq_len, batch_size, input_size)`
    /// * `h` - Hidden state tensor
    /// * `c` - Cell state tensor
    /// * `weight_idx` - Index into weight arrays (for bidirectional: layer*2 or layer*2+1)
    /// * `dims` - Tuple of (seq_len, batch_size, input_size)
    ///
    /// # Returns
    /// Tuple of (hidden_output, cell_output)
    #[allow(clippy::type_complexity)]
    fn forward_layer_unidirectional_lstm(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        h: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        c: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        weight_idx: usize,
        dims: (usize, usize, usize),
    ) -> Result<(
        Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    )> {
        let (seq_len, batch_size, input_size) = dims;
        let current_input_size = input_size;

        // Handle empty sequence edge case
        if seq_len == 0 {
            // Return empty tensors with correct shapes
            let empty_output = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[
                0,
                batch_size,
                self.hidden_size,
            ])?;
            let empty_cell =
                Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[batch_size, self.hidden_size])?;
            return Ok((empty_output, empty_cell));
        }

        // Prepare weight matrices
        let weight_ih_data = self.weight_ih[weight_idx].data().as_slice().to_vec();
        let weight_hh_data = self.weight_hh[weight_idx].data().as_slice().to_vec();

        let weight_ih = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            weight_ih_data,
            &[4 * self.hidden_size, current_input_size],
        )?;
        let weight_hh = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            weight_hh_data,
            &[4 * self.hidden_size, self.hidden_size],
        )?;

        // Prepare bias tensors if enabled
        let bias_ih = if self.bias {
            let bias_data = self.bias_ih[weight_idx].data().as_slice().to_vec();
            Some(Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                bias_data,
                &[4 * self.hidden_size],
            )?)
        } else {
            None
        };

        let bias_hh = if self.bias {
            let bias_data = self.bias_hh[weight_idx].data().as_slice().to_vec();
            Some(Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                bias_data,
                &[4 * self.hidden_size],
            )?)
        } else {
            None
        };

        // Initialize output tensor: (seq_len, batch_size, hidden_size)
        let mut output_data = Vec::with_capacity(seq_len * batch_size * self.hidden_size);

        // Get initial hidden/cell states for this layer/direction
        let layer_offset = weight_idx * batch_size * self.hidden_size;
        let mut current_hidden = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            h.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
            &[batch_size, self.hidden_size],
        )?;
        let mut current_cell = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            c.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
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

            // Compute gates: W_ih @ x_t + W_hh @ h_{t-1}
            let ih_gates = x_t.matmul(&weight_ih.transpose(1, 0)?)?;
            let hh_gates = current_hidden.matmul(&weight_hh.transpose(1, 0)?)?;

            // Combine gates and add biases
            let mut gates = &ih_gates + &hh_gates;

            if let (Some(ref bias_i), Some(ref bias_h)) = (&bias_ih, &bias_hh) {
                gates = &gates + &(bias_i + bias_h);
            }

            // Split gates into i, f, g, o components
            let gates_data = gates.as_slice();
            let mut i_data = Vec::with_capacity(batch_size * self.hidden_size);
            let mut f_data = Vec::with_capacity(batch_size * self.hidden_size);
            let mut g_data = Vec::with_capacity(batch_size * self.hidden_size);
            let mut o_data = Vec::with_capacity(batch_size * self.hidden_size);

            for b in 0..batch_size {
                let offset = b * 4 * self.hidden_size;
                i_data.extend_from_slice(&gates_data[offset..offset + self.hidden_size]);
                f_data.extend_from_slice(
                    &gates_data[offset + self.hidden_size..offset + 2 * self.hidden_size],
                );
                g_data.extend_from_slice(
                    &gates_data[offset + 2 * self.hidden_size..offset + 3 * self.hidden_size],
                );
                o_data.extend_from_slice(
                    &gates_data[offset + 3 * self.hidden_size..offset + 4 * self.hidden_size],
                );
            }

            // Create gate tensors
            let i_gate = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                i_data,
                &[batch_size, self.hidden_size],
            )?;
            let f_gate = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                f_data,
                &[batch_size, self.hidden_size],
            )?;
            let g_gate = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                g_data,
                &[batch_size, self.hidden_size],
            )?;
            let o_gate = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                o_data,
                &[batch_size, self.hidden_size],
            )?;

            // Apply activations: sigmoid for i, f, o; tanh for g
            let i_activated = crate::functional::sigmoid(&i_gate)?;
            let f_activated = crate::functional::sigmoid(&f_gate)?;
            let g_activated = crate::functional::tanh(&g_gate)?;
            let o_activated = crate::functional::sigmoid(&o_gate)?;

            // Update cell state: C_t = f_t * C_{t-1} + i_t * g_t
            let f_times_c = &f_activated * &current_cell;
            let i_times_g = &i_activated * &g_activated;
            current_cell = &f_times_c + &i_times_g;

            // Update hidden state: h_t = o_t * tanh(C_t)
            let tanh_c = crate::functional::tanh(&current_cell)?;
            current_hidden = &o_activated * &tanh_c;

            // Store output for this time step
            output_data.extend_from_slice(current_hidden.as_slice());
        }

        // Create output tensor: (seq_len, batch_size, hidden_size)
        let layer_output = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            output_data,
            &[seq_len, batch_size, self.hidden_size],
        )?;

        Ok((layer_output, current_cell))
    }

    /// Forward pass with explicit state handling.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `state` - Optional initial hidden and cell states (h_0, c_0)
    ///
    /// # Returns
    /// Tuple of (output, (h_n, c_n))
    pub fn forward(
        &self,
        input: &Tensor<B, S, T>,
        state: Option<(
            &Tensor<B, S, T>,
            &Tensor<B, S, T>,
        )>,
    ) -> Result<(
        Tensor<B, S, T>,
        (
            Tensor<B, S, T>,
            Tensor<B, S, T>,
        ),
    )> {
        let input_shape = input.shape().dims();

        // Determine actual dimensions based on batch_first
        let (seq_len, batch_size, input_size) = if self.batch_first {
            // Input is (batch, seq, input) -> need (seq, batch, input)
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            // Input is already (seq, batch, input)
            (input_shape[0], input_shape[1], input_shape[2])
        };

        // Initialize hidden and cell states on CPU for processing
        let num_directions = if self.bidirectional { 2 } else { 1 };
        
        let (h_cpu, c_cpu) = if let Some((h_0, c_0)) = state {
            (
                Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    h_0.as_slice().to_vec(),
                    h_0.shape().dims(),
                )?,
                Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    c_0.as_slice().to_vec(),
                    c_0.shape().dims(),
                )?,
            )
        } else {
            let h = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ])?;
            let c = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ])?;
            (h, c)
        };

        // Transpose input if batch_first: (batch, seq_len, input_size) -> (seq_len, batch, input_size)
        let input_seq = if self.batch_first {
            Self::transpose_3d(input, input_shape[0], input_shape[1], input_shape[2])?
        } else {
            input.clone()
        };

        // Convert input to CPU for processing
        let mut layer_input_cpu = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            input_seq.as_slice().to_vec(),
            input_seq.shape().dims(),
        )?;

        // Storage for final states
        let mut h_n_data = Vec::with_capacity(self.num_layers * num_directions * batch_size * self.hidden_size);
        let mut c_n_data = Vec::with_capacity(self.num_layers * num_directions * batch_size * self.hidden_size);

        // Process each layer with bidirectional support
        for layer in 0..self.num_layers {
            if self.bidirectional {
                // Bidirectional: process forward and backward directions separately

                // Determine input size for this layer
                let layer_input_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size * 2 // Previous layer output is concatenated
                };

                // Forward direction (use weights at layer*2)
                let (forward_output, forward_cell) = self.forward_layer_unidirectional_lstm(
                    &layer_input_cpu,
                    &h_cpu,
                    &c_cpu,
                    layer * 2,
                    (seq_len, batch_size, layer_input_size),
                )?;
                
                // Extract final hidden state from forward_output (last time step)
                let f_out_slice = forward_output.as_slice();
                let last_t_start = (seq_len - 1) * batch_size * self.hidden_size;
                h_n_data.extend_from_slice(&f_out_slice[last_t_start..]);
                
                // Cell state is returned directly
                c_n_data.extend_from_slice(forward_cell.as_slice());

                // Backward direction (use weights at layer*2+1)
                let reversed_input =
                    LSTM::<CpuBackend<T>, DenseStorage<T>, T>::reverse_sequence(&layer_input_cpu, seq_len, batch_size, layer_input_size)?;
                let (backward_output_reversed, backward_cell) = self
                    .forward_layer_unidirectional_lstm(
                        &reversed_input,
                        &h_cpu,
                        &c_cpu,
                        layer * 2 + 1,
                        (seq_len, batch_size, layer_input_size),
                    )?;
                let backward_output = LSTM::<CpuBackend<T>, DenseStorage<T>, T>::reverse_sequence(
                    &backward_output_reversed,
                    seq_len,
                    batch_size,
                    self.hidden_size,
                )?;

                // Extract final hidden state from backward_output
                let b_out_rev_slice = backward_output_reversed.as_slice();
                let last_t_start_rev = (seq_len - 1) * batch_size * self.hidden_size;
                h_n_data.extend_from_slice(&b_out_rev_slice[last_t_start_rev..]);
                
                c_n_data.extend_from_slice(backward_cell.as_slice());

                // Concatenate forward and backward outputs along hidden dimension
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

                layer_input_cpu = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    concatenated_data,
                    &[seq_len, batch_size, self.hidden_size * 2],
                )?;
            } else {
                // Unidirectional: process forward direction only
                let current_input_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size
                };
                let (layer_output, layer_cell) = self.forward_layer_unidirectional_lstm(
                    &layer_input_cpu,
                    &h_cpu,
                    &c_cpu,
                    layer,
                    (seq_len, batch_size, current_input_size),
                )?;
                
                // Extract final hidden state
                let out_slice = layer_output.as_slice();
                let last_t_start = (seq_len - 1) * batch_size * self.hidden_size;
                h_n_data.extend_from_slice(&out_slice[last_t_start..]);
                
                c_n_data.extend_from_slice(layer_cell.as_slice());
                
                layer_input_cpu = layer_output;
            }
        }

        // Convert final output back to generic backend
        let layer_input_generic = Tensor::<B, S, T>::from_vec(
            layer_input_cpu.as_slice().to_vec(),
            layer_input_cpu.shape().dims(),
        )?;

        // Transpose output if batch_first: (seq_len, batch, hidden_size) -> (batch, seq_len, hidden_size)
        let output = if self.batch_first {
            let output_hidden_size = if self.bidirectional {
                self.hidden_size * 2
            } else {
                self.hidden_size
            };
            Self::transpose_3d(&layer_input_generic, seq_len, batch_size, output_hidden_size)?
        } else {
            layer_input_generic
        };
        
        // Construct h_n and c_n tensors
        let h_n = Tensor::from_vec(
            h_n_data,
            &[self.num_layers * num_directions, batch_size, self.hidden_size],
        )?;
        let c_n = Tensor::from_vec(
            c_n_data,
            &[self.num_layers * num_directions, batch_size, self.hidden_size],
        )?;

        Ok((output, (h_n, c_n)))
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T> for LSTM<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let (output, _) = self.forward(input, None)?;
        Ok(output)
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
        // LSTM layers don't have training-specific behavior
    }

    fn name(&self) -> &str {
        "LSTM"
    }
}

impl<B, S, T> fmt::Display for LSTM<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LSTM(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, bidirectional={})",
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
mod tests {
    use dtype::float::Float32;
    use super::*;
    use backend::{Backend, CpuBackend};
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_lstm_new() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
            .unwrap();
        assert_eq!(lstm.input_size, 10);
        assert_eq!(lstm.hidden_size, 20);
        assert_eq!(lstm.num_layers, 1);
        assert!(lstm.bias);
        assert!(!lstm.batch_first);
        assert!(!lstm.bidirectional);
    }

    #[test]
    fn test_lstm_bidirectional() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, true)
            .unwrap();
        assert!(lstm.bidirectional);
    }

    #[test]
    fn test_lstm_forward() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
            .unwrap();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5, 3, 10]).unwrap(); // (seq, batch, input)
        let (output, (h_n, c_n)) = lstm.forward(&input, None).unwrap();

        assert_eq!(output.shape().dims(), &[5, 3, 20]);
        assert_eq!(h_n.shape().dims(), &[1, 3, 20]);
        assert_eq!(c_n.shape().dims(), &[1, 3, 20]);
    }

    #[test]
    fn test_lstm_shapes() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, false, false)
            .unwrap();
        // Weights
        // Input-Hidden: 4 * hidden_size * input_size
        assert_eq!(lstm.parameters()[0].data().shape().dims(), &[800]);
        // Hidden-Hidden: 4 * hidden_size * hidden_size
        assert_eq!(lstm.parameters()[1].data().shape().dims(), &[1600]);

        // Bias
        // Input-Hidden: 4 * hidden_size
        assert_eq!(lstm.parameters()[2].data().shape().dims(), &[80]);
        // Hidden-Hidden: 4 * hidden_size
        assert_eq!(lstm.parameters()[3].data().shape().dims(), &[80]);
    }

    #[test]
    fn test_lstm_no_bias() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, false, false, false)
            .unwrap();
        // Should only have weights, no biases
        assert_eq!(lstm.parameters().len(), 2);
    }

    #[test]
    fn test_lstm_multiple_layers() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 3, true, false, false)
            .unwrap();
        // (4 weights per layer) * 3 layers = 12
        assert_eq!(lstm.parameters().len(), 12);
    }

    #[test]
    fn test_lstm_dropout() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, true, false)
            .unwrap();
        // Dropout is an internal property, hard to test directly without running forward
        // But construction should succeed
    }

    #[test]
    fn test_lstm_invalid_args() {
        assert!(LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(0, 20, 1, true, false, false)
            .is_err());
        assert!(LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 0, 1, true, false, false)
            .is_err());
        assert!(LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 0, true, false, false)
            .is_err());
    }

    #[test]
    fn test_lstm_projection() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, false)
            .unwrap()
            .with_proj_size(10);
        
        assert_eq!(lstm.proj_size, Some(10));
    }

    #[test]
    fn test_reverse_sequence() {
        let input_data = vec![
            Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
            Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),
            Float32::new(7.0), Float32::new(8.0), Float32::new(9.0),
            Float32::new(10.0), Float32::new(11.0), Float32::new(12.0),
            Float32::new(13.0), Float32::new(14.0), Float32::new(15.0),
            Float32::new(16.0), Float32::new(17.0), Float32::new(18.0),
        ];
        // Shape: (3, 2, 3) -> (seq, batch, input)
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[3, 2, 3])
            .unwrap();

        let reversed =
            LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::reverse_sequence(&input, 3, 2, 3).unwrap();
        
        let reversed_data = reversed.as_slice().to_vec();
        
        // First sequence, batch 1: 13,14,15 (was 1,2,3)
        assert_eq!(reversed_data[0], Float32::new(13.0));
        assert_eq!(reversed_data[1], Float32::new(14.0));
        assert_eq!(reversed_data[2], Float32::new(15.0));
        
        // First sequence, batch 2: 16,17,18 (was 4,5,6)
        assert_eq!(reversed_data[3], Float32::new(16.0));
        assert_eq!(reversed_data[4], Float32::new(17.0));
        assert_eq!(reversed_data[5], Float32::new(18.0));

        // Last sequence, batch 1: 1,2,3 (was 13,14,15)
        assert_eq!(reversed_data[12], Float32::new(1.0));
        assert_eq!(reversed_data[13], Float32::new(2.0));
        assert_eq!(reversed_data[14], Float32::new(3.0));
    }

    #[test]
    fn test_lstm_batch_first() {
        // Create LSTM with batch_first=true
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 1, true, true, false)
            .unwrap()
            .with_batch_first(true);
            
        // Input shape: (batch, seq, input) = (3, 5, 10)
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[3, 5, 10]).unwrap();
        
        let (output, (h_n, c_n)) = lstm.forward(&input, None).unwrap();
        
        // Output shape should be (batch, seq, num_directions * hidden_size)
        assert_eq!(output.shape().dims(), &[3, 5, 20]);
        // h_n and c_n should be (num_layers * num_directions, batch, hidden_size)
        assert_eq!(h_n.shape().dims(), &[1, 3, 20]);
        assert_eq!(c_n.shape().dims(), &[1, 3, 20]);
    }

    #[test]
    fn test_reverse_sequence_batch_first() {
        let input_data = vec![
            Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0),
            Float32::new(5.0), Float32::new(6.0), Float32::new(7.0), Float32::new(8.0),
            Float32::new(9.0), Float32::new(10.0), Float32::new(11.0), Float32::new(12.0),
            
            Float32::new(13.0), Float32::new(14.0), Float32::new(15.0), Float32::new(16.0),
            Float32::new(17.0), Float32::new(18.0), Float32::new(19.0), Float32::new(20.0),
            Float32::new(21.0), Float32::new(22.0), Float32::new(23.0), Float32::new(24.0),
        ];
        // Shape: (2, 3, 4) -> (batch, seq, input)
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[2, 3, 4])
            .unwrap();
            
        // For batch_first, we transpose, reverse, then transpose back
        let input_t = input.transpose(0, 1).unwrap();
        let reversed_t = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::reverse_sequence(&input_t, 3, 2, 4).unwrap();
        let reversed = reversed_t.transpose(0, 1).unwrap();
        
        let reversed_data = reversed.as_slice().to_vec();
        
        // Batch 1, Seq 1 (was 1,2,3,4) -> should be 9,10,11,12
        assert_eq!(reversed_data[0], Float32::new(9.0));
        
        // Batch 1, Seq 3 (was 9,10,11,12) -> should be 1,2,3,4
        assert_eq!(reversed_data[8], Float32::new(1.0));
        
        // Batch 2, Seq 1 (was 13,14,15,16) -> should be 21,22,23,24
        assert_eq!(reversed_data[12], Float32::new(21.0));
        
        // Batch 2, Seq 3 (was 21,22,23,24) -> should be 13,14,15,16
        assert_eq!(reversed_data[20], Float32::new(13.0));
    }

    #[test]
    fn test_bidirectional_forward() {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 20, 2, true, false, true)
            .unwrap();
        
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5, 3, 10]).unwrap(); // (seq, batch, input)
        let (output, (h_n, c_n)) = lstm.forward(&input, None).unwrap();
        
        // Output shape: (seq, batch, num_directions * hidden_size)
        assert_eq!(output.shape().dims(), &[5, 3, 40]); // 2 * 20
        
        // Hidden/Cell states: (num_layers * num_directions, batch, hidden_size)
        // 2 layers * 2 directions = 4
        assert_eq!(h_n.shape().dims(), &[4, 3, 20]);
        assert_eq!(c_n.shape().dims(), &[4, 3, 20]);
    }
}