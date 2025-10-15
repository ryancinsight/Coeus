//! GRU Module trait implementation.
//!
//! This module implements the Module trait for GRU layers.

use coeus_backend::CpuBackend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;
use crate::rnn::gru_core::GRU;
use crate::rnn::gru_forward::GRUForward;
use crate::rnn::gru_display;

/// Type aliases to reduce complexity
type CpuTensor<T> = Tensor<CpuBackend, DenseStorage<T>, T>;

impl<T> GRUForward<CpuBackend, DenseStorage<T>, T> for GRU<CpuBackend, DenseStorage<T>, T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward_layer_unidirectional(
        &self,
        input: &CpuTensor<T>,
        h: &CpuTensor<T>,
        weight_idx: usize,
        dims: (usize, usize, usize),
    ) -> Result<CpuTensor<T>> {
        let (seq_len, batch_size, input_size) = dims;

        // Reshape input from (seq_len, batch_size, input_size) to (seq_len * batch_size, input_size)
        let input_seq_batch =
            input.reshape(&[(seq_len * batch_size) as isize, input_size as isize])?;

        // Get previous hidden state for this weight index
        let layer_offset = weight_idx * batch_size * self.hidden_size;
        let h_prev_flat =
            h.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec();
        let h_prev = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            h_prev_flat.clone(),
            &[batch_size, self.hidden_size],
        )?;

        let weight_ih_data = self.weight_ih[weight_idx].data().as_slice().to_vec();
        let weight_hh_data = self.weight_hh[weight_idx].data().as_slice().to_vec();

        // weight_ih/hh have shape (3*hidden_size, input_size/hidden_size) for GRU (r, z, n gates)
        let weight_ih = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            weight_ih_data,
            &[3 * self.hidden_size, input_size],
        )?;
        let weight_hh = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            weight_hh_data,
            &[3 * self.hidden_size, self.hidden_size],
        )?;

        // Compute gates: (seq_len * batch_size, 3 * hidden_size)
        let ih_gates = input_seq_batch.matmul(&weight_ih.transpose(1, 0)?)?;
        let hh_gates = h_prev.matmul(&weight_hh.transpose(1, 0)?)?;

        // Expand hh_gates to match sequence length
        let hh_gates_expanded_data = hh_gates.as_slice().repeat(seq_len);
        let hh_gates_expanded = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            hh_gates_expanded_data,
            &[seq_len * batch_size, 3 * self.hidden_size],
        )?;

        // Add biases if enabled
        let gates = if self.bias {
            let bias_ih_data = self.bias_ih[weight_idx].data().as_slice();
            let bias_hh_data = self.bias_hh[weight_idx].data().as_slice();

            // Create bias tensor with correct shape
            let bias_ih_tensor = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
                bias_ih_data.to_vec(),
                &[3 * self.hidden_size],
            )?;
            let bias_hh_tensor = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
                bias_hh_data.to_vec(),
                &[3 * self.hidden_size],
            )?;

            let bias_combined = &bias_ih_tensor + &bias_hh_tensor;

            // Expand to match sequence length
            let bias_expanded_data = bias_combined.as_slice().repeat(seq_len * batch_size);
            let bias_expanded = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
                bias_expanded_data,
                &[seq_len * batch_size, 3 * self.hidden_size],
            )?;

            &(&ih_gates + &hh_gates_expanded) + &bias_expanded
        } else {
            &ih_gates + &hh_gates_expanded
        };

        // Proper GRU: split gates and apply correct activations
        let gates_data = gates.as_slice();
        let total_elements = seq_len * batch_size;

        // Split gates into r, z, n
        let mut r_gate_data = Vec::with_capacity(total_elements * self.hidden_size);
        let mut z_gate_data = Vec::with_capacity(total_elements * self.hidden_size);
        let mut n_gate_data = Vec::with_capacity(total_elements * self.hidden_size);

        for chunk in gates_data.chunks(3 * self.hidden_size) {
            r_gate_data.extend_from_slice(&chunk[0..self.hidden_size]);
            z_gate_data.extend_from_slice(&chunk[self.hidden_size..2 * self.hidden_size]);
            n_gate_data.extend_from_slice(&chunk[2 * self.hidden_size..3 * self.hidden_size]);
        }

        // Create tensors for each gate
        let r_gate = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            r_gate_data,
            &[total_elements, self.hidden_size],
        )?;
        let z_gate = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            z_gate_data,
            &[total_elements, self.hidden_size],
        )?;
        let n_gate = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            n_gate_data,
            &[total_elements, self.hidden_size],
        )?;

        // Apply activations: sigmoid for r, z; tanh for n
        let _r_activated = crate::functional::sigmoid(&r_gate)?;
        let z_activated = crate::functional::sigmoid(&z_gate)?;
        let n_activated = crate::functional::tanh(&n_gate)?;

        // Expand h_prev to match sequence length
        let h_prev_expanded_data: Vec<T> = h_prev_flat
            .iter()
            .copied()
            .cycle()
            .take(total_elements * self.hidden_size)
            .collect();
        let h_prev_expanded = Tensor::<CpuBackend, DenseStorage<T>, T>::from_vec(
            h_prev_expanded_data,
            &[total_elements, self.hidden_size],
        )?;

        // Compute hidden state: h_t = (1 - z_t) ? h_{t-1} + z_t ? n_t
        let ones =
            Tensor::<CpuBackend, DenseStorage<T>, T>::ones(&[total_elements, self.hidden_size])?;
        let one_minus_z = &ones - &z_activated;

        let h_prev_component = &one_minus_z * &h_prev_expanded;
        let n_component = &z_activated * &n_activated;
        let h_new = &h_prev_component + &n_component;

        // Reshape output back to (seq_len, batch_size, hidden_size)
        let layer_output = h_new.reshape(&[
            seq_len as isize,
            batch_size as isize,
            self.hidden_size as isize,
        ])?;

        Ok(layer_output)
    }
}

impl<T> Module<CpuBackend, DenseStorage<T>, T> for GRU<CpuBackend, DenseStorage<T>, T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        // Determine actual dimensions based on batch_first
        let (seq_len, batch_size, input_size) = if self.batch_first {
            // Input is (batch, seq, input) ? need (seq, batch, input)
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            // Input is already (seq, batch, input)
            (input_shape[0], input_shape[1], input_shape[2])
        };

        // Initialize hidden state
        let num_directions = if self.bidirectional { 2 } else { 1 };
        let h = Tensor::<CpuBackend, DenseStorage<T>, T>::zeros(&[
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
        ])?;

        // Transpose input if batch_first: (batch, seq_len, input_size) ? (seq_len, batch, input_size)
        let input_seq = if self.batch_first {
            Self::transpose_3d(input, input_shape[0], input_shape[1], input_shape[2])?
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
                    input_size
                } else {
                    self.hidden_size * 2 // Previous layer output is concatenated
                };

                // Forward direction (use weights at layer*2)
                let forward_output = self.forward_layer_unidirectional(
                    &layer_input,
                    &h,
                    layer * 2,
                    (seq_len, batch_size, layer_input_size),
                )?;

                // Backward direction (use weights at layer*2+1)
                let reversed_input =
                    Self::reverse_sequence(&layer_input, seq_len, batch_size, layer_input_size)?;
                let backward_output_reversed = self.forward_layer_unidirectional(
                    &reversed_input,
                    &h,
                    layer * 2 + 1,
                    (seq_len, batch_size, layer_input_size),
                )?;
                let backward_output = Self::reverse_sequence(
                    &backward_output_reversed,
                    seq_len,
                    batch_size,
                    self.hidden_size,
                )?;

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

                layer_input = Tensor::from_vec(
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
                let layer_output = self.forward_layer_unidirectional(
                    &layer_input,
                    &h,
                    layer,
                    (seq_len, batch_size, current_input_size),
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

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend, DenseStorage<T>, T>> {
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
        // GRU layers don't have training-specific behavior
    }

    fn name(&self) -> &str {
        "GRU"
    }
}
