//! GRU Module trait implementation.
//!
//! This module implements the Module trait for GRU layers.

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::{ops::arithmetic::add, ops::arithmetic::mul, ops::arithmetic::sub, Tensor};

use super::gru_core::GRU;
use super::gru_forward::GRUForward;
use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Type aliases to reduce complexity
type CpuTensor<T> = Tensor<CpuBackend<T>, DenseStorage<T>, T>;

impl<T> GRUForward<CpuBackend<T>, DenseStorage<T>, T> for GRU<CpuBackend<T>, DenseStorage<T>, T>
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
        let weight_ih_data = self.weight_ih[weight_idx].data().as_slice().to_vec();
        let weight_hh_data = self.weight_hh[weight_idx].data().as_slice().to_vec();

        let weight_ih = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            weight_ih_data,
            &[3 * self.hidden_size, input_size],
        )?;
        let weight_hh = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            weight_hh_data,
            &[3 * self.hidden_size, self.hidden_size],
        )?;

        let mut output_data = Vec::with_capacity(seq_len * batch_size * self.hidden_size);
        let layer_offset = weight_idx * batch_size * self.hidden_size;
        let mut current_hidden = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            h.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
            &[batch_size, self.hidden_size],
        )?;

        for t in 0..seq_len {
            let x_t_start = t * batch_size * input_size;
            let x_t_end = (t + 1) * batch_size * input_size;
            let x_t = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                input.as_slice()[x_t_start..x_t_end].to_vec(),
                &[batch_size, input_size],
            )?;

            let mut ih_gates = x_t.matmul(&weight_ih.transpose(1, 0)?)?;
            let mut hh_gates = current_hidden.matmul(&weight_hh.transpose(1, 0)?)?;

            if self.bias {
                let bias_ih_data = self.bias_ih[weight_idx].data().as_slice();
                let bias_hh_data = self.bias_hh[weight_idx].data().as_slice();
                let b_ih = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    bias_ih_data.to_vec(),
                    &[3 * self.hidden_size],
                )?;
                let b_hh = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    bias_hh_data.to_vec(),
                    &[3 * self.hidden_size],
                )?;
                ih_gates = add(&ih_gates, &b_ih)?;
                hh_gates = add(&hh_gates, &b_hh)?;
            }

            let h = self.hidden_size as i32;
            let r_ih = ih_gates.advanced_slice(&[(None, None, 1), (Some(0), Some(h), 1)])?;
            let z_ih = ih_gates.advanced_slice(&[(None, None, 1), (Some(h), Some(2 * h), 1)])?;
            let n_ih =
                ih_gates.advanced_slice(&[(None, None, 1), (Some(2 * h), Some(3 * h), 1)])?;

            let r_hh = hh_gates.advanced_slice(&[(None, None, 1), (Some(0), Some(h), 1)])?;
            let z_hh = hh_gates.advanced_slice(&[(None, None, 1), (Some(h), Some(2 * h), 1)])?;
            let n_hh =
                hh_gates.advanced_slice(&[(None, None, 1), (Some(2 * h), Some(3 * h), 1)])?;

            let r = crate::functional_api::sigmoid(&add(&r_ih, &r_hh)?)?;
            let z = crate::functional_api::sigmoid(&add(&z_ih, &z_hh)?)?;
            let n = crate::functional_api::tanh(&add(&n_ih, &mul(&r, &n_hh)?)?)?;

            let ones =
                Tensor::<CpuBackend<T>, DenseStorage<T>, T>::ones(current_hidden.shape().dims())?;
            let one_minus_z = sub(&ones, &z)?;
            let n_part = mul(&one_minus_z, &n)?;
            let h_part = mul(&z, &current_hidden)?;
            current_hidden = add(&n_part, &h_part)?;

            output_data.extend_from_slice(current_hidden.as_slice());
        }

        let layer_output = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            output_data,
            &[seq_len, batch_size, self.hidden_size],
        )?;

        Ok(layer_output)
    }
}

impl<T> GRU<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Forward pass with hidden state management.
    pub fn forward_with_hidden(
        &self,
        input: &CpuTensor<T>,
        hidden: Option<&CpuTensor<T>>,
    ) -> Result<(CpuTensor<T>, CpuTensor<T>)> {
        let input_shape = input.shape().dims();
        let (seq_len, batch_size, input_size) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let h = if let Some(h_init) = hidden {
            h_init.clone()
        } else {
            Tensor::zeros(&[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ])?
        };

        let input_seq = if self.batch_first {
            Self::transpose_3d(input, batch_size, seq_len, input_size)?
        } else {
            input.clone()
        };

        let mut layer_input = input_seq;
        let mut h_n_parts = Vec::with_capacity(self.num_layers * num_directions);

        for layer in 0..self.num_layers {
            if self.bidirectional {
                let layer_input_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size * 2
                };

                // Forward
                let forward_output =
                    <Self as GRUForward<CpuBackend<T>, DenseStorage<T>, T>>::forward_layer_unidirectional(
                        self,
                        &layer_input,
                        &h,
                        layer * 2,
                        (seq_len, batch_size, layer_input_size),
                    )?;
                let forward_h = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    forward_output.as_slice()[(seq_len - 1) * batch_size * self.hidden_size
                        ..seq_len * batch_size * self.hidden_size]
                        .to_vec(),
                    &[batch_size, self.hidden_size],
                )?;
                h_n_parts.push(forward_h);

                // Backward
                let reversed_input =
                    Self::reverse_sequence(&layer_input, seq_len, batch_size, layer_input_size)?;
                let backward_output_reversed = <Self as GRUForward<
                    CpuBackend<T>,
                    DenseStorage<T>,
                    T,
                >>::forward_layer_unidirectional(
                    self,
                    &reversed_input,
                    &h,
                    layer * 2 + 1,
                    (seq_len, batch_size, layer_input_size),
                )?;
                let backward_h = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    backward_output_reversed.as_slice()[(seq_len - 1)
                        * batch_size
                        * self.hidden_size
                        ..seq_len * batch_size * self.hidden_size]
                        .to_vec(),
                    &[batch_size, self.hidden_size],
                )?;
                h_n_parts.push(backward_h);

                let backward_output = Self::reverse_sequence(
                    &backward_output_reversed,
                    seq_len,
                    batch_size,
                    self.hidden_size,
                )?;

                let forward_data = forward_output.as_slice();
                let backward_data = backward_output.as_slice();
                let mut concatenated_data =
                    Vec::with_capacity(seq_len * batch_size * self.hidden_size * 2);

                for t in 0..seq_len {
                    for b in 0..batch_size {
                        let forward_start = (t * batch_size + b) * self.hidden_size;
                        let backward_start = (t * batch_size + b) * self.hidden_size;
                        concatenated_data.extend_from_slice(
                            &forward_data[forward_start..forward_start + self.hidden_size],
                        );
                        concatenated_data.extend_from_slice(
                            &backward_data[backward_start..backward_start + self.hidden_size],
                        );
                    }
                }

                layer_input = Tensor::from_vec(
                    concatenated_data,
                    &[seq_len, batch_size, self.hidden_size * 2],
                )?;
            } else {
                let current_input_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size
                };
                let l_output =
                    <Self as GRUForward<CpuBackend<T>, DenseStorage<T>, T>>::forward_layer_unidirectional(
                        self,
                        &layer_input,
                        &h,
                        layer,
                        (seq_len, batch_size, current_input_size),
                    )?;
                let l_h = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    l_output.as_slice()[(seq_len - 1) * batch_size * self.hidden_size
                        ..seq_len * batch_size * self.hidden_size]
                        .to_vec(),
                    &[batch_size, self.hidden_size],
                )?;
                layer_input = l_output;
                h_n_parts.push(l_h);
            }
        }

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

        // Combine hidden states: [num_layers * num_directions, batch_size, hidden_size]
        let mut final_h_data =
            Vec::with_capacity(self.num_layers * num_directions * batch_size * self.hidden_size);
        for part in h_n_parts {
            final_h_data.extend_from_slice(part.as_slice());
        }
        let final_h = Tensor::from_vec(
            final_h_data,
            &[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ],
        )?;

        Ok((output, final_h))
    }

    fn transpose_3d(input: &CpuTensor<T>, d1: usize, d2: usize, d3: usize) -> Result<CpuTensor<T>> {
        let data = input.as_slice();
        let mut transposed_data = Vec::with_capacity(data.len());
        for i in 0..d2 {
            for j in 0..d1 {
                let start = (j * d2 + i) * d3;
                transposed_data.extend_from_slice(&data[start..start + d3]);
            }
        }
        Ok(Tensor::from_vec(transposed_data, &[d2, d1, d3])?)
    }

    fn reverse_sequence(
        input: &CpuTensor<T>,
        seq_len: usize,
        batch_size: usize,
        hidden_size: usize,
    ) -> Result<CpuTensor<T>> {
        let data = input.as_slice();
        let mut reversed_data = Vec::with_capacity(data.len());
        for t in (0..seq_len).rev() {
            let start = t * batch_size * hidden_size;
            reversed_data.extend_from_slice(&data[start..start + batch_size * hidden_size]);
        }
        Ok(Tensor::from_vec(
            reversed_data,
            &[seq_len, batch_size, hidden_size],
        )?)
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T> for GRU<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let (output, _) = self.forward_with_hidden(input, None)?;
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
        // GRU layers don't have training-specific behavior
    }

    fn name(&self) -> &str {
        "GRU"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
