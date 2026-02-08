//! LSTM Module trait implementation.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use super::core::{LstmOutput, LstmState, LSTM};
use super::forward::LSTMForward;
use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

impl<B, S, T> LSTMForward<B, S, T> for LSTM<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward_layer_unidirectional_lstm(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        h: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        c: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        weight_idx: usize,
        dims: (usize, usize, usize),
    ) -> Result<(
        Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        (
            Tensor<CpuBackend<T>, DenseStorage<T>, T>,
            Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        ),
    )> {
        let (seq_len, batch_size, input_size) = dims;

        if seq_len == 0 {
            let empty_output = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[
                0,
                batch_size,
                self.hidden_size,
            ])?;
            let layer_offset = weight_idx * batch_size * self.hidden_size;
            let init_hidden = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                h.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
                &[batch_size, self.hidden_size],
            )?;
            let init_cell = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                c.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
                &[batch_size, self.hidden_size],
            )?;
            return Ok((empty_output, (init_hidden, init_cell)));
        }

        let weight_ih = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            self.weight_ih[weight_idx].data().as_slice().to_vec(),
            &[4 * self.hidden_size, input_size],
        )?;
        let weight_hh = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            self.weight_hh[weight_idx].data().as_slice().to_vec(),
            &[4 * self.hidden_size, self.hidden_size],
        )?;

        let (bias_ih, bias_hh) = if self.bias {
            (
                Some(Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    self.bias_ih[weight_idx].data().as_slice().to_vec(),
                    &[4 * self.hidden_size],
                )?),
                Some(Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    self.bias_hh[weight_idx].data().as_slice().to_vec(),
                    &[4 * self.hidden_size],
                )?),
            )
        } else {
            (None, None)
        };

        let mut output_data = Vec::with_capacity(seq_len * batch_size * self.hidden_size);
        let layer_offset = weight_idx * batch_size * self.hidden_size;
        let mut current_hidden = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            h.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
            &[batch_size, self.hidden_size],
        )?;
        let mut current_cell = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            c.as_slice()[layer_offset..layer_offset + batch_size * self.hidden_size].to_vec(),
            &[batch_size, self.hidden_size],
        )?;

        for t in 0..seq_len {
            let input_start = t * batch_size * input_size;
            let x_t = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                input.as_slice()[input_start..input_start + batch_size * input_size].to_vec(),
                &[batch_size, input_size],
            )?;

            let mut gates = &tensor::ops::matmul(&x_t, &weight_ih.transpose(1, 0)?)?
                + &tensor::ops::matmul(&current_hidden, &weight_hh.transpose(1, 0)?)?;
            if let (Some(ref bi), Some(ref bh)) = (&bias_ih, &bias_hh) {
                gates = &gates + &(bi + bh);
            }

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

            let i = crate::functional_api::sigmoid(
                &Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    i_data,
                    &[batch_size, self.hidden_size],
                )?,
            )?;
            let f = crate::functional_api::sigmoid(
                &Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    f_data,
                    &[batch_size, self.hidden_size],
                )?,
            )?;
            let g = crate::functional_api::tanh(
                &Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    g_data,
                    &[batch_size, self.hidden_size],
                )?,
            )?;
            let o = crate::functional_api::sigmoid(
                &Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                    o_data,
                    &[batch_size, self.hidden_size],
                )?,
            )?;

            current_cell = tensor::ops::add(
                &tensor::ops::mul(&f, &current_cell)?,
                &tensor::ops::mul(&i, &g)?,
            )?;
            current_hidden = tensor::ops::mul(&o, &crate::functional_api::tanh(&current_cell)?)?;
            output_data.extend_from_slice(current_hidden.as_slice());
        }

        let output = Tensor::from_vec(output_data, &[seq_len, batch_size, self.hidden_size])?;
        Ok((output, (current_hidden, current_cell)))
    }
}

impl<B, S, T> LSTM<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    pub fn forward(
        &self,
        input: &Tensor<B, S, T>,
        state: LstmState<'_, B, S, T>,
    ) -> Result<LstmOutput<B, S, T>> {
        let input_shape = input.shape().dims();
        let (seq_len, batch_size, input_size) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

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
            let shape = [
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ];
            (Tensor::zeros(&shape)?, Tensor::zeros(&shape)?)
        };

        let input_seq = if self.batch_first {
            Self::transpose_3d(input, input_shape[0], input_shape[1], input_shape[2])?
        } else {
            input.clone()
        };

        let mut layer_input_cpu = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            input_seq.as_slice().to_vec(),
            input_seq.shape().dims(),
        )?;
        let mut h_n_parts = Vec::new();
        let mut c_n_parts = Vec::new();

        for layer in 0..self.num_layers {
            if self.bidirectional {
                let layer_input_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size * 2
                };
                let (f_out, (f_h, f_c)) = self.forward_layer_unidirectional_lstm(
                    &layer_input_cpu,
                    &h_cpu,
                    &c_cpu,
                    layer * 2,
                    (seq_len, batch_size, layer_input_size),
                )?;
                h_n_parts.push(f_h);
                c_n_parts.push(f_c);

                let rev_in = LSTM::<CpuBackend<T>, DenseStorage<T>, T>::reverse_sequence(
                    &layer_input_cpu,
                    seq_len,
                    batch_size,
                    layer_input_size,
                )?;
                let (b_out_rev, (b_h, b_c)) = self.forward_layer_unidirectional_lstm(
                    &rev_in,
                    &h_cpu,
                    &c_cpu,
                    layer * 2 + 1,
                    (seq_len, batch_size, layer_input_size),
                )?;
                h_n_parts.push(b_h);
                c_n_parts.push(b_c);

                let b_out = LSTM::<CpuBackend<T>, DenseStorage<T>, T>::reverse_sequence(
                    &b_out_rev,
                    seq_len,
                    batch_size,
                    self.hidden_size,
                )?;

                let mut concat = Vec::with_capacity(seq_len * batch_size * self.hidden_size * 2);
                let fd = f_out.as_slice();
                let bd = b_out.as_slice();
                for t in 0..seq_len {
                    for b in 0..batch_size {
                        let offset = (t * batch_size + b) * self.hidden_size;
                        concat.extend_from_slice(&fd[offset..offset + self.hidden_size]);
                        concat.extend_from_slice(&bd[offset..offset + self.hidden_size]);
                    }
                }
                layer_input_cpu =
                    Tensor::from_vec(concat, &[seq_len, batch_size, self.hidden_size * 2])?;
            } else {
                let current_input_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size
                };
                let (l_out, (l_h, l_c)) = self.forward_layer_unidirectional_lstm(
                    &layer_input_cpu,
                    &h_cpu,
                    &c_cpu,
                    layer,
                    (seq_len, batch_size, current_input_size),
                )?;
                h_n_parts.push(l_h);
                c_n_parts.push(l_c);
                layer_input_cpu = l_out;
            }
        }

        let layer_input_generic = Tensor::<B, S, T>::from_vec(
            layer_input_cpu.as_slice().to_vec(),
            layer_input_cpu.shape().dims(),
        )?;
        let output = if self.batch_first {
            let out_h = if self.bidirectional {
                self.hidden_size * 2
            } else {
                self.hidden_size
            };
            Self::transpose_3d(&layer_input_generic, seq_len, batch_size, out_h)?
        } else {
            layer_input_generic
        };

        let mut fh = Vec::new();
        for p in h_n_parts {
            fh.extend_from_slice(p.as_slice());
        }
        let mut fc = Vec::new();
        for p in c_n_parts {
            fc.extend_from_slice(p.as_slice());
        }
        let h_n = Tensor::from_vec(
            fh,
            &[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ],
        )?;
        let c_n = Tensor::from_vec(
            fc,
            &[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ],
        )?;

        Ok((output, (h_n, c_n)))
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T> for LSTM<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt,
{
    type Input = Tensor<CpuBackend<T>, DenseStorage<T>, T>;
    type Output = Tensor<CpuBackend<T>, DenseStorage<T>, T>;

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
        for p in self
            .weight_ih
            .iter_mut()
            .chain(self.weight_hh.iter_mut())
            .chain(self.bias_ih.iter_mut())
            .chain(self.bias_hh.iter_mut())
        {
            p.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {}
    fn name(&self) -> &str {
        "LSTM"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
