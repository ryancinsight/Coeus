//! Basic RNN Module trait implementation.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec};
use tensor::Tensor;

use super::core::{CpuTensor, TensorPair, RNN};
use super::forward::RNNForward;
use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

impl<B, S, T> RNNForward<B, S, T> for RNN<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + storage::StorageToDense<T> + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::cmp::PartialOrd,
{
    fn forward_layer(
        &self,
        input: &CpuTensor<T>,
        hidden: &CpuTensor<T>,
        weight_idx: usize,
        seq_len: usize,
        batch_size: usize,
        input_size: usize,
    ) -> Result<TensorPair<T>> {
        let weight_ih = self.weight_ih[weight_idx].data().clone();
        let weight_hh = self.weight_hh[weight_idx].data().clone();

        let bias_ih = self.bias.then(|| self.bias_ih[weight_idx].data().clone());
        let bias_hh = self.bias.then(|| self.bias_hh[weight_idx].data().clone());

        let requires_grad = input.requires_grad()
            || weight_ih.requires_grad()
            || weight_hh.requires_grad()
            || bias_ih.as_ref().is_some_and(Tensor::requires_grad)
            || bias_hh.as_ref().is_some_and(Tensor::requires_grad);
        let weight_ih_t = weight_ih.transpose(1, 0)?.requires_grad_(requires_grad);

        let mut output_data = Vec::with_capacity(seq_len * batch_size * self.hidden_size);
        let h_slice = hidden.as_slice();
        let h_start = weight_idx * batch_size * self.hidden_size;
        let mut current_hidden = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            h_slice[h_start..h_start + batch_size * self.hidden_size].to_vec(),
            &[batch_size, self.hidden_size],
        )?
        .requires_grad_(requires_grad);

        for t in 0..seq_len {
            let x_t = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
                input.as_slice()[t * batch_size * input_size..(t + 1) * batch_size * input_size]
                    .to_vec(),
                &[batch_size, input_size],
            )?
            .requires_grad_(requires_grad);

            let ih_out = tensor::ops::matmul(&x_t, &weight_ih_t)?.requires_grad_(requires_grad);
            let ih_with_bias = if let Some(ref b) = bias_ih {
                &ih_out + b
            } else {
                ih_out
            };
            let hh_out = tensor::ops::matmul(&current_hidden, &weight_hh)?
                .requires_grad_(requires_grad);
            let hh_with_bias = if let Some(ref b) = bias_hh {
                &hh_out + b
            } else {
                hh_out
            };

            current_hidden = crate::functional_api::tanh(&(&ih_with_bias + &hh_with_bias))?
                .requires_grad_(requires_grad);
            output_data.extend_from_slice(current_hidden.as_slice());
        }

        let layer_output = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::from_vec(
            output_data,
            &[seq_len, batch_size, self.hidden_size],
        )?
        .requires_grad_(requires_grad);
        Ok((layer_output, current_hidden.requires_grad_(requires_grad)))
    }
}

impl<B, S, T> RNN<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + storage::StorageToDense<T> + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::cmp::PartialOrd,
{
    pub fn forward_with_hidden(
        &self,
        input: &CpuTensor<T>,
        hidden: Option<&CpuTensor<T>>,
    ) -> Result<TensorPair<T>> {
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

        let mut layer_input = if self.batch_first {
            Self::transpose_3d(input, batch_size, seq_len, input_size)?
        } else {
            input.clone()
        };

        let mut h_n_parts = Vec::new();
        for layer in 0..self.num_layers {
            if self.bidirectional {
                let l_in_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size * 2
                };
                let (f_out, f_h) = self.forward_layer(
                    &layer_input,
                    &h,
                    layer * 2,
                    seq_len,
                    batch_size,
                    l_in_size,
                )?;
                h_n_parts.push(f_h);

                let rev_in = Self::reverse_sequence(&layer_input, seq_len, batch_size, l_in_size)?;
                let (b_out_rev, b_h) =
                    self.forward_layer(&rev_in, &h, layer * 2 + 1, seq_len, batch_size, l_in_size)?;
                h_n_parts.push(b_h);

                let b_out =
                    Self::reverse_sequence(&b_out_rev, seq_len, batch_size, self.hidden_size)?;
                let fd = f_out.as_slice();
                let bd = b_out.as_slice();
                let mut concat = Vec::with_capacity(seq_len * batch_size * self.hidden_size * 2);
                for t in 0..seq_len {
                    for b in 0..batch_size {
                        let offset = (t * batch_size + b) * self.hidden_size;
                        concat.extend_from_slice(&fd[offset..offset + self.hidden_size]);
                        concat.extend_from_slice(&bd[offset..offset + self.hidden_size]);
                    }
                }
                layer_input =
                    Tensor::from_vec(concat, &[seq_len, batch_size, self.hidden_size * 2])?
                        .requires_grad_(layer_input.requires_grad());
            } else {
                let l_in_size = if layer == 0 {
                    input_size
                } else {
                    self.hidden_size
                };
                let (l_out, l_h) =
                    self.forward_layer(&layer_input, &h, layer, seq_len, batch_size, l_in_size)?;
                layer_input = l_out;
                h_n_parts.push(l_h);
            }
        }

        let output = if self.batch_first {
            let out_h = if self.bidirectional {
                self.hidden_size * 2
            } else {
                self.hidden_size
            };
            Self::transpose_3d(&layer_input, seq_len, batch_size, out_h)?
        } else {
            layer_input
        };

        let mut fh = Vec::new();
        for p in h_n_parts {
            fh.extend_from_slice(p.as_slice());
        }
        let final_h = Tensor::from_vec(
            fh,
            &[
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            ],
        )?
        .requires_grad_(output.requires_grad());

        Ok((output, final_h))
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T> for RNN<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + std::cmp::PartialOrd + num_traits::Float,
{
    fn forward(&self, input: &CpuTensor<T>) -> Result<CpuTensor<T>> {
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
        "RNN"
    }
    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
