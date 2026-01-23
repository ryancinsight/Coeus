//! Transformer Decoder Block.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use storage::{DenseStorage, Storage, StorageFromVec};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use crate::ops::activation;
use crate::modules::attention::MultiHeadAttention;
use crate::modules::linear::Linear;
use crate::modules::normalization::LayerNorm;
use crate::modules::regularization::dropout::Dropout;

/// Transformer Decoder Block.
#[derive(Clone, Debug)]
pub struct TransformerDecoder<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType + std::cmp::PartialOrd,
{
    pub self_attn: MultiHeadAttention<CpuBackend<T>, DenseStorage<T>, T>,
    pub cross_attn: MultiHeadAttention<CpuBackend<T>, DenseStorage<T>, T>,
    pub norm1: LayerNorm<CpuBackend<T>, DenseStorage<T>, T>,
    pub norm2: LayerNorm<CpuBackend<T>, DenseStorage<T>, T>,
    pub linear1: Linear<CpuBackend<T>, DenseStorage<T>, T>,
    pub linear2: Linear<CpuBackend<T>, DenseStorage<T>, T>,
    pub norm3: LayerNorm<CpuBackend<T>, DenseStorage<T>, T>,
    pub dropout: Dropout,
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub dropout_p: f64,
    training: bool,
    memory: Option<Tensor<CpuBackend<T>, DenseStorage<T>, T>>,
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> TransformerDecoder<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
{
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, dropout: f64) -> Result<Self> {
        if d_model == 0 || nhead == 0 || dim_feedforward == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "Parameters must be > 0".to_string(),
            });
        }
        if d_model % nhead != 0 {
            return Err(NNError::InvalidConfiguration {
                message: format!(
                    "d_model ({}) must be divisible by nhead ({})",
                    d_model, nhead
                ),
            });
        }

        Ok(Self {
            self_attn: MultiHeadAttention::<CpuBackend<T>, DenseStorage<T>, T>::new(
                d_model, nhead,
            )?,
            cross_attn: MultiHeadAttention::<CpuBackend<T>, DenseStorage<T>, T>::new(
                d_model, nhead,
            )?,
            norm1: LayerNorm::<CpuBackend<T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5),
            norm2: LayerNorm::<CpuBackend<T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5),
            linear1: Linear::<CpuBackend<T>, DenseStorage<T>, T>::new(d_model, dim_feedforward)
                .unwrap(),
            linear2: Linear::<CpuBackend<T>, DenseStorage<T>, T>::new(dim_feedforward, d_model)
                .unwrap(),
            norm3: LayerNorm::<CpuBackend<T>, DenseStorage<T>, T>::new(vec![d_model], 1e-5),
            dropout: Dropout::new(dropout),
            d_model,
            nhead,
            dim_feedforward,
            dropout_p: dropout,
            training: true,
            memory: None,
            _phantom: PhantomData,
        })
    }

    pub fn set_memory(&mut self, memory: Option<Tensor<CpuBackend<T>, DenseStorage<T>, T>>) {
        self.memory = memory;
    }

    pub fn forward_with_memory(
        &self,
        tgt: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        memory: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let tgt_output = self.self_attn.forward(tgt)?;
        let norm1_output = self.norm1.forward(&(tgt + &tgt_output))?;

        let cross_attn_output =
            self.cross_attn
                .forward_cross_attention(&norm1_output, memory, memory)?;
        let norm2_output = self.norm2.forward(&(&norm1_output + &cross_attn_output))?;

        let [batch, seq, d_model] = [
            norm2_output.shape().dims()[0],
            norm2_output.shape().dims()[1],
            norm2_output.shape().dims()[2],
        ];
        let reshaped = norm2_output.reshape(&[(batch * seq) as isize, d_model as isize])?;
        let l1_out = self.linear1.forward(&reshaped)?;
        let relu_out = activation::relu(&l1_out)?;
        let l2_out = self.linear2.forward(&relu_out)?;

        let ff_out_2d = if self.training {
            self.dropout.forward(&l2_out)?
        } else {
            l2_out
        };
        let ff_output = ff_out_2d.reshape(&[batch as isize, seq as isize, d_model as isize])?;
        self.norm3.forward(&(&norm2_output + &ff_output))
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T>
    for TransformerDecoder<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_dense = input.to_dense_generic()?;
        let self_attn_output = self.self_attn.forward(&input_dense)?;
        let norm1_output = self.norm1.forward(&(&input_dense + &self_attn_output))?;

        let cross_attn_output = if let Some(ref memory) = self.memory {
            self.cross_attn
                .forward_cross_attention(&norm1_output, memory, memory)?
        } else {
            self.cross_attn
                .forward_cross_attention(&norm1_output, &norm1_output, &norm1_output)?
        };

        let norm2_output = self.norm2.forward(&(&norm1_output + &cross_attn_output))?;

        let [batch, seq, d_model] = [
            norm2_output.shape().dims()[0],
            norm2_output.shape().dims()[1],
            norm2_output.shape().dims()[2],
        ];
        let reshaped = norm2_output.reshape(&[(batch * seq) as isize, d_model as isize])?;
        let l1_out = self.linear1.forward(&reshaped)?;
        let relu_out = activation::relu(&l1_out)?;
        let l2_out = self.linear2.forward(&relu_out)?;

        let ff_output = l2_out.reshape(&[batch as isize, seq as isize, d_model as isize])?;
        self.norm3.forward(&(&norm2_output + &ff_output))
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        let mut p = Vec::new();
        p.extend(self.self_attn.parameters());
        p.extend(self.cross_attn.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p.extend(self.norm3.parameters());
        p
    }

    fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.cross_attn.zero_grad();
        self.norm1.zero_grad();
        self.norm2.zero_grad();
        self.linear1.zero_grad();
        self.linear2.zero_grad();
        self.norm3.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    fn name(&self) -> &str {
        "TransformerDecoder"
    }
    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}

impl<B, S, T> std::fmt::Display for TransformerDecoder<B, S, T>
where
    B: Backend,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType + std::cmp::PartialOrd,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerDecoder(d_model={}, nhead={}, dim_feedforward={}, dropout={})",
            self.d_model, self.nhead, self.dim_feedforward, self.dropout_p
        )
    }
}
