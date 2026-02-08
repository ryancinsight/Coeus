//! Transformer Encoder Block.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use std::marker::PhantomData;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::{Module, Parameter};
use crate::ops::activation;
use crate::modules::attention::MultiHeadAttention;
use crate::modules::linear::Linear;
use crate::modules::normalization::LayerNorm;
use crate::modules::regularization::dropout::Dropout;

/// Transformer Encoder Block.
#[derive(Debug, Clone)]
pub struct TransformerEncoder<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + std::cmp::PartialOrd,
{
    pub self_attn: MultiHeadAttention<B, S, T>,
    pub norm1: LayerNorm<B, S, T>,
    pub linear1: Linear<B, S, T>,
    pub linear2: Linear<B, S, T>,
    pub norm2: LayerNorm<B, S, T>,
    pub dropout: Dropout,
    pub d_model: usize,
    pub nhead: usize,
    pub dim_feedforward: usize,
    pub dropout_p: f64,
    training: bool,
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> TransformerEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive + num_traits::Zero,
    Dropout: Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>,
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
            self_attn: MultiHeadAttention::<B, S, T>::new(
                d_model, nhead,
            )?,
            norm1: LayerNorm::<B, S, T>::new(vec![d_model], 1e-5),
            linear1: Linear::<B, S, T>::new(d_model, dim_feedforward)
                .unwrap(),
            linear2: Linear::<B, S, T>::new(dim_feedforward, d_model)
                .unwrap(),
            norm2: LayerNorm::<B, S, T>::new(vec![d_model], 1e-5),
            dropout: Dropout::new(dropout),
            d_model,
            nhead,
            dim_feedforward,
            dropout_p: dropout,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<B, S, T> Module<B, S, T> for TransformerEncoder<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd + num_traits::FromPrimitive,
    Dropout: Module<B, S, T, Input = Tensor<B, S, T>, Output = Tensor<B, S, T>>,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let x = input;
        let attn_output = self.self_attn.forward(x)?;
        let norm1_output = self.norm1.forward(&(x + &attn_output))?;

        let [batch, seq, d_model] = [
            norm1_output.shape().dims()[0],
            norm1_output.shape().dims()[1],
            norm1_output.shape().dims()[2],
        ];

        let reshaped = norm1_output.reshape(&[(batch * seq) as isize, d_model as isize])?;
        // Convert back to generic storage S
        let l1_input = Tensor::<B, S, T>::from_vec(reshaped.as_slice().to_vec(), reshaped.shape().dims())?;
        let l1_out = self.linear1.forward(&l1_input)?;
        let relu_out = activation::relu(&l1_out)?;
        let l2_out = self.linear2.forward(&relu_out)?;

        let ff_out_2d = if self.training {
            Module::<B, S, T>::forward(&self.dropout, &l2_out)?
        } else {
            l2_out
        };
        let ff_output = ff_out_2d.reshape(&[batch as isize, seq as isize, d_model as isize])?;

        self.norm2.forward(&(&norm1_output + &ff_output))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        let mut p = Vec::new();
        p.extend(self.self_attn.parameters());
        p.extend(self.norm1.parameters());
        p.extend(self.linear1.parameters());
        p.extend(self.linear2.parameters());
        p.extend(self.norm2.parameters());
        p
    }

    fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.norm1.zero_grad();
        self.linear1.zero_grad();
        self.linear2.zero_grad();
        self.norm2.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    fn name(&self) -> &str {
        "TransformerEncoder"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

impl<B, S, T> std::fmt::Display for TransformerEncoder<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + std::cmp::PartialOrd,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerEncoder(d_model={}, nhead={}, dim_feedforward={}, dropout={})",
            self.d_model, self.nhead, self.dim_feedforward, self.dropout_p
        )
    }
}
