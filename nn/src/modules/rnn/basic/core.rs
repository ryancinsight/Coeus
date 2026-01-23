//! Basic RNN core structures and constructors.

use std::marker::PhantomData;

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec};
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::parameter::Parameter;

pub type CpuTensor<T> = Tensor<CpuBackend<T>, DenseStorage<T>, T>;
pub type TensorPair<T> = (CpuTensor<T>, CpuTensor<T>);

/// Basic Recurrent Neural Network (RNN) layer.
///
/// Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.
#[derive(Debug, Clone)]
pub struct RNN<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + std::cmp::PartialOrd,
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
    pub bias: bool,
    /// Whether input/output tensors are (batch, seq, feature)
    pub batch_first: bool,
    /// Whether this is a bidirectional RNN
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
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> Result<Self> {
        if input_size == 0 || hidden_size == 0 || num_layers == 0 {
            return Err(crate::core::error::NNError::InvalidInput {
                message: "input_size, hidden_size, and num_layers must be > 0".to_string(),
            });
        }

        let num_directions = if bidirectional { 2 } else { 1 };
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..num_layers {
            for _dir in 0..num_directions {
                let layer_input_size = if layer == 0 {
                    input_size
                } else {
                    hidden_size * num_directions
                };
                let w_ih = Self::xavier_uniform_init(hidden_size, layer_input_size, T::one());
                let w_hh = Self::xavier_uniform_init(hidden_size, hidden_size, T::one());

                weight_ih.push(Parameter::new(
                    w_ih.requires_grad_(true),
                    format!("weight_ih_l{}", layer),
                ));
                weight_hh.push(Parameter::new(
                    w_hh.requires_grad_(true),
                    format!("weight_hh_l{}", layer),
                ));

                if bias {
                    bias_ih.push(Parameter::new(
                        Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[hidden_size])?
                            .requires_grad_(true),
                        format!("bias_ih_l{}", layer),
                    ));
                    bias_hh.push(Parameter::new(
                        Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[hidden_size])?
                            .requires_grad_(true),
                        format!("bias_hh_l{}", layer),
                    ));
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

    fn xavier_uniform_init(rows: usize, cols: usize, _limit: T) -> CpuTensor<T> {
        let mut tensor = Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[rows, cols]).unwrap();
        crate::init::xavier_uniform_(&mut tensor, 1.0).unwrap();
        tensor
    }

    pub fn transpose_3d(
        input: &CpuTensor<T>,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> Result<CpuTensor<T>> {
        let data = input.as_slice();
        let mut transposed_data = Vec::with_capacity(data.len());
        for i in 0..d2 {
            for j in 0..d1 {
                let start = (j * d2 + i) * d3;
                transposed_data.extend_from_slice(&data[start..start + d3]);
            }
        }
        Ok(Tensor::from_vec(transposed_data, &[d2, d1, d3])?.requires_grad_(input.requires_grad()))
    }

    pub fn reverse_sequence(
        input: &CpuTensor<T>,
        seq_len: usize,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<CpuTensor<T>> {
        let data = input.as_slice();
        let mut reversed_data = Vec::with_capacity(data.len());
        for t in (0..seq_len).rev() {
            let start = t * batch_size * feature_size;
            reversed_data.extend_from_slice(&data[start..start + batch_size * feature_size]);
        }
        Ok(
            Tensor::from_vec(reversed_data, &[seq_len, batch_size, feature_size])?
                .requires_grad_(input.requires_grad()),
        )
    }
}
