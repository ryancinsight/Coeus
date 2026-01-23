//! LSTM (Long Short-Term Memory) core structures and constructors.

use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::parameter::Parameter;

pub type LstmState<'a, B, S, T> = Option<(&'a Tensor<B, S, T>, &'a Tensor<B, S, T>)>;
pub type LstmOutput<B, S, T> = (Tensor<B, S, T>, (Tensor<B, S, T>, Tensor<B, S, T>));

/// LSTM (Long Short-Term Memory) layer for sequence modeling.
///
/// Implements the LSTM architecture with forget, input, output, and candidate gates.
/// Provides better gradient flow than basic RNNs for long sequences.
#[derive(Debug, Clone)]
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
    pub bias: bool,
    /// Whether input/output tensors are (batch, seq, feature)
    pub batch_first: bool,
    /// Whether this is a bidirectional LSTM
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
                let gate_size = 4 * hidden_size;

                let limit = (T::from(6.0).unwrap()
                    / T::from(layer_input_size + hidden_size).unwrap())
                .sqrt();
                let w_ih = Self::xavier_uniform_init(gate_size, layer_input_size, limit);
                let w_hh = Self::xavier_uniform_init(gate_size, hidden_size, limit);

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
                        Tensor::<B, S, T>::zeros(&[gate_size])?.requires_grad_(true),
                        format!("bias_ih_l{}", layer),
                    ));
                    bias_hh.push(Parameter::new(
                        Tensor::<B, S, T>::zeros(&[gate_size])?.requires_grad_(true),
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

    fn xavier_uniform_init(rows: usize, cols: usize, _limit: T) -> Tensor<B, S, T> {
        let mut tensor = Tensor::<B, S, T>::zeros(&[rows, cols]).unwrap();
        crate::init::xavier_uniform_(&mut tensor, 1.0).unwrap();
        tensor
    }

    pub fn transpose_3d(
        input: &Tensor<B, S, T>,
        dim0: usize,
        dim1: usize,
        dim2: usize,
    ) -> Result<Tensor<B, S, T>> {
        let input_data = input.as_slice();
        let mut transposed_data = Vec::with_capacity(dim0 * dim1 * dim2);
        for b in 0..dim1 {
            for t in 0..dim0 {
                let start = (t * dim1 + b) * dim2;
                transposed_data.extend_from_slice(&input_data[start..start + dim2]);
            }
        }
        Ok(Tensor::from_vec(transposed_data, &[dim1, dim0, dim2])?)
    }

    pub fn reverse_sequence(
        input: &Tensor<B, S, T>,
        seq_len: usize,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<Tensor<B, S, T>> {
        let input_data = input.as_slice();
        let mut reversed_data = Vec::with_capacity(seq_len * batch_size * feature_size);
        for t in (0..seq_len).rev() {
            let start = t * batch_size * feature_size;
            reversed_data.extend_from_slice(&input_data[start..start + batch_size * feature_size]);
        }
        Ok(Tensor::from_vec(
            reversed_data,
            &[seq_len, batch_size, feature_size],
        )?)
    }
}
