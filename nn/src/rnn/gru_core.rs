//! GRU (Gated Recurrent Unit) core structures and constructors.
//!
//! This module contains the GRU struct definition and basic construction/initialization methods.

use std::marker::PhantomData;

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec};
use tensor::Tensor;

use crate::error::Result;
use crate::parameter::Parameter;

/// GRU (Gated Recurrent Unit) layer for sequence modeling.
///
/// Implements the GRU architecture with reset and update gates.
/// More efficient than LSTM while maintaining similar performance.
#[derive(Debug)]
pub struct GRU<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + std::cmp::PartialOrd,
{
    /// Input-to-hidden weights for each gate (r, z, n) and layer
    pub weight_ih: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Hidden-to-hidden weights for each gate (r, z, n) and layer
    pub weight_hh: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Input-to-hidden biases for each gate (r, z, n) and layer
    pub bias_ih: Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>>,
    /// Hidden-to-hidden biases for each gate (r, z, n) and layer
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
    /// Whether this is a bidirectional GRU
    #[allow(dead_code)]
    pub bidirectional: bool,
    /// Phantom data to ensure B and S are used for type safety
    _phantom: PhantomData<(B, S)>,
}

impl<B, S, T> GRU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new GRU layer.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input
    /// * `hidden_size` - The number of features in the hidden state
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `bias` - If false, the layer does not use bias weights (default: true)
    /// * `batch_first` - If true, input/output tensors are (batch, seq, feature) (default: false)
    /// * `bidirectional` - If true, becomes a bidirectional GRU (default: false)
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

                // GRU has 3 gates (r, z, n), so weights are 3x larger
                let gate_size = 3 * hidden_size;

                // Xavier/Glorot uniform initialization
                let limit = (T::from(6.0).unwrap()
                    / T::from(layer_input_size + hidden_size).unwrap())
                .sqrt();
                let w_ih = Self::xavier_uniform_init(gate_size, layer_input_size, limit);
                let w_hh = Self::xavier_uniform_init(gate_size, hidden_size, limit);

                let weight_ih_var = w_ih.requires_grad_(true);
                let weight_hh_var = w_hh.requires_grad_(true);

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
                        Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[gate_size]).unwrap();
                    let b_hh =
                        Tensor::<CpuBackend<T>, DenseStorage<T>, T>::zeros(&[gate_size]).unwrap();

                    let bias_ih_var = b_ih.requires_grad_(true);
                    let bias_hh_var = b_hh.requires_grad_(true);

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
