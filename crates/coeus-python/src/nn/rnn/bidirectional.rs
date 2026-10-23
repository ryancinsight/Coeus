//! Python-exposed Bidirectional RNN wrapper.

use crate::error::map_module_error;
use coeus_nn::Module as _;
use pyo3::prelude::*;

/// Python-exposed Bidirectional RNN wrapper.
///
/// Wraps a sequence module and runs it forward + backward (time-reversed),
/// concatenating outputs to `[batch, seq, 2*hidden]`.
#[pyclass(name = "Bidirectional")]
pub struct PyBidirectional {
    /// Forward-direction LSTM.
    inner_fwd: coeus_nn::rnn::Lstm<f64, coeus_core::MoiraiBackend>,
    /// Backward-direction LSTM.
    inner_bwd: coeus_nn::rnn::Lstm<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyBidirectional {
    /// Create a Bidirectional LSTM with independent forward/backward weights.
    #[new]
    pub fn new(input_size: usize, hidden_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner_fwd: coeus_nn::rnn::Lstm::<f64, coeus_core::MoiraiBackend>::new(
                input_size,
                hidden_size,
            )
            .map_err(crate::init::map_initialization_error)?,
            inner_bwd: coeus_nn::rnn::Lstm::<f64, coeus_core::MoiraiBackend>::new(
                input_size,
                hidden_size,
            )
            .map_err(crate::init::map_initialization_error)?,
        })
    }

    /// Forward: `x [N, T, D_in]` → `[N, T, 2*D_hidden]`.
    pub fn forward(
        &self,
        input: &crate::tensor::PyTensor,
        py: Python<'_>,
    ) -> PyResult<crate::tensor::PyTensor> {
        let input_var = input.inner.clone();
        let bi = coeus_nn::rnn::Bidirectional::new(self.inner_fwd.clone(), self.inner_bwd.clone());
        let out = py.allow_threads(move || bi.forward(&input_var));
        out.map(crate::tensor::PyTensor::from_var)
            .map_err(map_module_error)
    }
}
