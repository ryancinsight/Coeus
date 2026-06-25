use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-exposed FeedForward (2-layer MLP) transformer sub-block.
#[pyclass(name = "FeedForward")]
pub struct PyFeedForward {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout_p: f64,
}

#[pymethods]
impl PyFeedForward {
    /// Create a FeedForward sub-layer.
    ///
    /// Computes `Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)`.
    #[new]
    #[pyo3(signature = (d_model, d_ff, dropout_p = 0.0))]
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "FeedForward: dropout_p must be in [0.0, 1.0)",
            ));
        }
        Ok(Self {
            d_model,
            d_ff,
            dropout_p,
        })
    }

    /// Forward pass through the feed-forward block.
    ///
    /// Input: `[batch, seq, d_model]` or `[batch, d_model]`.
    /// Output: same shape.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::transformer::ffn::FeedForward;
        use coeus_nn::Module;
        let (d_model, d_ff, dropout_p) = (self.d_model, self.d_ff, self.dropout_p);
        let x = input.inner.clone();
        let inner = py.allow_threads(move || {
            let ffn = FeedForward::<f64, coeus_core::MoiraiBackend>::new(d_model, d_ff, dropout_p);
            ffn.forward(&x)
        });
        Ok(PyTensor::from_var(inner))
    }
}
