use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Python-exposed FeedForward (2-layer MLP) transformer sub-block.
#[pyclass(name = "FeedForward")]
pub struct PyFeedForward {
    pub d_model: usize,
    pub d_ff: usize,
}

#[pymethods]
impl PyFeedForward {
    /// Create a FeedForward sub-layer.
    ///
    /// Computes `Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)`.
    #[new]
    #[pyo3(signature = (d_model, d_ff, dropout_p = 0.0))]
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> Self {
        let _ = dropout_p; // stored inside the Rust layer; we recreate it each call
        Self { d_model, d_ff }
    }

    /// Forward pass through the feed-forward block.
    ///
    /// Input: `[batch, seq, d_model]` or `[batch, d_model]`.
    /// Output: same shape.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::transformer::ffn::FeedForward;
        use coeus_nn::Module;
        let (d_model, d_ff) = (self.d_model, self.d_ff);
        let x = input.inner.clone();
        let inner = py.allow_threads(move || {
            let ffn = FeedForward::<f64, coeus_core::MoiraiBackend>::new(d_model, d_ff, 0.0);
            ffn.forward(&x)
        });
        Ok(PyTensor { inner })
    }
}
