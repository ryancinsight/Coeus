use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── SinusoidalEncoding ───────────────────────────────────────────────────────

/// Python-exposed sinusoidal positional encoding.
///
/// ```python
/// pe = pycoeus.SinusoidalEncoding(max_len=512, d_model=64)
/// out = pe.forward(embeddings)   # [batch, seq, d_model]
/// ```
#[pyclass(name = "SinusoidalEncoding")]
pub struct PySinusoidalEncoding {
    /// Maximum sequence length for which encodings are pre-computed.
    #[pyo3(get)]
    pub max_len: usize,
    /// Model embedding dimensionality (must be positive and even).
    #[pyo3(get)]
    pub d_model: usize,
}

#[pymethods]
impl PySinusoidalEncoding {
    #[new]
    /// Create a SinusoidalEncoding table of shape `[max_len, d_model]`.
    pub fn new(max_len: usize, d_model: usize) -> PyResult<Self> {
        if d_model == 0 || !d_model.is_multiple_of(2) {
            return Err(PyValueError::new_err(
                "SinusoidalEncoding: d_model must be a positive even integer",
            ));
        }
        Ok(Self { max_len, d_model })
    }

    /// Add sinusoidal positional encoding to `input`.
    ///
    /// - `input`: `[batch, seq_len, d_model]`
    ///
    /// Returns `[batch, seq_len, d_model]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let x = input.inner.clone();
        let (max_len, d_model) = (self.max_len, self.d_model);
        let inner = py.allow_threads(move || {
            use coeus_nn::positional::sinusoidal::SinusoidalEncoding;
            use coeus_nn::Module;
            let pe = SinusoidalEncoding::<f64, coeus_core::MoiraiBackend>::new(max_len, d_model);
            pe.forward(&x)
        });
        Ok(PyTensor::from_var(inner))
    }
}
