use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-exposed LocalResponseNorm (cross-channel LRN) layer.
///
/// Mirrors `torch.nn.LocalResponseNorm(size)`: defaults `alpha=1e-4`,
/// `beta=0.75`, `k=1.0`. Operates on `[N, C, *spatial]`, normalizing each
/// activation by the response of its `size` neighbouring channels. Has no
/// learnable parameters.
///
/// ```python
/// lrn = pycoeus.LocalResponseNorm(size=5)
/// y = lrn.forward(x)   # x: [N, C, H, W]
/// ```
#[pyclass(name = "LocalResponseNorm")]
pub struct PyLocalResponseNorm {
    /// Number of neighbouring channels summed over.
    #[pyo3(get)]
    pub size: usize,
    /// Multiplicative factor.
    #[pyo3(get)]
    pub alpha: f64,
    /// Exponent.
    #[pyo3(get)]
    pub beta: f64,
    /// Additive constant.
    #[pyo3(get)]
    pub k: f64,
}

#[pymethods]
impl PyLocalResponseNorm {
    #[new]
    #[pyo3(signature = (size, alpha = 0.0001, beta = 0.75, k = 1.0))]
    /// Create an LRN layer over `size` neighbouring channels.
    pub fn new(size: usize, alpha: f64, beta: f64, k: f64) -> PyResult<Self> {
        if size < 1 {
            return Err(PyValueError::new_err(
                "LocalResponseNorm: size must be >= 1",
            ));
        }
        Ok(Self {
            size,
            alpha,
            beta,
            k,
        })
    }

    /// Forward pass: cross-channel local response normalization.
    ///
    /// Differentiable (autograd-graph forward in coeus-nn), so gradients flow to
    /// the input and the layer is trainable.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let x = input.inner.clone();
        let lrn =
            coeus_nn::LocalResponseNorm::with_params(self.size, self.alpha, self.beta, self.k);
        let out = py.allow_threads(move || lrn.forward(&x));
        Ok(PyTensor::from_var(out))
    }

    /// LocalResponseNorm has no learnable parameters.
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// No-op: LocalResponseNorm has no parameters.
    pub fn zero_grad(&self, _py: Python<'_>) {}
}
