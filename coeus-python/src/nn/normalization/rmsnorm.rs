use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed RMS Normalization layer.
#[pyclass(name = "RMSNorm")]
pub struct PyRMSNorm {
    /// Learnable scale (gamma), shape `[normalized_shape]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Numerical stability epsilon added to the RMS denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-8))]
    /// Create an RMSNorm layer normalizing over `normalized_shape` dimensions.
    pub fn new(py: Python<'_>, normalized_shape: usize, eps: f64) -> PyResult<Self> {
        let rms = coeus_nn::normalization::rmsnorm::RMSNorm::<f64, coeus_core::MoiraiBackend>::new(
            normalized_shape,
            eps,
        );
        let weight = Py::new(py, PyTensor { inner: rms.weight })?;
        Ok(Self { weight, eps })
    }

    /// Forward pass through the RMSNorm layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let eps_val = self.eps;

        let inner = py.allow_threads(move || {
            let rms = coeus_nn::normalization::rmsnorm::RMSNorm::from_parts(w_var, eps_val);
            rms.forward(&input_var)
        });
        Ok(PyTensor::from_var(inner))
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py)]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
    }
}
