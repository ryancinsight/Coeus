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
    /// Create an RMSNorm layer normalizing over `normalized_shape` dimensions.
    ///
    /// Mirrors `torch.nn.RMSNorm`-style argument conventions (Coeus parallels
    /// PyTorch numerics even where the upstream surface differs): accepts an
    /// `int` (`RMSNorm(8)`) or a length-1 sequence (`RMSNorm([8])`).  Multi-dim
    /// normalization is not supported by the Rust core; the binding reduces a
    /// single-element sequence and rejects longer entries with NotImplementedError.
    #[pyo3(signature = (normalized_shape, eps=None))]
    pub fn new(
        py: Python<'_>,
        normalized_shape: &Bound<'_, PyAny>,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-8);
        let shape_int: usize = match normalized_shape.extract() {
            Ok(v) => v,
            Err(_) => {
                let seq: Vec<usize> = normalized_shape.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "RMSNorm: normalized_shape must be int or sequence of ints",
                    )
                })?;
                if seq.len() != 1 {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                        "RMSNorm: multi-dim normalized_shape {seq:?} not supported"
                    )));
                }
                seq[0]
            }
        };
        let rms = coeus_nn::normalization::rmsnorm::RMSNorm::<f64, coeus_core::MoiraiBackend>::new(
            shape_int, eps,
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
