use pyo3::prelude::*;
use crate::tensor::{PyTensor, PyStateDict};

/// Python-exposed Dropout layer.
#[pyclass(name = "Dropout")]
pub struct PyDropout {
    #[pyo3(get)]
    pub p: f64,
    #[pyo3(get)]
    pub is_training: bool,
    #[pyo3(get)]
    pub seed: u64,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p = 0.5))]
    pub fn new(p: f64) -> Self {
        Self {
            p,
            is_training: true,
            seed: 42,
        }
    }

    /// Set training mode on the Dropout layer.
    pub fn train(&mut self, mode: bool) {
        self.is_training = mode;
    }

    /// Forward pass through the Dropout layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let rust_dropout = coeus_nn::Dropout {
            p: self.p,
            is_training: self.is_training,
            seed: self.seed,
        };

        let inner = py.allow_threads(move || {
            rust_dropout.forward(&input_var)
        });
        Ok(PyTensor { inner })
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict { inner: coeus_tensor::checkpoint::StateDict::new() }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }
}
