use pyo3::prelude::*;
use crate::tensor::{PyTensor, PyStateDict};

/// Python-exposed Linear layer.
#[pyclass(name = "Linear")]
pub struct PyLinear {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub bias: Option<Py<PyTensor>>,
}

#[pymethods]
impl PyLinear {
    /// Create a Linear layer.
    #[new]
    #[pyo3(signature = (in_features, out_features, bias = true))]
    fn new(py: Python<'_>, in_features: usize, out_features: usize, bias: bool) -> PyResult<Self> {
        let linear = coeus_nn::linear::Linear::new(in_features, out_features, bias);
        let weight = Py::new(py, PyTensor { inner: linear.weight })?;
        let bias = if let Some(b) = linear.bias {
            Some(Py::new(py, PyTensor { inner: b })?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// Forward pass through the linear layer.
    fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let input_var = input.inner.clone();

        let inner = py.allow_threads(move || {
            let linear = coeus_nn::linear::Linear {
                weight: w_var,
                bias: b_var,
            };
            linear.forward(&input_var)
        });
        Ok(PyTensor { inner })
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.bias {
            sd.insert("bias", b.bind(py).borrow().inner.tensor.clone());
        }
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(ref b) = self.bias {
            if let Some(bias_tensor) = state_dict.inner.get("bias") {
                b.bind(py).borrow_mut().inner.tensor = bias_tensor.clone();
            }
        }
        Ok(())
    }
}
