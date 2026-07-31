use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

/// Python-exposed Linear layer.
#[pyclass(name = "Linear")]
pub struct PyLinear {
    /// Learnable weight matrix, shape `[out_features, in_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Optional learnable bias vector, shape `[out_features]`.
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
        let weight = Py::new(
            py,
            PyTensor {
                inner: linear.weight,
            },
        )?;
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
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let input_var = input.inner.clone();

        let inner = py.allow_threads(move || {
            let linear = coeus_nn::linear::Linear {
                weight: w_var,
                bias: b_var,
            };
            linear.forward(&input_var)
        });
        inner.map(PyTensor::from_var).map_err(map_module_error)
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

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut params = vec![self.weight.clone_ref(py)];
        if let Some(ref b) = self.bias {
            params.push(b.clone_ref(py));
        }
        params
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        if let Some(ref b) = self.bias {
            b.bind(py).borrow().zero_grad();
        }
    }
}
