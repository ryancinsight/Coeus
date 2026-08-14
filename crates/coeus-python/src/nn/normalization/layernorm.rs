use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

pub(crate) fn parse_normalized_shape(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let shape = if let Ok(dimension) = value.extract::<usize>() {
        vec![dimension]
    } else {
        value.extract::<Vec<usize>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "LayerNorm: normalized_shape must be int or sequence of ints",
            )
        })?
    };
    if shape.is_empty() || shape.contains(&0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LayerNorm: normalized_shape must contain positive dimensions",
        ));
    }
    Ok(shape)
}

/// Python-exposed Layer Normalization layer.
#[pyclass(name = "LayerNorm")]
pub struct PyLayerNorm {
    /// Learnable scale with shape `normalized_shape`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Learnable shift with shape `normalized_shape`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    /// Create a LayerNorm layer over one or more trailing dimensions.
    ///
    /// Mirrors `torch.nn.LayerNorm`: `normalized_shape` accepts an integer or
    /// a non-empty sequence of positive integers.
    #[pyo3(signature = (normalized_shape, eps=None))]
    pub fn new(
        py: Python<'_>,
        normalized_shape: &Bound<'_, PyAny>,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        let normalized_shape = parse_normalized_shape(normalized_shape)?;
        let eps = eps.unwrap_or(1e-5);
        let layer = coeus_nn::normalization::layernorm::LayerNorm::<
            f64,
            coeus_core::MoiraiBackend,
        >::from_shape(normalized_shape, eps);
        let weight = Py::new(
            py,
            PyTensor {
                inner: layer.weight,
            },
        )?;
        let bias = Py::new(py, PyTensor { inner: layer.bias })?;
        Ok(Self { weight, bias, eps })
    }

    /// Forward pass through LayerNorm over the configured trailing dimensions.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        self.forward_nd(input, py)
    }

    /// Forward pass accepting any rank ≥ 2 input.
    ///
    /// The configured suffix is normalized as one feature domain and the
    /// original input shape is preserved. All reshape operations are tracked,
    /// so gradients flow through the flatten → normalize → unflatten chain.
    pub fn forward_nd(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let eps_val = self.eps;

        let inner = py.allow_threads(move || {
            let layer =
                coeus_nn::normalization::layernorm::LayerNorm::from_parts(w_var, b_var, eps_val);
            layer.forward_nd(&input_var)
        });
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        sd.insert("bias", self.bias.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            self.bias.bind(py).borrow_mut().inner.tensor = b.clone();
        }
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py), self.bias.clone_ref(py)]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        self.bias.bind(py).borrow().zero_grad();
    }
}
