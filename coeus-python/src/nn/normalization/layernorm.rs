use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed Layer Normalization layer.
#[pyclass(name = "LayerNorm")]
pub struct PyLayerNorm {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    pub fn new(py: Python<'_>, normalized_shape: usize, eps: f64) -> PyResult<Self> {
        let ln =
            coeus_nn::normalization::layernorm::LayerNorm::<f64, coeus_core::MoiraiBackend>::new(
                normalized_shape,
                eps,
            );
        let weight = Py::new(py, PyTensor { inner: ln.weight })?;
        let bias = Py::new(py, PyTensor { inner: ln.bias })?;
        Ok(Self { weight, bias, eps })
    }

    /// Forward pass through the LayerNorm layer.
    ///
    /// Accepts 2-D input `[N, D]`. For higher-rank inputs (`[batch, seq, D]`, etc.)
    /// call `forward_nd` which handles any rank ≥ 2 via transparent reshape.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let eps_val = self.eps;

        let inner = py.allow_threads(move || {
            let ln =
                coeus_nn::normalization::layernorm::LayerNorm::from_parts(w_var, b_var, eps_val);
            ln.forward(&input_var)
        });
        Ok(PyTensor::from_var(inner))
    }

    /// Forward pass accepting any rank ≥ 2 input.
    ///
    /// Applies LayerNorm over the last dimension regardless of the number of leading
    /// dimensions.  Equivalent to `torch.nn.LayerNorm` called on 3-D Transformer
    /// hidden states `[batch, seq, d_model]` or any other rank-N tensor.
    ///
    /// All reshape operations are tracked, so gradients flow through the entire
    /// flatten → normalize → unflatten chain.
    pub fn forward_nd(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self.bias.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let eps_val = self.eps;

        let inner = py.allow_threads(move || {
            let ln =
                coeus_nn::normalization::layernorm::LayerNorm::from_parts(w_var, b_var, eps_val);
            ln.forward_nd(&input_var)
        });
        Ok(PyTensor::from_var(inner))
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
