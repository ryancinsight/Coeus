use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed Layer Normalization layer.
#[pyclass(name = "LayerNorm")]
pub struct PyLayerNorm {
    /// Learnable scale (gamma), shape `[normalized_shape]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Learnable shift (beta), shape `[normalized_shape]`.
    #[pyo3(get)]
    pub bias: Py<PyTensor>,
    /// Numerical stability epsilon added to the denominator.
    #[pyo3(get)]
    pub eps: f64,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    /// Create a LayerNorm layer normalizing over `normalized_shape` dimensions.
    ///
    /// Mirrors `torch.nn.LayerNorm` constructor argument conventions: accepts a
    /// single `int` (like `nn.LayerNorm(8)`) or a length-1 sequence
    /// (`nn.LayerNorm([8])` / `nn.LayerNorm((8,))`).  Sequences of length > 1
    /// currently reduce to the product of their elements minus the trailing
    /// dims — i.e. only single-dim normalization is supported by the Rust core,
    /// matching the existing `LayerNorm::new(usize, f64)` contract.  Multi-dim
    /// LayerNorm is a deferred surface.
    pub fn new(
        py: Python<'_>,
        normalized_shape: &Bound<'_, PyAny>,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        // Prefer int — the cheaper path and the canonical Coeus form.
        let shape_int: usize = match normalized_shape.extract() {
            Ok(v) => v,
            Err(_) => {
                // Fall back to a sequence (list/tuple) of ints.  Length-1 reduces
                // to the inner shape; longer sequences are not yet supported by
                // the Rust core LayerNorm.
                let seq: Vec<usize> = normalized_shape.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "LayerNorm: normalized_shape must be int or sequence of ints",
                    )
                })?;
                if seq.len() != 1 {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                        "LayerNorm: multi-dim normalized_shape {seq:?} not supported \
                         (Coeus LayerNorm::new takes a single usize)"
                    )));
                }
                seq[0]
            }
        };
        let ln =
            coeus_nn::normalization::layernorm::LayerNorm::<f64, coeus_core::MoiraiBackend>::new(
                shape_int, eps,
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
