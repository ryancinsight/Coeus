// ── PyBilinear — Python wrapper for coeus_nn::Bilinear ──

use crate::{
    init::map_initialization_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

/// Python-exposed Bilinear interaction layer.
///
/// Computes `out[n, k] = x1[n,:] @ W[k,:,:] @ x2[n,:].T + b[k]`.
///
/// # Shapes
/// - `x1`: `[batch, in1_features]`
/// - `x2`: `[batch, in2_features]`
/// - `W`:  `[out_features, in1_features, in2_features]`
/// - Output: `[batch, out_features]`
///
/// # Example
/// ```python
/// bil = pycoeus.Bilinear(in1_features=4, in2_features=3, out_features=2)
/// out = bil.bilinear_forward(x1, x2)
/// ```
#[pyclass(name = "Bilinear")]
pub struct PyBilinear {
    /// Bilinear weight tensor, shape `[out_features, in1_features, in2_features]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Optional bias, shape `[out_features]`.
    #[pyo3(get)]
    pub bias: Option<Py<PyTensor>>,
    /// Number of features in the first input vector.
    pub in1_features: usize,
    /// Number of features in the second input vector.
    pub in2_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

#[pymethods]
impl PyBilinear {
    #[new]
    #[pyo3(signature = (in1_features, in2_features, out_features, bias = true))]
    /// Create a Bilinear layer with given feature dimensions and optional bias.
    pub fn new(
        py: Python<'_>,
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        bias: bool,
    ) -> PyResult<Self> {
        let bil = coeus_nn::bilinear::Bilinear::<f64, coeus_core::MoiraiBackend>::new(
            in1_features,
            in2_features,
            out_features,
            bias,
        )
        .map_err(map_initialization_error)?;
        let weight = Py::new(py, PyTensor { inner: bil.weight })?;
        let b = if let Some(bv) = bil.bias {
            Some(Py::new(py, PyTensor { inner: bv })?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: b,
            in1_features,
            in2_features,
            out_features,
        })
    }

    /// Bilinear forward: `bilinear_forward(x1, x2)`.
    ///
    /// - `x1`: `[batch, in1_features]`
    /// - `x2`: `[batch, in2_features]`
    /// - Returns: `[batch, out_features]`
    pub fn bilinear_forward(
        &self,
        x1: &PyTensor,
        x2: &PyTensor,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let b_var = self
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x1_v = x1.inner.clone();
        let x2_v = x2.inner.clone();

        let inner = py.allow_threads(move || {
            coeus_nn::bilinear::bilinear(&x1_v, &x2_v, &w_var, b_var.as_ref())
        });
        Ok(PyTensor::from_var(inner))
    }

    /// Return a StateDict containing the layer weights.
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.bias {
            sd.insert("bias", b.bind(py).borrow().inner.tensor.clone());
        }
        Ok(PyStateDict { inner: sd })
    }

    /// Load weights from a StateDict into this layer.
    pub fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("bias") {
            if let Some(ref my_b) = self.bias {
                my_b.bind(py).borrow_mut().inner.tensor = b.clone();
            }
        }
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut p = vec![self.weight.clone_ref(py)];
        if let Some(ref b) = self.bias {
            p.push(b.clone_ref(py));
        }
        p
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
        if let Some(ref b) = self.bias {
            b.bind(py).borrow().zero_grad();
        }
    }
}
