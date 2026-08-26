// ── SwiGLU ────────────────────────────────────────────────────────
//
// PySwiGlu binds the SwiGLU gated feed-forward unit (coeus_nn::SwiGlu).
// It holds two PyLinear sub-modules — the inner SiLU-gated projection and the
// outer value projection — exposed via `#[pyo3(get)]` so Python can inject
// weights (e.g. for PyTorch parity testing).

use crate::nn::linear::PyLinear;
use crate::{nn::error::map_module_error, tensor::PyTensor};
use coeus_nn::Module;
use pyo3::prelude::*;

/// Python-exposed SwiGLU gated linear unit.
///
/// `SwiGLU(x) = silu(linear_inner(x)) ⊙ linear_outer(x)`. Both projections
/// (`d_input → d_output`) are accessible and mutable from Python.
///
/// ```python
/// sg = pycoeus.SwiGlu(d_input=256, d_output=512, bias=False)
/// out = sg.forward(x)                  # x: [..., d_input]
/// sg.linear_inner.weight.data = wi     # inject weights for parity
/// ```
#[pyclass(name = "SwiGlu")]
pub struct PySwiGlu {
    /// Inner (SiLU-gated) projection, `d_input → d_output`.
    #[pyo3(get)]
    pub linear_inner: Py<PyLinear>,
    /// Outer (value) projection, `d_input → d_output`.
    #[pyo3(get)]
    pub linear_outer: Py<PyLinear>,
}

#[pymethods]
impl PySwiGlu {
    #[new]
    #[pyo3(signature = (d_input, d_output, bias = false))]
    /// Create a SwiGLU unit projecting `d_input → d_output`, with optional bias
    /// on both linear layers (default `false`, matching Burn).
    pub fn new(py: Python<'_>, d_input: usize, d_output: usize, bias: bool) -> PyResult<Self> {
        let init = coeus_nn::SwiGlu::<f64, coeus_core::MoiraiBackend>::new(d_input, d_output, bias)
            .map_err(crate::init::map_initialization_error)?;
        let linear_inner = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: init.linear_inner.weight,
                    },
                )?,
                bias: init
                    .linear_inner
                    .bias
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        let linear_outer = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: init.linear_outer.weight,
                    },
                )?,
                bias: init
                    .linear_outer
                    .bias
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        Ok(Self {
            linear_inner,
            linear_outer,
        })
    }

    /// Forward pass: `silu(linear_inner(x)) ⊙ linear_outer(x)`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let w_inner = self
            .linear_inner
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let b_inner = self
            .linear_inner
            .bind(py)
            .borrow()
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let w_outer = self
            .linear_outer
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let b_outer = self
            .linear_outer
            .bind(py)
            .borrow()
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x = input.inner.clone();

        let out = py.allow_threads(move || {
            let swiglu = coeus_nn::SwiGlu {
                linear_inner: coeus_nn::linear::Linear {
                    weight: w_inner,
                    bias: b_inner,
                },
                linear_outer: coeus_nn::linear::Linear {
                    weight: w_outer,
                    bias: b_outer,
                },
            };
            swiglu.forward(&x)
        });
        out.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Learnable parameters: both projections' weights (and biases if present).
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut params = self.linear_inner.bind(py).borrow().parameters(py);
        params.extend(self.linear_outer.bind(py).borrow().parameters(py));
        params
    }

    /// Zero gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.linear_inner.bind(py).borrow().zero_grad(py);
        self.linear_outer.bind(py).borrow().zero_grad(py);
    }
}
