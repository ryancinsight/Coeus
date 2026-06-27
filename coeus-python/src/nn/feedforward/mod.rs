// ── FeedForward ───────────────────────────────────────────────────
//
// PyFeedForward is the view-compatible bindings for the ffn module.
// The Transformer stack types (encoder, decoder, seq2seq) live under
// the `transformer/` sub-directory; SinusoidalEncoding lives under
// `positional.rs`.

mod positional;
mod transformer;

use crate::nn::linear::PyLinear;
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-exposed FeedForward (2-layer MLP) transformer sub-block.
///
/// Stores learned parameters as `linear1` (`d_model → d_ff`) and `linear2`
/// (`d_ff → d_model`), both accessible and mutable from Python.
///
/// ```python
/// ffn = pycoeus.FeedForward(d_model=64, d_ff=256)
/// out = ffn.forward(x)   # x: [batch, seq, d_model]
/// ffn.linear1.weight.data = my_weights
/// ```
#[pyclass(name = "FeedForward")]
pub struct PyFeedForward {
    /// First linear projection (`d_model → d_ff`).
    #[pyo3(get)]
    pub linear1: Py<PyLinear>,
    /// Second linear projection (`d_ff → d_model`).
    #[pyo3(get)]
    pub linear2: Py<PyLinear>,
    /// Dropout probability applied between the two projections.
    #[pyo3(get)]
    pub dropout_p: f64,
}

#[pymethods]
impl PyFeedForward {
    #[new]
    #[pyo3(signature = (d_model, d_ff, dropout_p = 0.0))]
    /// Create a FeedForward block with `d_model` → `d_ff` → `d_model` projections.
    pub fn new(py: Python<'_>, d_model: usize, d_ff: usize, dropout_p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "FeedForward: dropout_p must be in [0.0, 1.0)",
            ));
        }
        let ffn_init =
            coeus_nn::transformer::ffn::FeedForward::<f64, coeus_core::MoiraiBackend>::new(
                d_model, d_ff, dropout_p,
            );
        let linear1 = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: ffn_init.linear1.weight,
                    },
                )?,
                bias: ffn_init
                    .linear1
                    .bias
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        let linear2 = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: ffn_init.linear2.weight,
                    },
                )?,
                bias: ffn_init
                    .linear2
                    .bias
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        Ok(Self {
            linear1,
            linear2,
            dropout_p,
        })
    }

    /// Forward pass: `Linear1 → GELU → Dropout → Linear2`.
    ///
    /// Accepts any rank ≥ 2 input; the standard transformer shape is
    /// `[batch, seq, d_model]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let w1 = self
            .linear1
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let b1 = self
            .linear1
            .bind(py)
            .borrow()
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let w2 = self
            .linear2
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let b2 = self
            .linear2
            .bind(py)
            .borrow()
            .bias
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let dropout_p = self.dropout_p;
        let x = input.inner.clone();
        let inner = py.allow_threads(move || {
            coeus_nn::feed_forward(&x, &w1, b1.as_ref(), &w2, b2.as_ref(), dropout_p)
        });
        Ok(PyTensor::from_var(inner))
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut p = self.linear1.bind(py).borrow().parameters(py);
        p.extend(self.linear2.bind(py).borrow().parameters(py));
        p
    }

    /// Zero gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.linear1.bind(py).borrow().zero_grad(py);
        self.linear2.bind(py).borrow().zero_grad(py);
    }
}

pub use positional::PySinusoidalEncoding;
pub use transformer::decoder::PyTransformerDecoder;
pub use transformer::decoder_layer::PyTransformerDecoderLayer;
pub use transformer::encoder::PyTransformerEncoder;
pub use transformer::encoder_layer::PyTransformerEncoderLayer;
pub use transformer::seq2seq::PyTransformer;
