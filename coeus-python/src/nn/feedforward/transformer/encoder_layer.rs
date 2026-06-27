use super::super::PyFeedForward;
use crate::nn::attention::PyMultiHeadAttention;
use crate::nn::linear::PyLinear;
use crate::nn::normalization::layernorm::PyLayerNorm;
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── TransformerEncoderLayer ──────────────────────────────────────────────────

/// Python-exposed Transformer Encoder Layer (Pre-LayerNorm).
///
/// Stores all learnable parameters as accessible sub-modules so weights can be
/// read, written, and differentiated from Python.
///
/// ```python
/// enc = pycoeus.TransformerEncoderLayer(d_model=64, d_ff=256, num_heads=4)
/// out = enc.forward(src)         # src: [batch, seq, d_model]
/// enc.self_attn.w_q.data = wq   # set projection weights
/// enc.norm1.weight.data = gamma  # set LayerNorm scale
/// ```
#[pyclass(name = "TransformerEncoderLayer")]
pub struct PyTransformerEncoderLayer {
    /// Pre-LayerNorm before self-attention.
    #[pyo3(get)]
    pub norm1: Py<PyLayerNorm>,
    /// Multi-head self-attention sub-layer.
    #[pyo3(get)]
    pub self_attn: Py<PyMultiHeadAttention>,
    /// Pre-LayerNorm before FFN.
    #[pyo3(get)]
    pub norm2: Py<PyLayerNorm>,
    /// Position-wise feed-forward sub-layer.
    #[pyo3(get)]
    pub ffn: Py<PyFeedForward>,
    /// Model embedding dimensionality.
    #[pyo3(get)]
    pub d_model: usize,
    /// Feed-forward hidden dimensionality.
    #[pyo3(get)]
    pub d_ff: usize,
    /// Number of attention heads.
    #[pyo3(get)]
    pub num_heads: usize,
    /// Dropout probability.
    #[pyo3(get)]
    pub dropout_p: f64,
}

#[pymethods]
impl PyTransformerEncoderLayer {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, dropout_p = 0.0))]
    /// Create a `TransformerEncoderLayer` with Kaiming-initialised weights.
    ///
    /// Sub-modules (`norm1`, `self_attn`, `norm2`, `ffn`) are fully accessible
    /// and mutable so weights can be inspected, set, and differentiated.
    pub fn new(
        py: Python<'_>,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        dropout_p: f64,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "TransformerEncoderLayer: dropout_p must be in [0.0, 1.0)",
            ));
        }
        if num_heads == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(PyValueError::new_err(format!(
                "TransformerEncoderLayer: d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )));
        }
        // Build a fresh Rust encoder layer, then unpack into Python sub-objects via
        // the SSOT helper `build_from_layer` (avoids duplicating extraction logic
        // between `PyTransformerEncoderLayer::new` and `PyTransformerEncoder::new`).
        macro_rules! build {
            ($($h:literal),*) => {
                match num_heads {
                    $($h => {
                        use coeus_nn::transformer::encoder_layer::TransformerEncoderLayer;
                        use coeus_autograd::NullMask;
                        let enc = TransformerEncoderLayer::<
                            f64, coeus_core::MoiraiBackend, $h, NullMask,
                        >::new(d_model, d_ff, dropout_p);
                        Self::build_from_layer::<$h>(py, enc, d_model, d_ff, dropout_p)
                    },)*
                    _ => Err(PyValueError::new_err(format!(
                        "TransformerEncoderLayer: unsupported num_heads={num_heads}; \
                         supported: 1,2,4,8,16,32"
                    ))),
                }
            }
        }
        build!(1, 2, 4, 8, 16, 32)
    }

    /// Pre-LayerNorm self-attention encoder forward.
    ///
    /// - `src`: `[batch, seq, d_model]`
    ///
    /// Returns `[batch, seq, d_model]`.
    pub fn forward(&self, src: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_autograd::NullMask;

        let src_var = src.inner.clone();
        let num_heads = self.num_heads;
        let dropout_p = self.dropout_p;

        // Extract weights from Python sub-objects.
        let n1w = self
            .norm1
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let n1b = self
            .norm1
            .bind(py)
            .borrow()
            .bias
            .bind(py)
            .borrow()
            .inner
            .clone();
        let wq = self
            .self_attn
            .bind(py)
            .borrow()
            .w_q
            .bind(py)
            .borrow()
            .inner
            .clone();
        let bq = self
            .self_attn
            .bind(py)
            .borrow()
            .b_q
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let wk = self
            .self_attn
            .bind(py)
            .borrow()
            .w_k
            .bind(py)
            .borrow()
            .inner
            .clone();
        let bk = self
            .self_attn
            .bind(py)
            .borrow()
            .b_k
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let wv = self
            .self_attn
            .bind(py)
            .borrow()
            .w_v
            .bind(py)
            .borrow()
            .inner
            .clone();
        let bv = self
            .self_attn
            .bind(py)
            .borrow()
            .b_v
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let wo = self
            .self_attn
            .bind(py)
            .borrow()
            .w_o
            .bind(py)
            .borrow()
            .inner
            .clone();
        let bo = self
            .self_attn
            .bind(py)
            .borrow()
            .b_o
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let n2w = self
            .norm2
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let n2b = self
            .norm2
            .bind(py)
            .borrow()
            .bias
            .bind(py)
            .borrow()
            .inner
            .clone();
        let fw1 = self
            .ffn
            .bind(py)
            .borrow()
            .linear1
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let fb1 = self
            .ffn
            .bind(py)
            .borrow()
            .linear1
            .bind(py)
            .borrow()
            .bias
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let fw2 = self
            .ffn
            .bind(py)
            .borrow()
            .linear2
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let fb2 = self
            .ffn
            .bind(py)
            .borrow()
            .linear2
            .bind(py)
            .borrow()
            .bias
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());

        let inner = py.allow_threads(move || -> PyResult<_> {
            macro_rules! dispatch {
                ($($h:literal),*) => {
                    match num_heads {
                        $($h => {
                            Ok(coeus_nn::transformer_encoder_layer::<
                                f64, coeus_core::MoiraiBackend, $h, NullMask,
                            >(
                                &src_var,
                                None,
                                coeus_nn::TransformerEncoderLayerParams {
                                    norm1_weight: &n1w,
                                    norm1_bias: &n1b,
                                    self_attn: coeus_nn::MhaProjectionParams {
                                        w_q: &wq,
                                        b_q: bq.as_ref(),
                                        w_k: &wk,
                                        b_k: bk.as_ref(),
                                        w_v: &wv,
                                        b_v: bv.as_ref(),
                                        w_o: &wo,
                                        b_o: bo.as_ref(),
                                    },
                                    norm2_weight: &n2w,
                                    norm2_bias: &n2b,
                                    ffn_w1: &fw1,
                                    ffn_b1: fb1.as_ref(),
                                    ffn_w2: &fw2,
                                    ffn_b2: fb2.as_ref(),
                                    attn_residual_dropout_p: dropout_p,
                                    attn_residual_training: dropout_p > 0.0,
                                    ffn_hidden_dropout_p: dropout_p,
                                    ffn_hidden_training: dropout_p > 0.0,
                                    ffn_residual_dropout_p: dropout_p,
                                    ffn_residual_training: dropout_p > 0.0,
                                },
                            ))
                        },)*
                        _ => Err(PyValueError::new_err(format!(
                            "TransformerEncoderLayer: unsupported num_heads={num_heads}"
                        ))),
                    }
                }
            }
            dispatch!(1, 2, 4, 8, 16, 32)
        });
        Ok(PyTensor::from_var(inner?))
    }

    /// Return all learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut p = self.norm1.bind(py).borrow().parameters(py);
        p.extend(self.self_attn.bind(py).borrow().parameters(py));
        p.extend(self.norm2.bind(py).borrow().parameters(py));
        p.extend(self.ffn.bind(py).borrow().parameters(py));
        p
    }

    /// Zero gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.norm1.bind(py).borrow().zero_grad(py);
        self.self_attn.bind(py).borrow().zero_grad(py);
        self.norm2.bind(py).borrow().zero_grad(py);
        self.ffn.bind(py).borrow().zero_grad(py);
    }
}

/// Non-`#[pymethods]` constructors shared between `PyTransformerEncoderLayer` and
/// [`PyTransformerEncoder`] to avoid duplicating the field-extraction logic.
impl PyTransformerEncoderLayer {
    /// Extract every sub-component of a Rust `TransformerEncoderLayer` into its Python
    /// counterpart. The const generic `H` pins the head count at compile time so the
    /// function is called once per monomorphization, never per runtime head value.
    pub(crate) fn build_from_layer<const H: usize>(
        py: Python<'_>,
        enc: coeus_nn::transformer::encoder_layer::TransformerEncoderLayer<
            f64,
            coeus_core::MoiraiBackend,
            H,
            coeus_autograd::NullMask,
        >,
        d_model: usize,
        d_ff: usize,
        dropout_p: f64,
    ) -> PyResult<Self> {
        let norm1 = Py::new(
            py,
            PyLayerNorm {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: enc.norm1.weight,
                    },
                )?,
                bias: Py::new(
                    py,
                    PyTensor {
                        inner: enc.norm1.bias,
                    },
                )?,
                eps: 1e-5,
            },
        )?;
        let self_attn = Py::new(
            py,
            PyMultiHeadAttention {
                d_model,
                num_heads: H,
                w_q: Py::new(
                    py,
                    PyTensor {
                        inner: enc.self_attn.w_q,
                    },
                )?,
                b_q: enc
                    .self_attn
                    .b_q
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_k: Py::new(
                    py,
                    PyTensor {
                        inner: enc.self_attn.w_k,
                    },
                )?,
                b_k: enc
                    .self_attn
                    .b_k
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_v: Py::new(
                    py,
                    PyTensor {
                        inner: enc.self_attn.w_v,
                    },
                )?,
                b_v: enc
                    .self_attn
                    .b_v
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_o: Py::new(
                    py,
                    PyTensor {
                        inner: enc.self_attn.w_o,
                    },
                )?,
                b_o: enc
                    .self_attn
                    .b_o
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        let norm2 = Py::new(
            py,
            PyLayerNorm {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: enc.norm2.weight,
                    },
                )?,
                bias: Py::new(
                    py,
                    PyTensor {
                        inner: enc.norm2.bias,
                    },
                )?,
                eps: 1e-5,
            },
        )?;
        let ffn_l1 = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: enc.ffn.linear1.weight,
                    },
                )?,
                bias: enc
                    .ffn
                    .linear1
                    .bias
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        let ffn_l2 = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: enc.ffn.linear2.weight,
                    },
                )?,
                bias: enc
                    .ffn
                    .linear2
                    .bias
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        let ffn = Py::new(
            py,
            PyFeedForward {
                linear1: ffn_l1,
                linear2: ffn_l2,
                dropout_p,
            },
        )?;
        Ok(Self {
            norm1,
            self_attn,
            norm2,
            ffn,
            d_model,
            d_ff,
            num_heads: H,
            dropout_p,
        })
    }

    /// Wrap [`build_from_layer`](Self::build_from_layer) in a `Py<Self>` for storage in
    /// `Vec<Py<PyTransformerEncoderLayer>>` inside [`PyTransformerEncoder`].
    pub(crate) fn from_rust_layer<const H: usize>(
        py: Python<'_>,
        enc: coeus_nn::transformer::encoder_layer::TransformerEncoderLayer<
            f64,
            coeus_core::MoiraiBackend,
            H,
            coeus_autograd::NullMask,
        >,
        d_model: usize,
        d_ff: usize,
        dropout_p: f64,
    ) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self::build_from_layer::<H>(py, enc, d_model, d_ff, dropout_p)?,
        )
    }
}
