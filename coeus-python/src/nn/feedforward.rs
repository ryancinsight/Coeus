use crate::nn::attention::PyMultiHeadAttention;
use crate::nn::linear::PyLinear;
use crate::nn::normalization::layernorm::PyLayerNorm;
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
        let ffn_init2 =
            coeus_nn::transformer::ffn::FeedForward::<f64, coeus_core::MoiraiBackend>::new(
                d_model, d_ff, dropout_p,
            );
        let linear2 = Py::new(
            py,
            PyLinear {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: ffn_init2.linear2.weight,
                    },
                )?,
                bias: ffn_init2
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
        use coeus_nn::transformer::ffn::FeedForward;
        use coeus_nn::Module;
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
            let mut ffn = FeedForward::<f64, coeus_core::MoiraiBackend>::new(1, 1, dropout_p);
            ffn.linear1.weight = w1;
            ffn.linear1.bias = b1;
            ffn.linear2.weight = w2;
            ffn.linear2.bias = b2;
            ffn.forward(&x)
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

/// Python-exposed Transformer Decoder Layer (Pre-LayerNorm).
///
/// Stores all learnable parameters as accessible sub-modules so weights can be
/// read, written, and differentiated from Python.
///
/// ```python
/// dec = pycoeus.TransformerDecoderLayer(d_model=64, d_ff=256, num_heads=4)
/// out = dec.forward(tgt, memory)          # tgt, memory: [batch, seq, d_model]
/// dec.self_attn.w_q.data = wq            # set self-attention projection
/// dec.cross_attn.w_q.data = wq_cross     # set cross-attention projection
/// dec.norm1.weight.data = gamma          # set LayerNorm scale
/// len(dec.parameters())                  # 26 (with biases)
/// ```
#[pyclass(name = "TransformerDecoderLayer")]
pub struct PyTransformerDecoderLayer {
    /// Pre-LayerNorm before masked self-attention.
    #[pyo3(get)]
    pub norm1: Py<PyLayerNorm>,
    /// Multi-head masked self-attention sub-layer (causal mask).
    #[pyo3(get)]
    pub self_attn: Py<PyMultiHeadAttention>,
    /// Pre-LayerNorm before cross-attention.
    #[pyo3(get)]
    pub norm2: Py<PyLayerNorm>,
    /// Multi-head cross-attention sub-layer (encoder memory as keys/values).
    #[pyo3(get)]
    pub cross_attn: Py<PyMultiHeadAttention>,
    /// Pre-LayerNorm before FFN.
    #[pyo3(get)]
    pub norm3: Py<PyLayerNorm>,
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
impl PyTransformerDecoderLayer {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, dropout_p = 0.0))]
    /// Create a `TransformerDecoderLayer` with Kaiming-initialised weights.
    ///
    /// Sub-modules (`norm1`, `self_attn`, `norm2`, `cross_attn`, `norm3`, `ffn`)
    /// are fully accessible and mutable so weights can be inspected, set, and
    /// differentiated.
    pub fn new(
        py: Python<'_>,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        dropout_p: f64,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "TransformerDecoderLayer: dropout_p must be in [0.0, 1.0)",
            ));
        }
        if num_heads == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(PyValueError::new_err(format!(
                "TransformerDecoderLayer: d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )));
        }
        macro_rules! build {
            ($($h:literal),*) => {
                match num_heads {
                    $($h => {
                        use coeus_nn::transformer::decoder_layer::TransformerDecoderLayer;
                        use coeus_autograd::{CausalMask, NullMask};
                        let dec = TransformerDecoderLayer::<
                            f64, coeus_core::MoiraiBackend, $h, CausalMask, NullMask,
                        >::new(d_model, d_ff, dropout_p);
                        Self::build_from_layer::<$h>(py, dec, d_model, d_ff, dropout_p)
                    },)*
                    _ => Err(PyValueError::new_err(format!(
                        "TransformerDecoderLayer: unsupported num_heads={num_heads}; \
                         supported: 1,2,4,8,16,32"
                    ))),
                }
            }
        }
        build!(1, 2, 4, 8, 16, 32)
    }

    /// Pre-LN cross-attention decoder forward.
    ///
    /// - `tgt`:    `[batch, seq_tgt, d_model]`
    /// - `memory`: `[batch, seq_src, d_model]`
    ///
    /// Returns `[batch, seq_tgt, d_model]`.
    #[pyo3(signature = (tgt, memory))]
    pub fn forward(&self, tgt: &PyTensor, memory: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::transformer::decoder_layer::TransformerDecoderLayer;

        let tgt_var = tgt.inner.clone();
        let mem_var = memory.inner.clone();
        let num_heads = self.num_heads;
        let dropout_p = self.dropout_p;
        let d_model = self.d_model;
        let d_ff = self.d_ff;

        // Extract norm1
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
        // Extract self_attn
        let sa_wq = self
            .self_attn
            .bind(py)
            .borrow()
            .w_q
            .bind(py)
            .borrow()
            .inner
            .clone();
        let sa_bq = self
            .self_attn
            .bind(py)
            .borrow()
            .b_q
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let sa_wk = self
            .self_attn
            .bind(py)
            .borrow()
            .w_k
            .bind(py)
            .borrow()
            .inner
            .clone();
        let sa_bk = self
            .self_attn
            .bind(py)
            .borrow()
            .b_k
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let sa_wv = self
            .self_attn
            .bind(py)
            .borrow()
            .w_v
            .bind(py)
            .borrow()
            .inner
            .clone();
        let sa_bv = self
            .self_attn
            .bind(py)
            .borrow()
            .b_v
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let sa_wo = self
            .self_attn
            .bind(py)
            .borrow()
            .w_o
            .bind(py)
            .borrow()
            .inner
            .clone();
        let sa_bo = self
            .self_attn
            .bind(py)
            .borrow()
            .b_o
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        // Extract norm2
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
        // Extract cross_attn
        let ca_wq = self
            .cross_attn
            .bind(py)
            .borrow()
            .w_q
            .bind(py)
            .borrow()
            .inner
            .clone();
        let ca_bq = self
            .cross_attn
            .bind(py)
            .borrow()
            .b_q
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let ca_wk = self
            .cross_attn
            .bind(py)
            .borrow()
            .w_k
            .bind(py)
            .borrow()
            .inner
            .clone();
        let ca_bk = self
            .cross_attn
            .bind(py)
            .borrow()
            .b_k
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let ca_wv = self
            .cross_attn
            .bind(py)
            .borrow()
            .w_v
            .bind(py)
            .borrow()
            .inner
            .clone();
        let ca_bv = self
            .cross_attn
            .bind(py)
            .borrow()
            .b_v
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        let ca_wo = self
            .cross_attn
            .bind(py)
            .borrow()
            .w_o
            .bind(py)
            .borrow()
            .inner
            .clone();
        let ca_bo = self
            .cross_attn
            .bind(py)
            .borrow()
            .b_o
            .as_ref()
            .map(|v| v.bind(py).borrow().inner.clone());
        // Extract norm3
        let n3w = self
            .norm3
            .bind(py)
            .borrow()
            .weight
            .bind(py)
            .borrow()
            .inner
            .clone();
        let n3b = self
            .norm3
            .bind(py)
            .borrow()
            .bias
            .bind(py)
            .borrow()
            .inner
            .clone();
        // Extract FFN
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
                            use coeus_nn::normalization::layernorm::LayerNorm;
                            use coeus_autograd::{CausalMask, NullMask};
                            let mut dec = TransformerDecoderLayer::<
                                f64, coeus_core::MoiraiBackend, $h, CausalMask, NullMask,
                            >::new(d_model, d_ff, dropout_p);
                            dec.norm1 = LayerNorm::from_parts(n1w, n1b, 1e-5);
                            dec.self_attn.w_q = sa_wq;
                            dec.self_attn.b_q = sa_bq;
                            dec.self_attn.w_k = sa_wk;
                            dec.self_attn.b_k = sa_bk;
                            dec.self_attn.w_v = sa_wv;
                            dec.self_attn.b_v = sa_bv;
                            dec.self_attn.w_o = sa_wo;
                            dec.self_attn.b_o = sa_bo;
                            dec.norm2 = LayerNorm::from_parts(n2w, n2b, 1e-5);
                            dec.cross_attn.w_q = ca_wq;
                            dec.cross_attn.b_q = ca_bq;
                            dec.cross_attn.w_k = ca_wk;
                            dec.cross_attn.b_k = ca_bk;
                            dec.cross_attn.w_v = ca_wv;
                            dec.cross_attn.b_v = ca_bv;
                            dec.cross_attn.w_o = ca_wo;
                            dec.cross_attn.b_o = ca_bo;
                            dec.norm3 = LayerNorm::from_parts(n3w, n3b, 1e-5);
                            dec.ffn.linear1.weight = fw1;
                            dec.ffn.linear1.bias = fb1;
                            dec.ffn.linear2.weight = fw2;
                            dec.ffn.linear2.bias = fb2;
                            Ok(dec.forward_decoder(&tgt_var, &mem_var))
                        },)*
                        _ => Err(PyValueError::new_err(format!(
                            "TransformerDecoderLayer: unsupported num_heads={num_heads}"
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
        p.extend(self.cross_attn.bind(py).borrow().parameters(py));
        p.extend(self.norm3.bind(py).borrow().parameters(py));
        p.extend(self.ffn.bind(py).borrow().parameters(py));
        p
    }

    /// Zero gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.norm1.bind(py).borrow().zero_grad(py);
        self.self_attn.bind(py).borrow().zero_grad(py);
        self.norm2.bind(py).borrow().zero_grad(py);
        self.cross_attn.bind(py).borrow().zero_grad(py);
        self.norm3.bind(py).borrow().zero_grad(py);
        self.ffn.bind(py).borrow().zero_grad(py);
    }
}

/// Non-`#[pymethods]` constructors shared between `PyTransformerDecoderLayer` and
/// [`PyTransformerDecoder`] to avoid duplicating the field-extraction logic.
impl PyTransformerDecoderLayer {
    /// Extract every sub-component of a Rust `TransformerDecoderLayer` into its Python
    /// counterpart. The const generic `H` pins the head count at compile time.
    pub(crate) fn build_from_layer<const H: usize>(
        py: Python<'_>,
        dec: coeus_nn::transformer::decoder_layer::TransformerDecoderLayer<
            f64,
            coeus_core::MoiraiBackend,
            H,
            coeus_autograd::CausalMask,
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
                        inner: dec.norm1.weight,
                    },
                )?,
                bias: Py::new(
                    py,
                    PyTensor {
                        inner: dec.norm1.bias,
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
                        inner: dec.self_attn.w_q,
                    },
                )?,
                b_q: dec
                    .self_attn
                    .b_q
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_k: Py::new(
                    py,
                    PyTensor {
                        inner: dec.self_attn.w_k,
                    },
                )?,
                b_k: dec
                    .self_attn
                    .b_k
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_v: Py::new(
                    py,
                    PyTensor {
                        inner: dec.self_attn.w_v,
                    },
                )?,
                b_v: dec
                    .self_attn
                    .b_v
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_o: Py::new(
                    py,
                    PyTensor {
                        inner: dec.self_attn.w_o,
                    },
                )?,
                b_o: dec
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
                        inner: dec.norm2.weight,
                    },
                )?,
                bias: Py::new(
                    py,
                    PyTensor {
                        inner: dec.norm2.bias,
                    },
                )?,
                eps: 1e-5,
            },
        )?;
        let cross_attn = Py::new(
            py,
            PyMultiHeadAttention {
                d_model,
                num_heads: H,
                w_q: Py::new(
                    py,
                    PyTensor {
                        inner: dec.cross_attn.w_q,
                    },
                )?,
                b_q: dec
                    .cross_attn
                    .b_q
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_k: Py::new(
                    py,
                    PyTensor {
                        inner: dec.cross_attn.w_k,
                    },
                )?,
                b_k: dec
                    .cross_attn
                    .b_k
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_v: Py::new(
                    py,
                    PyTensor {
                        inner: dec.cross_attn.w_v,
                    },
                )?,
                b_v: dec
                    .cross_attn
                    .b_v
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
                w_o: Py::new(
                    py,
                    PyTensor {
                        inner: dec.cross_attn.w_o,
                    },
                )?,
                b_o: dec
                    .cross_attn
                    .b_o
                    .map(|v| Py::new(py, PyTensor { inner: v }))
                    .transpose()?,
            },
        )?;
        let norm3 = Py::new(
            py,
            PyLayerNorm {
                weight: Py::new(
                    py,
                    PyTensor {
                        inner: dec.norm3.weight,
                    },
                )?,
                bias: Py::new(
                    py,
                    PyTensor {
                        inner: dec.norm3.bias,
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
                        inner: dec.ffn.linear1.weight,
                    },
                )?,
                bias: dec
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
                        inner: dec.ffn.linear2.weight,
                    },
                )?,
                bias: dec
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
            cross_attn,
            norm3,
            ffn,
            d_model,
            d_ff,
            num_heads: H,
            dropout_p,
        })
    }

    /// Wrap [`build_from_layer`](Self::build_from_layer) in a `Py<Self>` for storage in
    /// `Vec<Py<PyTransformerDecoderLayer>>` inside [`PyTransformerDecoder`].
    pub(crate) fn from_rust_layer<const H: usize>(
        py: Python<'_>,
        dec: coeus_nn::transformer::decoder_layer::TransformerDecoderLayer<
            f64,
            coeus_core::MoiraiBackend,
            H,
            coeus_autograd::CausalMask,
            coeus_autograd::NullMask,
        >,
        d_model: usize,
        d_ff: usize,
        dropout_p: f64,
    ) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self::build_from_layer::<H>(py, dec, d_model, d_ff, dropout_p)?,
        )
    }
}

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
        use coeus_nn::transformer::encoder_layer::TransformerEncoderLayer;
        use coeus_nn::Module as _;

        let src_var = src.inner.clone();
        let num_heads = self.num_heads;
        let dropout_p = self.dropout_p;
        let d_model = self.d_model;
        let d_ff = self.d_ff;

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
                            use coeus_nn::normalization::layernorm::LayerNorm;
                            let mut enc = TransformerEncoderLayer::<
                                f64, coeus_core::MoiraiBackend, $h, NullMask,
                            >::new(d_model, d_ff, dropout_p);
                            enc.norm1 = LayerNorm::from_parts(n1w, n1b, 1e-5);
                            enc.self_attn.w_q = wq;
                            enc.self_attn.b_q = bq;
                            enc.self_attn.w_k = wk;
                            enc.self_attn.b_k = bk;
                            enc.self_attn.w_v = wv;
                            enc.self_attn.b_v = bv;
                            enc.self_attn.w_o = wo;
                            enc.self_attn.b_o = bo;
                            enc.norm2 = LayerNorm::from_parts(n2w, n2b, 1e-5);
                            enc.ffn.linear1.weight = fw1;
                            enc.ffn.linear1.bias = fb1;
                            enc.ffn.linear2.weight = fw2;
                            enc.ffn.linear2.bias = fb2;
                            Ok(enc.forward(&src_var))
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

// ── TransformerEncoder ───────────────────────────────────────────────────────

/// Python-exposed Transformer Encoder stack (Pre-LayerNorm, N layers).
///
/// Each layer is stored as a fully-stateful [`PyTransformerEncoderLayer`] so weights
/// can be read, written, and differentiated from Python at per-layer resolution.
///
/// ```python
/// enc = pycoeus.TransformerEncoder(d_model=64, d_ff=256, num_heads=4, num_layers=2)
/// out = enc.forward(src)            # src: [batch, seq, d_model]
/// enc.layers[0].norm1.weight.data   # per-layer weight access
/// len(enc.parameters())             # 16 * num_layers
/// ```
#[pyclass(name = "TransformerEncoder")]
pub struct PyTransformerEncoder {
    /// Stack of independently-initialised encoder layers.
    #[pyo3(get)]
    pub layers: Vec<Py<PyTransformerEncoderLayer>>,
    /// Model embedding dimensionality.
    #[pyo3(get)]
    pub d_model: usize,
    /// Feed-forward hidden dimensionality.
    #[pyo3(get)]
    pub d_ff: usize,
    /// Number of attention heads per layer.
    #[pyo3(get)]
    pub num_heads: usize,
    /// Dropout probability applied within each layer.
    #[pyo3(get)]
    pub dropout_p: f64,
}

#[pymethods]
impl PyTransformerEncoder {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, num_layers = 6, dropout_p = 0.0))]
    /// Create a `TransformerEncoder` with `num_layers` independently-initialised layers.
    ///
    /// Each layer is stored as a [`PyTransformerEncoderLayer`] with full sub-module access.
    /// Supported `num_heads` values: 1, 2, 4, 8, 16, 32.
    /// Supported `num_layers` values: 1, 2, 4, 6, 8, 12.
    pub fn new(
        py: Python<'_>,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_p: f64,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "TransformerEncoder: dropout_p must be in [0.0, 1.0)",
            ));
        }
        if num_heads == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(PyValueError::new_err(format!(
                "TransformerEncoder: d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )));
        }
        macro_rules! build {
            ($(($h:literal, $n:literal)),*) => {
                match (num_heads, num_layers) {
                    $(($h, $n) => {
                        use coeus_nn::transformer::encoder::TransformerEncoder;
                        use coeus_autograd::NullMask;
                        let enc = TransformerEncoder::<
                            f64, coeus_core::MoiraiBackend, $h, $n, NullMask,
                        >::new(d_model, d_ff, dropout_p);
                        enc.layers
                            .into_iter()
                            .map(|layer| PyTransformerEncoderLayer::from_rust_layer::<$h>(
                                py, layer, d_model, d_ff, dropout_p,
                            ))
                            .collect::<PyResult<Vec<_>>>()?
                    },)*
                    _ => return Err(PyValueError::new_err(format!(
                        "TransformerEncoder: unsupported (num_heads={num_heads}, \
                         num_layers={num_layers}); supported heads: 1,2,4,8,16,32 \
                         and layers: 1,2,4,6,8,12"
                    ))),
                }
            }
        }
        let layers = build!(
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 6),
            (1, 8),
            (1, 12),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 6),
            (2, 8),
            (2, 12),
            (4, 1),
            (4, 2),
            (4, 4),
            (4, 6),
            (4, 8),
            (4, 12),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (8, 12),
            (16, 1),
            (16, 2),
            (16, 4),
            (16, 6),
            (16, 8),
            (16, 12),
            (32, 1),
            (32, 2),
            (32, 4),
            (32, 6),
            (32, 8),
            (32, 12)
        );
        Ok(Self {
            layers,
            d_model,
            d_ff,
            num_heads,
            dropout_p,
        })
    }

    /// Number of stacked encoder layers (convenience alias for `len(enc.layers)`).
    #[getter]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Chain each layer's Pre-LN forward sequentially.
    ///
    /// - `src`: `[batch, seq, d_model]`
    ///
    /// Returns `[batch, seq, d_model]`.
    pub fn forward(&self, src: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let mut current = PyTensor {
            inner: src.inner.clone(),
        };
        for layer_py in &self.layers {
            let layer = layer_py.bind(py).borrow();
            current = layer.forward(&current, py)?;
        }
        Ok(current)
    }

    /// Return all learnable parameters across every layer.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        self.layers
            .iter()
            .flat_map(|l| l.bind(py).borrow().parameters(py))
            .collect()
    }

    /// Zero gradients of all parameters across every layer.
    pub fn zero_grad(&self, py: Python<'_>) {
        for l in &self.layers {
            l.bind(py).borrow().zero_grad(py);
        }
    }
}

// ── TransformerDecoder ──────────────────────────────────────────────────────

/// Python-exposed Transformer Decoder stack (Pre-LayerNorm, N layers).
///
/// Each layer is stored as a fully-stateful [`PyTransformerDecoderLayer`] so weights
/// can be read, written, and differentiated from Python at per-layer resolution.
///
/// ```python
/// dec = pycoeus.TransformerDecoder(d_model=64, d_ff=256, num_heads=4, num_layers=2)
/// out = dec.forward(tgt, memory)    # tgt, memory: [batch, seq, d_model]
/// dec.layers[0].norm1.weight.data   # per-layer weight access
/// len(dec.parameters())             # 26 * num_layers
/// ```
#[pyclass(name = "TransformerDecoder")]
pub struct PyTransformerDecoder {
    /// Stack of independently-initialised decoder layers.
    #[pyo3(get)]
    pub layers: Vec<Py<PyTransformerDecoderLayer>>,
    /// Model embedding dimensionality.
    #[pyo3(get)]
    pub d_model: usize,
    /// Feed-forward hidden dimensionality.
    #[pyo3(get)]
    pub d_ff: usize,
    /// Number of attention heads per layer.
    #[pyo3(get)]
    pub num_heads: usize,
    /// Dropout probability applied within each layer.
    #[pyo3(get)]
    pub dropout_p: f64,
}

#[pymethods]
impl PyTransformerDecoder {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, num_layers = 6, dropout_p = 0.0))]
    /// Create a `TransformerDecoder` with `num_layers` independently-initialised layers.
    ///
    /// Each layer is stored as a [`PyTransformerDecoderLayer`] with full sub-module access.
    /// Supported `num_heads` values: 1, 2, 4, 8, 16, 32.
    /// Supported `num_layers` values: 1, 2, 4, 6, 8, 12.
    pub fn new(
        py: Python<'_>,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_p: f64,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "TransformerDecoder: dropout_p must be in [0.0, 1.0)",
            ));
        }
        if num_heads == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(PyValueError::new_err(format!(
                "TransformerDecoder: d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )));
        }
        macro_rules! build {
            ($(($h:literal, $n:literal)),*) => {
                match (num_heads, num_layers) {
                    $(($h, $n) => {
                        use coeus_nn::transformer::decoder::TransformerDecoder;
                        use coeus_autograd::{CausalMask, NullMask};
                        let dec = TransformerDecoder::<
                            f64, coeus_core::MoiraiBackend, $h, $n, CausalMask, NullMask,
                        >::new(d_model, d_ff, dropout_p);
                        dec.layers
                            .into_iter()
                            .map(|layer| PyTransformerDecoderLayer::from_rust_layer::<$h>(
                                py, layer, d_model, d_ff, dropout_p,
                            ))
                            .collect::<PyResult<Vec<_>>>()?
                    },)*
                    _ => return Err(PyValueError::new_err(format!(
                        "TransformerDecoder: unsupported (num_heads={num_heads}, \
                         num_layers={num_layers}); supported heads: 1,2,4,8,16,32 \
                         and layers: 1,2,4,6,8,12"
                    ))),
                }
            }
        }
        let layers = build!(
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 6),
            (1, 8),
            (1, 12),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 6),
            (2, 8),
            (2, 12),
            (4, 1),
            (4, 2),
            (4, 4),
            (4, 6),
            (4, 8),
            (4, 12),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (8, 12),
            (16, 1),
            (16, 2),
            (16, 4),
            (16, 6),
            (16, 8),
            (16, 12),
            (32, 1),
            (32, 2),
            (32, 4),
            (32, 6),
            (32, 8),
            (32, 12)
        );
        Ok(Self {
            layers,
            d_model,
            d_ff,
            num_heads,
            dropout_p,
        })
    }

    /// Number of stacked decoder layers (convenience alias for `len(dec.layers)`).
    #[getter]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Chain each layer's Pre-LN cross-attention forward sequentially.
    ///
    /// - `tgt`:    `[batch, seq_tgt, d_model]`
    /// - `memory`: `[batch, seq_src, d_model]`
    ///
    /// Returns `[batch, seq_tgt, d_model]`.
    pub fn forward(&self, tgt: &PyTensor, memory: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let mut current = PyTensor {
            inner: tgt.inner.clone(),
        };
        for layer_py in &self.layers {
            let layer = layer_py.bind(py).borrow();
            current = layer.forward(&current, memory, py)?;
        }
        Ok(current)
    }

    /// Return all learnable parameters across every layer.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        self.layers
            .iter()
            .flat_map(|l| l.bind(py).borrow().parameters(py))
            .collect()
    }

    /// Zero gradients of all parameters across every layer.
    pub fn zero_grad(&self, py: Python<'_>) {
        for l in &self.layers {
            l.bind(py).borrow().zero_grad(py);
        }
    }
}

// ── SinusoidalEncoding ───────────────────────────────────────────────────────

/// Python-exposed sinusoidal positional encoding.
///
/// ```python
/// pe = pycoeus.SinusoidalEncoding(max_len=512, d_model=64)
/// out = pe.forward(embeddings)   # [batch, seq, d_model]
/// ```
#[pyclass(name = "SinusoidalEncoding")]
pub struct PySinusoidalEncoding {
    /// Maximum sequence length for which encodings are pre-computed.
    #[pyo3(get)]
    pub max_len: usize,
    /// Model embedding dimensionality (must be positive and even).
    #[pyo3(get)]
    pub d_model: usize,
}

#[pymethods]
impl PySinusoidalEncoding {
    #[new]
    /// Create a SinusoidalEncoding table of shape `[max_len, d_model]`.
    pub fn new(max_len: usize, d_model: usize) -> PyResult<Self> {
        if d_model == 0 || !d_model.is_multiple_of(2) {
            return Err(PyValueError::new_err(
                "SinusoidalEncoding: d_model must be a positive even integer",
            ));
        }
        Ok(Self { max_len, d_model })
    }

    /// Add sinusoidal positional encoding to `input`.
    ///
    /// - `input`: `[batch, seq_len, d_model]`
    ///
    /// Returns `[batch, seq_len, d_model]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let x = input.inner.clone();
        let (max_len, d_model) = (self.max_len, self.d_model);
        let inner = py.allow_threads(move || {
            use coeus_nn::positional::sinusoidal::SinusoidalEncoding;
            use coeus_nn::Module;
            let pe = SinusoidalEncoding::<f64, coeus_core::MoiraiBackend>::new(max_len, d_model);
            pe.forward(&x)
        });
        Ok(PyTensor::from_var(inner))
    }
}
