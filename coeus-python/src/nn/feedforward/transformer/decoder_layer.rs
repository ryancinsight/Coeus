use super::super::PyFeedForward;
use crate::nn::attention::PyMultiHeadAttention;
use crate::nn::linear::PyLinear;
use crate::nn::normalization::layernorm::PyLayerNorm;
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
        let tgt_var = tgt.inner.clone();
        let mem_var = memory.inner.clone();
        let num_heads = self.num_heads;
        let dropout_p = self.dropout_p;

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
                            use coeus_autograd::{CausalMask, NullMask};
                            Ok(coeus_nn::transformer_decoder_layer::<
                                f64, coeus_core::MoiraiBackend, $h, CausalMask, NullMask,
                            >(
                                &tgt_var,
                                &mem_var,
                                coeus_nn::TransformerDecoderLayerParams {
                                    norm1_weight: &n1w,
                                    norm1_bias: &n1b,
                                    self_attn: coeus_nn::MhaProjectionParams {
                                        w_q: &sa_wq,
                                        b_q: sa_bq.as_ref(),
                                        w_k: &sa_wk,
                                        b_k: sa_bk.as_ref(),
                                        w_v: &sa_wv,
                                        b_v: sa_bv.as_ref(),
                                        w_o: &sa_wo,
                                        b_o: sa_bo.as_ref(),
                                    },
                                    norm2_weight: &n2w,
                                    norm2_bias: &n2b,
                                    cross_attn: coeus_nn::MhaProjectionParams {
                                        w_q: &ca_wq,
                                        b_q: ca_bq.as_ref(),
                                        w_k: &ca_wk,
                                        b_k: ca_bk.as_ref(),
                                        w_v: &ca_wv,
                                        b_v: ca_bv.as_ref(),
                                        w_o: &ca_wo,
                                        b_o: ca_bo.as_ref(),
                                    },
                                    norm3_weight: &n3w,
                                    norm3_bias: &n3b,
                                    ffn_w1: &fw1,
                                    ffn_b1: fb1.as_ref(),
                                    ffn_w2: &fw2,
                                    ffn_b2: fb2.as_ref(),
                                    self_attn_residual_dropout_p: dropout_p,
                                    self_attn_residual_training: dropout_p > 0.0,
                                    cross_attn_residual_dropout_p: dropout_p,
                                    cross_attn_residual_training: dropout_p > 0.0,
                                    ffn_hidden_dropout_p: dropout_p,
                                    ffn_hidden_training: dropout_p > 0.0,
                                    ffn_residual_dropout_p: dropout_p,
                                    ffn_residual_training: dropout_p > 0.0,
                                },
                            ))
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
