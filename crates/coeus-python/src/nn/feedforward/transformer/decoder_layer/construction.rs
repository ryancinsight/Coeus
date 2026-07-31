use super::PyTransformerDecoderLayer;
use crate::nn::attention::PyMultiHeadAttention;
use crate::nn::feedforward::PyFeedForward;
use crate::nn::linear::PyLinear;
use crate::nn::normalization::layernorm::PyLayerNorm;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

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
