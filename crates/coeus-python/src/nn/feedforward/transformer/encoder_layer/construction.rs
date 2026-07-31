use super::PyTransformerEncoderLayer;
use crate::nn::attention::PyMultiHeadAttention;
use crate::nn::feedforward::PyFeedForward;
use crate::nn::linear::PyLinear;
use crate::nn::normalization::layernorm::PyLayerNorm;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

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
