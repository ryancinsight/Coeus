use crate::nn::feedforward::transformer::decoder::PyTransformerDecoder;
use crate::nn::feedforward::transformer::encoder::PyTransformerEncoder;
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Full encoder–decoder transformer (Pre-LayerNorm).
///
/// Composes a `TransformerEncoder` and `TransformerDecoder`. Both sub-modules
/// are accessible and mutable from Python.
///
/// ```python
/// model = pycoeus.Transformer(d_model=64, d_ff=256, num_heads=4)
/// out = model.forward(src, tgt)   # src, tgt: [batch, seq, d_model]
/// ```
#[pyclass(name = "Transformer")]
pub struct PyTransformer {
    /// Encoder sub-module.
    #[pyo3(get)]
    pub encoder: Py<PyTransformerEncoder>,
    /// Decoder sub-module.
    #[pyo3(get)]
    pub decoder: Py<PyTransformerDecoder>,
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
impl PyTransformer {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads=8, num_enc_layers=6, num_dec_layers=6, dropout_p=0.0))]
    /// Create a full encoder–decoder transformer.
    pub fn new(
        py: Python<'_>,
        d_model: usize,
        d_ff: usize,
        num_heads: usize,
        num_enc_layers: usize,
        num_dec_layers: usize,
        dropout_p: f64,
    ) -> PyResult<Self> {
        if num_heads == 0 || d_model == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(PyValueError::new_err(format!(
                "Transformer: d_model ({d_model}) must be a positive multiple of num_heads ({num_heads})"
            )));
        }
        let encoder = Py::new(
            py,
            PyTransformerEncoder::new(py, d_model, d_ff, num_heads, num_enc_layers, dropout_p)?,
        )?;
        let decoder = Py::new(
            py,
            PyTransformerDecoder::new(py, d_model, d_ff, num_heads, num_dec_layers, dropout_p)?,
        )?;
        Ok(Self {
            encoder,
            decoder,
            d_model,
            d_ff,
            num_heads,
            dropout_p,
        })
    }

    /// Return the number of encoder layers.
    #[getter]
    pub fn num_enc_layers(&self, py: Python<'_>) -> usize {
        self.encoder.bind(py).borrow().num_layers()
    }

    /// Return the number of decoder layers.
    #[getter]
    pub fn num_dec_layers(&self, py: Python<'_>) -> usize {
        self.decoder.bind(py).borrow().num_layers()
    }

    /// Full seq2seq forward: encode `src`, then decode `tgt` with encoder memory.
    ///
    /// - `src`: `[batch, seq_src, d_model]`
    /// - `tgt`: `[batch, seq_tgt, d_model]`
    ///
    /// Returns `[batch, seq_tgt, d_model]`.
    pub fn forward(&self, src: &PyTensor, tgt: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let memory = self.encoder.bind(py).borrow().forward(src, py)?;
        self.decoder.bind(py).borrow().forward(tgt, &memory, py)
    }

    /// Return all learnable parameters across encoder and decoder.
    /// Collect all learnable parameters from encoder and decoder.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut p = self.encoder.bind(py).borrow().parameters(py);
        p.extend(self.decoder.bind(py).borrow().parameters(py));
        p
    }

    /// Zero gradients of all parameters across encoder and decoder.
    /// Zero all encoder and decoder parameter gradients.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.encoder.bind(py).borrow().zero_grad(py);
        self.decoder.bind(py).borrow().zero_grad(py);
    }
}
