use crate::{
    init::map_initialization_error,
    nn::feedforward::transformer::encoder_layer::PyTransformerEncoderLayer, tensor::PyTensor,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Stack of `TransformerEncoderLayer`s forming the encoder half of a
/// transformer (encoder–decoder architecture).
///
/// Pre-LayerNorm, self-attention + FFN. All sub-modules accessible and
/// mutable from Python.
#[pyclass(name = "TransformerEncoder")]
pub struct PyTransformerEncoder {
    /// The ordered list of encoder layers.
    #[pyo3(get)]
    pub layers: Vec<Py<PyTransformerEncoderLayer>>,
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
impl PyTransformerEncoder {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, num_layers = 6, dropout_p = 0.0))]
    /// Create a `TransformerEncoder` with `num_layers` encoder layers.
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
                        >::new(d_model, d_ff, dropout_p)
                            .map_err(map_initialization_error)?;
                        enc.layers.into_iter()
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

    /// Return the number of encoder layers.
    #[getter]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass through all encoder layers sequentially.
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

    /// Return all learnable parameters across all encoder layers.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        self.layers
            .iter()
            .flat_map(|l| l.bind(py).borrow().parameters(py))
            .collect()
    }

    /// Zero gradients of all parameters across all encoder layers.
    pub fn zero_grad(&self, py: Python<'_>) {
        for l in &self.layers {
            l.bind(py).borrow().zero_grad(py);
        }
    }
}
