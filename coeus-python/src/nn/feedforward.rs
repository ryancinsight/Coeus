use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-exposed FeedForward (2-layer MLP) transformer sub-block.
#[pyclass(name = "FeedForward")]
pub struct PyFeedForward {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout_p: f64,
}

#[pymethods]
impl PyFeedForward {
    #[new]
    #[pyo3(signature = (d_model, d_ff, dropout_p = 0.0))]
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "FeedForward: dropout_p must be in [0.0, 1.0)",
            ));
        }
        Ok(Self {
            d_model,
            d_ff,
            dropout_p,
        })
    }

    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::transformer::ffn::FeedForward;
        use coeus_nn::Module;
        let (d_model, d_ff, dropout_p) = (self.d_model, self.d_ff, self.dropout_p);
        let x = input.inner.clone();
        let inner = py.allow_threads(move || {
            let ffn = FeedForward::<f64, coeus_core::MoiraiBackend>::new(d_model, d_ff, dropout_p);
            ffn.forward(&x)
        });
        Ok(PyTensor::from_var(inner))
    }
}

/// Python-exposed Transformer Decoder Layer (Pre-LayerNorm).
///
/// ```python
/// dec = pycoeus.TransformerDecoderLayer(d_model=64, d_ff=256, num_heads=4)
/// out = dec.forward(tgt, memory)   # tgt, memory: [batch, seq, d_model]
/// ```
#[pyclass(name = "TransformerDecoderLayer")]
pub struct PyTransformerDecoderLayer {
    #[pyo3(get)]
    pub d_model: usize,
    #[pyo3(get)]
    pub d_ff: usize,
    #[pyo3(get)]
    pub num_heads: usize,
    #[pyo3(get)]
    pub dropout_p: f64,
}

#[pymethods]
impl PyTransformerDecoderLayer {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, dropout_p = 0.0))]
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize, dropout_p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "TransformerDecoderLayer: dropout_p must be in [0.0, 1.0)",
            ));
        }
        Ok(Self {
            d_model,
            d_ff,
            num_heads,
            dropout_p,
        })
    }

    /// Cross-attention decoder forward.
    ///
    /// - `tgt`:    `[batch, seq_tgt, d_model]`
    /// - `memory`: `[batch, seq_src, d_model]`
    ///
    /// Returns `[batch, seq_tgt, d_model]`.
    #[pyo3(signature = (tgt, memory))]
    pub fn forward(&self, tgt: &PyTensor, memory: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let tgt_var = tgt.inner.clone();
        let mem_var = memory.inner.clone();
        let (d_model, d_ff, num_heads, dropout_p) =
            (self.d_model, self.d_ff, self.num_heads, self.dropout_p);

        let inner = py.allow_threads(move || {
            macro_rules! dispatch {
                ($($h:literal),*) => {
                    match num_heads {
                        $($h => {
                            use coeus_nn::transformer::decoder_layer::TransformerDecoderLayer;
                            let dec = TransformerDecoderLayer::<
                                f64, coeus_core::MoiraiBackend, $h,
                                coeus_autograd::CausalMask, coeus_autograd::NullMask,
                            >::new(d_model, d_ff, dropout_p);
                            dec.forward_decoder(&tgt_var, &mem_var)
                        },)*
                        _ => panic!("TransformerDecoderLayer: unsupported num_heads={num_heads}; supported: 1,2,4,8,16,32"),
                    }
                }
            }
            dispatch!(1, 2, 4, 8, 16, 32)
        });
        Ok(PyTensor::from_var(inner))
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        // Weights are constructed fresh each forward pass (stateless wrapper).
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}
