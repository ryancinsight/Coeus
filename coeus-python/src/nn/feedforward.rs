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

        let inner = py.allow_threads(move || -> PyResult<_> {
            macro_rules! dispatch {
                ($($h:literal),*) => {
                    match num_heads {
                        $($h => {
                            use coeus_nn::transformer::decoder_layer::TransformerDecoderLayer;
                            let dec = TransformerDecoderLayer::<
                                f64, coeus_core::MoiraiBackend, $h,
                                coeus_autograd::CausalMask, coeus_autograd::NullMask,
                            >::new(d_model, d_ff, dropout_p);
                            Ok(dec.forward_decoder(&tgt_var, &mem_var))
                        },)*
                        _ => Err(PyValueError::new_err(format!(
                            "TransformerDecoderLayer: unsupported num_heads={num_heads}; supported: 1,2,4,8,16,32"
                        ))),
                    }
                }
            }
            dispatch!(1, 2, 4, 8, 16, 32)
        });
        Ok(PyTensor::from_var(inner?))
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        // Weights are constructed fresh each forward pass (stateless wrapper).
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}

// ── TransformerEncoderLayer ──────────────────────────────────────────────────

/// Python-exposed Transformer Encoder Layer (Pre-LayerNorm).
///
/// ```python
/// enc = pycoeus.TransformerEncoderLayer(d_model=64, d_ff=256, num_heads=4)
/// out = enc.forward(src)   # src: [batch, seq, d_model]
/// ```
#[pyclass(name = "TransformerEncoderLayer")]
pub struct PyTransformerEncoderLayer {
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
impl PyTransformerEncoderLayer {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, dropout_p = 0.0))]
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize, dropout_p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(
                "TransformerEncoderLayer: dropout_p must be in [0.0, 1.0)",
            ));
        }
        Ok(Self {
            d_model,
            d_ff,
            num_heads,
            dropout_p,
        })
    }

    /// Self-attention encoder forward.
    ///
    /// - `src`: `[batch, seq, d_model]`
    ///
    /// Returns `[batch, seq, d_model]`.
    pub fn forward(&self, src: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let src_var = src.inner.clone();
        let (d_model, d_ff, num_heads, dropout_p) =
            (self.d_model, self.d_ff, self.num_heads, self.dropout_p);
        let inner = py.allow_threads(move || -> PyResult<_> {
            macro_rules! dispatch {
                ($($h:literal),*) => {
                    match num_heads {
                        $($h => {
                            use coeus_nn::transformer::encoder_layer::TransformerEncoderLayer;
                            use coeus_nn::Module as _;
                            use coeus_autograd::NullMask;
                            let enc = TransformerEncoderLayer::<
                                f64, coeus_core::MoiraiBackend, $h, NullMask,
                            >::new(d_model, d_ff, dropout_p);
                            Ok(enc.forward(&src_var))
                        },)*
                        _ => Err(PyValueError::new_err(format!(
                            "TransformerEncoderLayer: unsupported num_heads={num_heads}; supported: 1,2,4,8,16,32"
                        ))),
                    }
                }
            }
            dispatch!(1, 2, 4, 8, 16, 32)
        });
        Ok(PyTensor::from_var(inner?))
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}

// ── TransformerEncoder ───────────────────────────────────────────────────────

/// Python-exposed Transformer Encoder stack (Pre-LayerNorm, N layers).
///
/// ```python
/// enc = pycoeus.TransformerEncoder(d_model=64, d_ff=256, num_heads=4, num_layers=6)
/// out = enc.forward(src)   # src: [batch, seq, d_model]
/// ```
#[pyclass(name = "TransformerEncoder")]
pub struct PyTransformerEncoder {
    #[pyo3(get)]
    pub d_model: usize,
    #[pyo3(get)]
    pub d_ff: usize,
    #[pyo3(get)]
    pub num_heads: usize,
    #[pyo3(get)]
    pub num_layers: usize,
    #[pyo3(get)]
    pub dropout_p: f64,
}

#[pymethods]
impl PyTransformerEncoder {
    #[new]
    #[pyo3(signature = (d_model, d_ff, num_heads = 8, num_layers = 6, dropout_p = 0.0))]
    pub fn new(
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
        Ok(Self {
            d_model,
            d_ff,
            num_heads,
            num_layers,
            dropout_p,
        })
    }

    /// Stack of encoder layers forward.
    ///
    /// - `src`: `[batch, seq, d_model]`
    ///
    /// Returns `[batch, seq, d_model]`.
    pub fn forward(&self, src: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let src_var = src.inner.clone();
        let (d_model, d_ff, num_heads, num_layers, dropout_p) = (
            self.d_model,
            self.d_ff,
            self.num_heads,
            self.num_layers,
            self.dropout_p,
        );
        let inner = py.allow_threads(move || -> PyResult<_> {
            macro_rules! dispatch {
                ($(($h:literal, $n:literal)),*) => {
                    match (num_heads, num_layers) {
                        $(($h, $n) => {
                            use coeus_nn::transformer::encoder::TransformerEncoder;
                            use coeus_nn::Module as _;
                            use coeus_autograd::NullMask;
                            let enc = TransformerEncoder::<
                                f64, coeus_core::MoiraiBackend, $h, $n, NullMask,
                            >::new(d_model, d_ff, dropout_p);
                            Ok(enc.forward(&src_var))
                        },)*
                        _ => Err(PyValueError::new_err(format!(
                            "TransformerEncoder: unsupported (num_heads={num_heads}, num_layers={num_layers}); \
                             supported heads: 1,2,4,8,16,32 and layers: 1,2,4,6,8,12"
                        ))),
                    }
                }
            }
            dispatch!(
                (1,1),(1,2),(1,4),(1,6),(1,8),(1,12),
                (2,1),(2,2),(2,4),(2,6),(2,8),(2,12),
                (4,1),(4,2),(4,4),(4,6),(4,8),(4,12),
                (8,1),(8,2),(8,4),(8,6),(8,8),(8,12),
                (16,1),(16,2),(16,4),(16,6),(16,8),(16,12),
                (32,1),(32,2),(32,4),(32,6),(32,8),(32,12)
            )
        });
        Ok(PyTensor::from_var(inner?))
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
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
    #[pyo3(get)]
    pub max_len: usize,
    #[pyo3(get)]
    pub d_model: usize,
}

#[pymethods]
impl PySinusoidalEncoding {
    #[new]
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
