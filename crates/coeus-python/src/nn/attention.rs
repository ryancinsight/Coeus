use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

pub(crate) fn validate_key_padding_mask(mask: Option<&PyTensor>) -> PyResult<()> {
    let Some(mask) = mask else {
        return Ok(());
    };
    let values = mask.inner.tensor.to_contiguous();
    if let Some((index, value)) = values
        .as_slice()
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value != 0.0 && *value != 1.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "key_padding_mask[{index}] must be 0 or 1, got {value}"
        )));
    }
    Ok(())
}

/// Stateless Scaled Dot-Product Attention module.
///
/// Implements scaled dot-product attention as a callable `nn.Module`.
/// `key_padding_mask` is an exact binary mask: `0` excludes a key and `1`
/// keeps it. Additive PyTorch `attn_mask` values are not accepted.
///
/// # Shapes
/// - `query`:  `[batch, seq_q, d_k]`
/// - `key`:    `[batch, seq_k, d_k]`
/// - `value`:  `[batch, seq_k, d_v]`
///
/// Returns `[batch, seq_q, d_v]`.
///
/// # Example
/// ```python
/// sdpa = pycoeus.ScaledDotProductAttention(scale=0.125)
/// out = sdpa.forward(q, k, v)
/// ```
#[pyclass(name = "ScaledDotProductAttention")]
#[derive(Clone)]
pub struct PyScaledDotProductAttention {
    /// Optional scaling factor applied before the softmax.  If `None` the
    /// default `1 / sqrt(d_k)` scaling is computed at forward-pass time.
    #[pyo3(get)]
    pub scale: Option<f64>,
    /// If `true` applies a causal lower-triangular mask.
    #[pyo3(get)]
    pub is_causal: bool,
}

#[pymethods]
impl PyScaledDotProductAttention {
    #[new]
    #[pyo3(signature = (scale = None, is_causal = false))]
    /// Create a ScaledDotProductAttention with optional manual `scale` and causal masking flag.
    pub fn new(scale: Option<f64>, is_causal: bool) -> Self {
        Self { scale, is_causal }
    }

    /// Forward pass: `(query, key, value) → attended output`.
    #[pyo3(signature = (query, key, value, key_padding_mask = None))]
    pub fn forward(
        &self,
        query: &PyTensor,
        key: &PyTensor,
        value: &PyTensor,
        key_padding_mask: Option<&PyTensor>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        validate_key_padding_mask(key_padding_mask)?;
        let q = query.inner.clone();
        let k = key.inner.clone();
        let v = value.inner.clone();
        let mask = key_padding_mask.map(|m| m.inner.clone());
        let scale = self.scale;
        let is_causal = self.is_causal;

        let inner = py.allow_threads(move || {
            if is_causal {
                let attention = coeus_nn::ScaledDotProductAttention::<
                    f64,
                    coeus_core::MoiraiBackend,
                    coeus_autograd::CausalMask,
                >::new();
                attention.forward(&q, &k, &v, mask.as_ref(), scale)
            } else {
                let attention = coeus_nn::ScaledDotProductAttention::<
                    f64,
                    coeus_core::MoiraiBackend,
                    coeus_autograd::NullMask,
                >::new();
                attention.forward(&q, &k, &v, mask.as_ref(), scale)
            }
        });
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Returns an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> PyStateDict {
        PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// No-op load (no parameters to restore).
    pub fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }

    /// Returns an empty parameter list.
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// No-op (no gradients to zero).
    pub fn zero_grad(&self, _py: Python<'_>) {}
}
///
/// Supported `num_heads` values at runtime: 1, 2, 4, 8, 16, 32.
/// `d_model` must be divisible by `num_heads`.
#[pyclass(name = "MultiHeadAttention")]
pub struct PyMultiHeadAttention {
    /// Query projection weight, shape `[d_model, d_model]`.
    #[pyo3(get)]
    pub w_q: Py<PyTensor>,
    /// Query projection bias, shape `[d_model]`.
    #[pyo3(get)]
    pub b_q: Option<Py<PyTensor>>,
    /// Key projection weight, shape `[d_model, d_model]`.
    #[pyo3(get)]
    pub w_k: Py<PyTensor>,
    /// Key projection bias, shape `[d_model]`.
    #[pyo3(get)]
    pub b_k: Option<Py<PyTensor>>,
    /// Value projection weight, shape `[d_model, d_model]`.
    #[pyo3(get)]
    pub w_v: Py<PyTensor>,
    /// Value projection bias, shape `[d_model]`.
    #[pyo3(get)]
    pub b_v: Option<Py<PyTensor>>,
    /// Output projection weight, shape `[d_model, d_model]`.
    #[pyo3(get)]
    pub w_o: Py<PyTensor>,
    /// Output projection bias, shape `[d_model]`.
    #[pyo3(get)]
    pub b_o: Option<Py<PyTensor>>,
    /// Model embedding dimensionality.
    #[pyo3(get)]
    pub d_model: usize,
    /// Number of attention heads.
    #[pyo3(get)]
    pub num_heads: usize,
}

#[pymethods]
impl PyMultiHeadAttention {
    #[new]
    #[pyo3(signature = (d_model, num_heads = 8, bias = true))]
    /// Create a MultiHeadAttention layer with `d_model` embedding dim and `num_heads` heads.
    pub fn new(py: Python<'_>, d_model: usize, num_heads: usize, bias: bool) -> PyResult<Self> {
        if num_heads == 0 || !d_model.is_multiple_of(num_heads) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "PyMultiHeadAttention: d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )));
        }
        // Construct via the monomorphized new() for the matching H, then extract weights.
        macro_rules! dispatch_mha_new {
            ($($h:literal),*) => {
                match num_heads {
                    $($h => {
                        let mha = coeus_nn::attention::mha::MultiHeadAttention::<
                            f64, coeus_core::MoiraiBackend, $h, coeus_autograd::NullMask,
                        >::new(d_model, bias);
                        let w_q = Py::new(py, PyTensor { inner: mha.w_q })?;
                        let b_q = if let Some(b) = mha.b_q { Some(Py::new(py, PyTensor { inner: b })?) } else { None };
                        let w_k = Py::new(py, PyTensor { inner: mha.w_k })?;
                        let b_k = if let Some(b) = mha.b_k { Some(Py::new(py, PyTensor { inner: b })?) } else { None };
                        let w_v = Py::new(py, PyTensor { inner: mha.w_v })?;
                        let b_v = if let Some(b) = mha.b_v { Some(Py::new(py, PyTensor { inner: b })?) } else { None };
                        let w_o = Py::new(py, PyTensor { inner: mha.w_o })?;
                        let b_o = if let Some(b) = mha.b_o { Some(Py::new(py, PyTensor { inner: b })?) } else { None };
                        Ok(Self { w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, d_model, num_heads })
                    },)*
                    _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "PyMultiHeadAttention: unsupported num_heads={num_heads}; \
                         supported: 1,2,4,8,16,32"
                    ))),
                }
            }
        }
        dispatch_mha_new!(1, 2, 4, 8, 16, 32)
    }

    /// Self-attention forward pass (Q = K = V = input).
    ///
    /// Input shape: `[batch, seq, d_model]`. Returns same shape.
    #[pyo3(signature = (input, key_padding_mask = None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        key_padding_mask: Option<&PyTensor>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        validate_key_padding_mask(key_padding_mask)?;
        let wq = self.w_q.bind(py).borrow().inner.clone();
        let bq = self.b_q.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let wk = self.w_k.bind(py).borrow().inner.clone();
        let bk = self.b_k.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let wv = self.w_v.bind(py).borrow().inner.clone();
        let bv = self.b_v.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let wo = self.w_o.bind(py).borrow().inner.clone();
        let bo = self.b_o.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let input_var = input.inner.clone();
        let mask_var = key_padding_mask.map(|m| m.inner.clone());
        let num_heads = self.num_heads;

        let inner = py.allow_threads(move || {
            macro_rules! dispatch_mha_fwd {
                ($($h:literal),*) => {
                    match num_heads {
                        $($h => {
                            coeus_nn::multi_head_attention_cross::<
                                f64, coeus_core::MoiraiBackend, $h, coeus_autograd::NullMask,
                            >(
                                &input_var,
                                &input_var,
                                &input_var,
                                coeus_nn::MhaProjectionParams {
                                    w_q: &wq,
                                    b_q: bq.as_ref(),
                                    w_k: &wk,
                                    b_k: bk.as_ref(),
                                    w_v: &wv,
                                    b_v: bv.as_ref(),
                                    w_o: &wo,
                                    b_o: bo.as_ref(),
                                },
                                mask_var.as_ref(),
                            )
                        },)*
                        _ => panic!(
                            "PyMultiHeadAttention: unsupported num_heads={num_heads}"
                        ),
                    }
                }
            }
            dispatch_mha_fwd!(1, 2, 4, 8, 16, 32)
        });
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Cross-attention forward pass.
    ///
    /// - `query`: `[batch, seq_q, d_model]`
    /// - `key`:   `[batch, seq_k, d_model]`
    /// - `value`: `[batch, seq_k, d_model]`
    ///
    /// Returns `[batch, seq_q, d_model]`.
    #[pyo3(signature = (query, key, value, key_padding_mask = None))]
    pub fn forward_cross(
        &self,
        query: &PyTensor,
        key: &PyTensor,
        value: &PyTensor,
        key_padding_mask: Option<&PyTensor>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        validate_key_padding_mask(key_padding_mask)?;
        let wq = self.w_q.bind(py).borrow().inner.clone();
        let bq = self.b_q.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let wk = self.w_k.bind(py).borrow().inner.clone();
        let bk = self.b_k.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let wv = self.w_v.bind(py).borrow().inner.clone();
        let bv = self.b_v.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let wo = self.w_o.bind(py).borrow().inner.clone();
        let bo = self.b_o.as_ref().map(|b| b.bind(py).borrow().inner.clone());
        let q_var = query.inner.clone();
        let k_var = key.inner.clone();
        let v_var = value.inner.clone();
        let mask_var = key_padding_mask.map(|m| m.inner.clone());
        let num_heads = self.num_heads;

        let inner = py.allow_threads(move || {
            macro_rules! dispatch_mha_cross {
                ($($h:literal),*) => {
                    match num_heads {
                        $($h => {
                            coeus_nn::multi_head_attention_cross::<
                                f64, coeus_core::MoiraiBackend, $h, coeus_autograd::NullMask,
                            >(
                                &q_var,
                                &k_var,
                                &v_var,
                                coeus_nn::MhaProjectionParams {
                                    w_q: &wq,
                                    b_q: bq.as_ref(),
                                    w_k: &wk,
                                    b_k: bk.as_ref(),
                                    w_v: &wv,
                                    b_v: bv.as_ref(),
                                    w_o: &wo,
                                    b_o: bo.as_ref(),
                                },
                                mask_var.as_ref(),
                            )
                        },)*
                        _ => panic!(
                            "PyMultiHeadAttention: unsupported num_heads={num_heads}"
                        ),
                    }
                }
            }
            dispatch_mha_cross!(1, 2, 4, 8, 16, 32)
        });
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("w_q", self.w_q.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.b_q {
            sd.insert("b_q", b.bind(py).borrow().inner.tensor.clone());
        }
        sd.insert("w_k", self.w_k.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.b_k {
            sd.insert("b_k", b.bind(py).borrow().inner.tensor.clone());
        }
        sd.insert("w_v", self.w_v.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.b_v {
            sd.insert("b_v", b.bind(py).borrow().inner.tensor.clone());
        }
        sd.insert("w_o", self.w_o.bind(py).borrow().inner.tensor.clone());
        if let Some(ref b) = self.b_o {
            sd.insert("b_o", b.bind(py).borrow().inner.tensor.clone());
        }
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("w_q") {
            self.w_q.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("b_q") {
            if let Some(ref my_b) = self.b_q {
                my_b.bind(py).borrow_mut().inner.tensor = b.clone();
            }
        }
        if let Some(w) = state_dict.inner.get("w_k") {
            self.w_k.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("b_k") {
            if let Some(ref my_b) = self.b_k {
                my_b.bind(py).borrow_mut().inner.tensor = b.clone();
            }
        }
        if let Some(w) = state_dict.inner.get("w_v") {
            self.w_v.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("b_v") {
            if let Some(ref my_b) = self.b_v {
                my_b.bind(py).borrow_mut().inner.tensor = b.clone();
            }
        }
        if let Some(w) = state_dict.inner.get("w_o") {
            self.w_o.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        if let Some(b) = state_dict.inner.get("b_o") {
            if let Some(ref my_b) = self.b_o {
                my_b.bind(py).borrow_mut().inner.tensor = b.clone();
            }
        }
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut params = vec![
            self.w_q.clone_ref(py),
            self.w_k.clone_ref(py),
            self.w_v.clone_ref(py),
            self.w_o.clone_ref(py),
        ];
        if let Some(ref b) = self.b_q {
            params.push(b.clone_ref(py));
        }
        if let Some(ref b) = self.b_k {
            params.push(b.clone_ref(py));
        }
        if let Some(ref b) = self.b_v {
            params.push(b.clone_ref(py));
        }
        if let Some(ref b) = self.b_o {
            params.push(b.clone_ref(py));
        }
        params
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.w_q.bind(py).borrow().zero_grad();
        self.w_k.bind(py).borrow().zero_grad();
        self.w_v.bind(py).borrow().zero_grad();
        self.w_o.bind(py).borrow().zero_grad();
        if let Some(ref b) = self.b_q {
            b.bind(py).borrow().zero_grad();
        }
        if let Some(ref b) = self.b_k {
            b.bind(py).borrow().zero_grad();
        }
        if let Some(ref b) = self.b_v {
            b.bind(py).borrow().zero_grad();
        }
        if let Some(ref b) = self.b_o {
            b.bind(py).borrow().zero_grad();
        }
    }
}

/// Python-exposed Rotary Positional Embedding (RoPE) layer.
#[pyclass(name = "RotaryEmbedding")]
pub struct PyRotaryEmbedding {
    /// Underlying Rust RotaryEmbedding instance.
    pub inner: coeus_nn::positional::RotaryEmbedding<f64, coeus_core::MoiraiBackend>,
    /// Maximum sequence length for which sinusoidal tables are pre-computed.
    #[pyo3(get)]
    pub max_len: usize,
    /// Per-head key/query dimension.
    #[pyo3(get)]
    pub d_head: usize,
    /// Base for the frequency progression (default 10000).
    #[pyo3(get)]
    pub base: f64,
}

#[pymethods]
impl PyRotaryEmbedding {
    #[new]
    #[pyo3(signature = (max_len, d_head, base = 10000.0))]
    /// Create a RotaryEmbedding with `max_len` positions, `d_head` key/query dim, and frequency `base`.
    pub fn new(max_len: usize, d_head: usize, base: f64) -> Self {
        let inner = coeus_nn::positional::RotaryEmbedding::new(max_len, d_head, base);
        Self {
            inner,
            max_len,
            d_head,
            base,
        }
    }

    /// Forward pass through the RotaryEmbedding layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let input_var = input.inner.clone();
        let rope = self.inner.clone();

        let inner = py.allow_threads(move || rope.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, _py: Python<'_>) {}
}
