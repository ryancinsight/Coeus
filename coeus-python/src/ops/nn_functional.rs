use crate::tensor::PyTensor;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (input, weight, bias = None))]
pub fn linear(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    py: Python<'_>,
) -> PyTensor {
    let w = weight.inner.clone();
    let b = bias.map(|b| b.inner.clone());
    let x = input.inner.clone();
    let inner = py.allow_threads(move || {
        use coeus_nn::linear::Linear;
        let lin = Linear { weight: w, bias: b };
        use coeus_nn::Module;
        lin.forward(&x)
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (input, norm_shape, weight = None, bias = None, eps = 1e-5))]
pub fn layer_norm(
    input: &PyTensor,
    norm_shape: usize,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: f64,
    py: Python<'_>,
) -> PyTensor {
    let w = weight.map(|w| w.inner.clone());
    let b = bias.map(|b| b.inner.clone());
    let x = input.inner.clone();
    let ndim = x.tensor.ndim();
    let inner = py.allow_threads(move || {
        use coeus_nn::normalization::LayerNorm;
        let backend = coeus_core::MoiraiBackend::new();
        let weight_var = w.unwrap_or_else(|| {
            coeus_autograd::Var::new(Tensor::ones_on([norm_shape], &backend), false)
        });
        let bias_var = b.unwrap_or_else(|| {
            coeus_autograd::Var::new(Tensor::zeros_on([norm_shape], &backend), false)
        });
        let ln = LayerNorm::from_parts(weight_var, bias_var, eps);
        if ndim == 2 {
            use coeus_nn::Module;
            ln.forward(&x)
        } else {
            ln.forward_nd(&x)
        }
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (input, p = 0.5, training = false))]
pub fn dropout(input: &PyTensor, p: f64, training: bool, py: Python<'_>) -> PyTensor {
    if !training || p == 0.0 {
        return input.clone();
    }
    let x = input.inner.clone();
    let inner = py.allow_threads(move || {
        use coeus_nn::Dropout;
        use coeus_nn::Module;
        let mut drop = Dropout::new(p);
        drop.set_training(true);
        drop.forward(&x)
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (input, size, mode = "nearest"))]
pub fn interpolate(
    input: &PyTensor,
    size: Vec<usize>,
    mode: &str,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let imode = match mode {
        "nearest" => coeus_nn::InterpolateMode::Nearest,
        "bilinear" => coeus_nn::InterpolateMode::Bilinear,
        other => {
            return Err(PyValueError::new_err(format!(
                "interpolate: unknown mode '{other}'; expected 'nearest' or 'bilinear'"
            )));
        }
    };
    let ndim = input.inner.tensor.ndim();
    let t = match ndim {
        3 => {
            assert_eq!(size.len(), 1, "interpolate: 1-D input needs size=[new_L]");
            py.allow_threads(|| coeus_nn::interpolate_1d(&input.inner.tensor, size[0], imode))
        }
        4 => {
            assert_eq!(
                size.len(),
                2,
                "interpolate: 2-D input needs size=[new_H, new_W]"
            );
            py.allow_threads(|| {
                coeus_nn::interpolate_2d(&input.inner.tensor, size[0], size[1], imode)
            })
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "interpolate: expected 3-D or 4-D input, got {ndim}-D"
            )));
        }
    };
    Ok(PyTensor {
        inner: coeus_autograd::Var::new(t, false),
    })
}

#[pyfunction]
#[pyo3(signature = (input, pads, value = 0.0))]
pub fn pad(input: &PyTensor, pads: Vec<(usize, usize)>, value: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::pad(&input.inner, &pads, value));
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (v, k = 0))]
pub fn diag(v: &PyTensor, k: i64, py: Python<'_>) -> PyResult<PyTensor> {
    if v.inner.tensor.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "diag: input must be 1-D, got {}-D",
            v.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::diag(&v.inner, k as isize));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
#[pyo3(signature = (m, k = 0))]
pub fn diagonal(m: &PyTensor, k: i64, py: Python<'_>) -> PyResult<PyTensor> {
    if m.inner.tensor.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "diagonal: input must be 2-D, got {}-D",
            m.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::diagonal(&m.inner, k as isize));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn matmul(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::matmul(&a.inner, &b.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn bmm(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let a_shape = a.inner.tensor.shape();
    let b_shape = b.inner.tensor.shape();
    if a_shape.len() != 3 || b_shape.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "bmm: expected 3-D inputs, got ranks {} and {}",
            a_shape.len(),
            b_shape.len()
        )));
    }
    if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
        return Err(PyValueError::new_err(format!(
            "bmm: incompatible shapes {:?} and {:?}",
            a_shape, b_shape
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::matmul(&a.inner, &b.inner));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn outer(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let a_shape = a.inner.tensor.shape();
    let b_shape = b.inner.tensor.shape();
    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "outer: expected 1-D inputs, got ranks {} and {}",
            a_shape.len(),
            b_shape.len()
        )));
    }
    let rows = a_shape[0];
    let cols = b_shape[0];
    let inner = py.allow_threads(|| {
        let a_col = coeus_autograd::reshape(&a.inner, vec![rows, 1]);
        let b_row = coeus_autograd::reshape(&b.inner, vec![1, cols]);
        coeus_autograd::matmul(&a_col, &b_row)
    });
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn softmax(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softmax(&input.inner, dim as isize));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn einsum(subscript: &str, operands: Vec<pyo3::Py<PyTensor>>, py: Python<'_>) -> PyTensor {
    let rust_vars: Vec<coeus_autograd::Var<f64>> = operands
        .iter()
        .map(|t| t.bind(py).borrow().inner.clone())
        .collect();
    let inner = py.allow_threads(move || {
        let refs: Vec<&coeus_autograd::Var<f64>> = rust_vars.iter().collect();
        coeus_autograd::einsum(subscript, &refs)
    });
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_softmax(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softmax(&input.inner, dim as isize));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_log_softmax(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log_softmax(&input.inner, dim));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_relu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::relu(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_sigmoid(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sigmoid(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_tanh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::tanh(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_gelu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::gelu(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_silu(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::silu(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn f_mse_loss(input: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    if input.inner.tensor.shape() != target.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "f_mse_loss: input shape {:?} must match target shape {:?}",
            input.inner.tensor.shape(),
            target.inner.tensor.shape()
        )));
    }
    let inner = py.allow_threads(|| coeus_nn::mse_loss(&input.inner, &target.inner));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn f_binary_cross_entropy(
    input: &PyTensor,
    target: &PyTensor,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if input.inner.tensor.shape() != target.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "f_binary_cross_entropy: input shape {:?} must match target shape {:?}",
            input.inner.tensor.shape(),
            target.inner.tensor.shape()
        )));
    }
    let inner =
        py.allow_threads(|| coeus_nn::binary_cross_entropy(&input.inner, &target.inner, 1e-7));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn f_cross_entropy(input: &PyTensor, targets: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::cross_entropy_loss(&input.inner, &targets));
    PyTensor::from_var(inner)
}

/// Functional (stateless) scaled dot-product attention.
///
/// Equivalent to `torch.nn.functional.scaled_dot_product_attention` /
/// `F.scaled_dot_product_attention`.
///
/// # Shapes
/// - `query`:  `[batch, seq_q, d_k]`
/// - `key`:    `[batch, seq_k, d_k]`
/// - `value`:  `[batch, seq_k, d_v]`
///
/// Returns the attended output `[batch, seq_q, d_v]`.
#[pyfunction]
#[pyo3(signature = (query, key, value, attn_mask = None, scale = None, is_causal = false))]
pub fn scaled_dot_product_attention(
    query: &PyTensor,
    key: &PyTensor,
    value: &PyTensor,
    attn_mask: Option<&PyTensor>,
    scale: Option<f64>,
    is_causal: bool,
    py: Python<'_>,
) -> PyTensor {
    let q = query.inner.clone();
    let k = key.inner.clone();
    let v = value.inner.clone();
    let mask = attn_mask.map(|m| m.inner.clone());
    let d_k = q.tensor.shape().last().copied().unwrap_or(1);
    let scale = scale.unwrap_or_else(|| 1.0 / (d_k as f64).sqrt());

    let inner = py.allow_threads(move || {
        let (out, _attn) =
            if is_causal {
                coeus_autograd::sdp_attention::<
                    f64,
                    coeus_core::MoiraiBackend,
                    coeus_autograd::CausalMask,
                >(&q, &k, &v, mask.as_ref(), scale)
            } else {
                coeus_autograd::sdp_attention::<
                    f64,
                    coeus_core::MoiraiBackend,
                    coeus_autograd::NullMask,
                >(&q, &k, &v, mask.as_ref(), scale)
            };
        out
    });
    PyTensor::from_var(inner)
}
