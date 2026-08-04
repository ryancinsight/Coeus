mod normalization;

pub use normalization::{batch_norm_1d, group_norm, layer_norm, rms_norm};

use crate::{
    nn::error::{map_backend_error, map_module_error},
    tensor::PyTensor,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (input, weight, bias = None))]
pub fn linear(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let w = weight.inner.clone();
    let b = bias.map(|b| b.inner.clone());
    let x = input.inner.clone();
    let inner = py.allow_threads(move || {
        use coeus_nn::linear::Linear;
        let lin = Linear { weight: w, bias: b };
        use coeus_nn::Module;
        lin.forward(&x)
    });
    inner.map(PyTensor::from_var).map_err(map_module_error)
}

#[pyfunction]
#[pyo3(signature = (input1, input2, weight, bias = None))]
pub fn bilinear(
    input1: &PyTensor,
    input2: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let x1_shape = input1.inner.tensor.shape();
    let x2_shape = input2.inner.tensor.shape();
    let w_shape = weight.inner.tensor.shape();
    if x1_shape.len() != 2 || x2_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "bilinear: input1 and input2 must be rank-2 tensors [batch, features]",
        ));
    }
    if w_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "bilinear: weight must be rank-3 [out_features, in1_features, in2_features]",
        ));
    }
    if x1_shape[0] != x2_shape[0] {
        return Err(PyValueError::new_err(
            "bilinear: input1 and input2 batch sizes must match",
        ));
    }
    if x1_shape[1] != w_shape[1] || x2_shape[1] != w_shape[2] {
        return Err(PyValueError::new_err(format!(
            "bilinear: incompatible shapes input1={x1_shape:?}, input2={x2_shape:?}, weight={w_shape:?}"
        )));
    }
    if let Some(b) = bias {
        if b.inner.tensor.shape() != [w_shape[0]] {
            return Err(PyValueError::new_err(
                "bilinear: bias must have shape [out_features]",
            ));
        }
    }

    let x1 = input1.inner.clone();
    let x2 = input2.inner.clone();
    let w = weight.inner.clone();
    let b = bias.map(|b| b.inner.clone());
    let inner = py.allow_threads(move || coeus_nn::bilinear::bilinear(&x1, &x2, &w, b.as_ref()));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
#[pyo3(signature = (input, p = 0.5, training = false))]
pub fn dropout(input: &PyTensor, p: f64, training: bool, py: Python<'_>) -> PyResult<PyTensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }
    let x = input.inner.clone();
    let inner = py.allow_threads(move || {
        use coeus_nn::Dropout;
        use coeus_nn::Module;
        let mut drop = Dropout::new(p);
        drop.set_training(true);
        drop.forward(&x)
    });
    inner.map(PyTensor::from_var).map_err(map_module_error)
}

#[pyfunction]
/// `F.interpolate(x, size=None, scale_factor=None, mode='nearest'|'bilinear')`.
///
/// Mirrors the PyTorch signature allowlist used in parity tests:
/// - `interpolate(x, [new_L], mode=...)` — explicit `size` for 1-D outputs.
/// - `interpolate(x, [new_H, new_W], mode=...)` — explicit `size` for 2-D
///   outputs.
/// - `interpolate(x, scale_factor=2.0, mode=...)` — uniform scale factor
///   applied to every spatial dim.
///
/// Exactly one of `size` or `scale_factor` must be supplied; passing both or
/// neither returns a typed `ValueError`.
#[pyo3(signature = (input, size=None, scale_factor=None, mode="nearest"))]
pub fn interpolate(
    input: &PyTensor,
    size: Option<Vec<usize>>,
    scale_factor: Option<f64>,
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
    let in_shape = input.inner.tensor.shape().to_vec();

    // Resolve the per-spatial-dim target shape — `size` takes precedence, then
    // `scale_factor`.  Cross-validate the option-pair so a caller passing both
    // / neither gets a typed error.
    let out_spatial: Vec<usize> = match (size.as_ref(), scale_factor) {
        (Some(s), None) => s.clone(),
        (None, Some(sf)) => {
            if !(sf > 0.0 && sf.is_finite()) {
                return Err(PyValueError::new_err(format!(
                    "interpolate: scale_factor must be a positive finite number, got {sf}"
                )));
            }
            in_shape[2..]
                .iter()
                .map(|&d| {
                    // PyTorch uses ceil(input * scale_factor) (matching the
                    // documented int-rounding for backward-compatible code).
                    let scaled = d as f64 * sf;
                    scaled.ceil() as usize
                })
                .collect()
        }
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "interpolate: only one of size or scale_factor may be specified",
            ));
        }
        (None, None) => {
            return Err(PyValueError::new_err(
                "interpolate: exactly one of size or scale_factor must be specified",
            ));
        }
    };

    let ndim = in_shape.len();
    let t = match ndim {
        3 => {
            if out_spatial.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "interpolate: 1-D (3-rank) input needs one spatial dim, got {}",
                    out_spatial.len()
                )));
            }
            py.allow_threads(|| {
                coeus_nn::interpolate_1d(&input.inner.tensor, out_spatial[0], imode)
            })
        }
        4 => {
            if out_spatial.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "interpolate: 2-D (4-rank) input needs two spatial dims, got {}",
                    out_spatial.len()
                )));
            }
            py.allow_threads(|| {
                coeus_nn::interpolate_2d(&input.inner.tensor, out_spatial[0], out_spatial[1], imode)
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

/// Softmin over `dim` (`torch.nn.functional.softmin`), i.e. `softmax(-input)`.
#[pyfunction]
pub fn softmin(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softmin(&input.inner, dim as isize));
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
        if refs.len() == 3 {
            coeus_autograd::einsum3(subscript, refs[0], refs[1], refs[2])
        } else {
            coeus_autograd::einsum(subscript, &refs)
        }
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
pub fn f_cross_entropy(
    input: &PyTensor,
    targets: Vec<usize>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_nn::cross_entropy_loss(&input.inner, &targets));
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}

/// Functional (stateless) scaled dot-product attention.
///
/// Scaled dot-product attention with an optional binary keep mask.
///
/// Unlike PyTorch's additive `attn_mask`, `key_padding_mask` accepts only 0
/// (masked) and 1 (kept). This explicit contract prevents floating values from
/// being silently reinterpreted as predicates.
///
/// # Shapes
/// - `query`:  `[batch, seq_q, d_k]`
/// - `key`:    `[batch, seq_k, d_k]`
/// - `value`:  `[batch, seq_k, d_v]`
///
/// Returns the attended output `[batch, seq_q, d_v]`.
#[pyfunction]
#[pyo3(signature = (query, key, value, key_padding_mask = None, scale = None, is_causal = false))]
pub fn scaled_dot_product_attention(
    query: &PyTensor,
    key: &PyTensor,
    value: &PyTensor,
    key_padding_mask: Option<&PyTensor>,
    scale: Option<f64>,
    is_causal: bool,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    crate::nn::attention::validate_key_padding_mask(key_padding_mask)?;
    let q = query.inner.clone();
    let k = key.inner.clone();
    let v = value.inner.clone();
    let mask = key_padding_mask.map(|m| m.inner.clone());
    let d_k = q.tensor.shape().last().copied().unwrap_or(1);
    let scale = scale.unwrap_or_else(|| 1.0 / (d_k as f64).sqrt());

    let inner = py.allow_threads(move || {
        let (out, _attn) =
            if is_causal {
                coeus_autograd::sdp_attention::<
                    f64,
                    coeus_core::MoiraiBackend,
                    coeus_autograd::CausalMask,
                >(&q, &k, &v, mask.as_ref(), scale)?
            } else {
                coeus_autograd::sdp_attention::<
                    f64,
                    coeus_core::MoiraiBackend,
                    coeus_autograd::NullMask,
                >(&q, &k, &v, mask.as_ref(), scale)?
            };
        Ok(out)
    });
    inner.map(PyTensor::from_var).map_err(map_backend_error)
}
