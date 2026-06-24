use crate::tensor::PyTensor;
use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── Internal helper ────────────────────────────────────────────────────────

/// Element-wise comparison: calls `f(a_elem, b_elem)` for each pair of elements
/// from contiguous CPU tensors of the same shape, returns a mask tensor.
///
/// Requires equal shapes; both tensors are materialised to contiguous first.
pub(crate) fn tensor_cmp<F>(
    a: &Tensor<f64, MoiraiBackend>,
    b: &Tensor<f64, MoiraiBackend>,
    f: F,
) -> PyResult<Tensor<f64, MoiraiBackend>>
where
    F: Fn(f64, f64) -> f64,
{
    let backend = MoiraiBackend::new();
    let a_c = a.to_contiguous_on(&backend);
    let b_c = b.to_contiguous_on(&backend);
    if a_c.shape() != b_c.shape() {
        return Err(PyValueError::new_err(format!(
            "tensor_cmp: shape mismatch: left={:?}, right={:?}",
            a_c.shape(),
            b_c.shape()
        )));
    }
    let a_s = a_c.as_slice();
    let b_s = b_c.as_slice();
    let out: Vec<f64> = a_s
        .iter()
        .zip(b_s.iter())
        .map(|(&ai, &bi)| f(ai, bi))
        .collect();
    Ok(Tensor::from_slice(a_c.shape().to_vec(), &out))
}

/// Element-wise exponential.
#[pyfunction]
pub fn exp(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::exp(&input.inner));
    PyTensor { inner }
}

/// Element-wise natural logarithm.
#[pyfunction]
pub fn log(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log(&input.inner));
    PyTensor { inner }
}

/// Sum along the specified axis.
#[pyfunction]
pub fn sum_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sum_axis(&input.inner, axis));
    PyTensor { inner }
}

/// Mean along the specified axis.
#[pyfunction]
pub fn mean_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::mean_axis(&input.inner, axis));
    PyTensor { inner }
}

/// Compute log-softmax along the specified axis.
#[pyfunction]
pub fn log_softmax(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log_softmax(&input.inner, axis));
    PyTensor { inner }
}

/// Cumulative sum along the specified axis.
#[pyfunction]
pub fn cumsum(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::cumsum(&input.inner, dim));
    PyTensor { inner }
}

/// Constant padding.
#[pyfunction]
#[pyo3(signature = (input, pads, value = 0.0))]
pub fn pad(input: &PyTensor, pads: Vec<(usize, usize)>, value: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::pad(&input.inner, &pads, value));
    PyTensor { inner }
}

/// Concatenate a sequence of tensors along the specified dimension.
#[pyfunction]
pub fn cat(inputs: Vec<Py<PyTensor>>, dim: usize, py: Python<'_>) -> PyTensor {
    let rust_inputs: Vec<coeus_autograd::Var<f64>> = inputs
        .iter()
        .map(|t| t.bind(py).borrow().inner.clone())
        .collect();
    let inner = py.allow_threads(move || {
        let ref_inputs: Vec<&coeus_autograd::Var<f64>> = rust_inputs.iter().collect();
        coeus_autograd::cat(&ref_inputs, dim)
    });
    PyTensor { inner }
}

/// Stack tensors along a new dimension.
///
/// All inputs must have the same shape.  The output has one extra dimension
/// of size `len(inputs)` inserted at position `dim`.
///
/// Example: `stack([a, b, c], dim=0)` with each of shape `[N]` → `[3, N]`.
#[pyfunction]
pub fn stack(inputs: Vec<Py<PyTensor>>, dim: usize, py: Python<'_>) -> PyTensor {
    let rust_inputs: Vec<coeus_autograd::Var<f64>> = inputs
        .iter()
        .map(|t| t.bind(py).borrow().inner.clone())
        .collect();
    let inner = py.allow_threads(move || {
        let ref_inputs: Vec<&coeus_autograd::Var<f64>> = rust_inputs.iter().collect();
        coeus_autograd::stack(&ref_inputs, dim)
    });
    PyTensor { inner }
}

/// Split a tensor into chunks of `chunk_size` along the specified dimension.
#[pyfunction]
pub fn split(input: &PyTensor, chunk_size: usize, dim: usize, py: Python<'_>) -> Vec<PyTensor> {
    let inner_chunks = py.allow_threads(|| coeus_autograd::split(&input.inner, chunk_size, dim));
    inner_chunks
        .into_iter()
        .map(|inner| PyTensor { inner })
        .collect()
}

/// Functional matmul: `a @ b`.
///
/// Equivalent to `a.__matmul__(b)`.  Provided as a free function to match the
/// `torch.matmul` / `jnp.matmul` / `mx.matmul` functional API style.
#[pyfunction]
pub fn matmul(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::matmul(&a.inner, &b.inner));
    PyTensor { inner }
}

/// Element-wise absolute value.
#[pyfunction]
pub fn abs(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::abs(&input.inner));
    PyTensor { inner }
}

/// Element-wise square root.
#[pyfunction]
pub fn sqrt(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sqrt(&input.inner));
    PyTensor { inner }
}

/// Element-wise negation.
#[pyfunction]
pub fn neg(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::neg(&input.inner));
    PyTensor { inner }
}

/// Element-wise clamp to `[min_val, max_val]`.
#[pyfunction]
pub fn clamp(input: &PyTensor, min_val: f64, max_val: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::clamp(&input.inner, min_val, max_val));
    PyTensor { inner }
}

/// Maximum along the specified axis (keep-dim).
#[pyfunction]
pub fn max_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::max_axis(&input.inner, axis));
    PyTensor { inner }
}

/// Minimum along the specified axis (keep-dim).
#[pyfunction]
pub fn min_axis(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::min_axis(&input.inner, axis));
    PyTensor { inner }
}

/// Numerically stable log-sum-exp along `axis`.
#[pyfunction]
pub fn log_sum_exp(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::log_sum_exp(&input.inner, axis));
    PyTensor { inner }
}

/// Global sum over all elements → scalar tensor of shape `[1]`.
#[pyfunction]
pub fn sum(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sum(&input.inner));
    PyTensor { inner }
}

/// Global mean over all elements → scalar tensor of shape `[1]`.
#[pyfunction]
pub fn mean(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::mean(&input.inner));
    PyTensor { inner }
}

// ── Tensor constructors ──────────────────────────────────────────────────

/// Create a zero tensor with the given shape.
///
/// `requires_grad` defaults to `False` (same as `torch.zeros`).
#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    let t = Tensor::<f64, MoiraiBackend>::zeros(shape);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

/// Create a ones tensor with the given shape.
#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn ones(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    let t = Tensor::<f64, MoiraiBackend>::ones(shape);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

/// Create a tensor filled with `value`.
#[pyfunction]
#[pyo3(signature = (shape, value, requires_grad = false))]
pub fn full(shape: Vec<usize>, value: f64, requires_grad: bool) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = Tensor::<f64, MoiraiBackend>::full_on(shape, value, &backend);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

/// Create a 1-D tensor of evenly spaced values in `[start, end)` with step `step`.
///
/// Matches `torch.arange(start, end, step)`.
#[pyfunction]
#[pyo3(signature = (start, end, step = 1.0))]
pub fn arange(start: f64, end: f64, step: f64) -> PyTensor {
    let n = ((end - start) / step).ceil() as usize;
    let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    let t = Tensor::from_slice(vec![n], &data);
    PyTensor {
        inner: Var::new(t, false),
    }
}

/// Create a 1-D tensor of `n` evenly spaced values from `start` to `end` (inclusive).
///
/// Matches `torch.linspace(start, end, steps)`.
#[pyfunction]
pub fn linspace(start: f64, end: f64, steps: usize) -> PyTensor {
    let data: Vec<f64> = if steps <= 1 {
        vec![start]
    } else {
        (0..steps)
            .map(|i| start + (end - start) * i as f64 / (steps - 1) as f64)
            .collect()
    };
    let t = Tensor::from_slice(vec![steps], &data);
    PyTensor {
        inner: Var::new(t, false),
    }
}

/// Reshape a tensor.  Equivalent to `tensor.reshape(shape)`.
#[pyfunction]
pub fn reshape(input: &PyTensor, shape: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::reshape(&input.inner, shape));
    PyTensor { inner }
}

/// Permute dimensions.  Equivalent to `tensor.permute(dims)`.
#[pyfunction]
pub fn permute(input: &PyTensor, dims: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::permute(&input.inner, &dims));
    PyTensor { inner }
}

/// 2-D transpose.  Equivalent to `tensor.t()`.
#[pyfunction]
pub fn t(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::transpose_2d(&input.inner));
    PyTensor { inner }
}

/// Power: `input ** exp` element-wise.
#[pyfunction]
pub fn pow(input: &PyTensor, exp: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::pow(&input.inner, exp));
    PyTensor { inner }
}

// ── Trigonometric ────────────────────────────────────────────────────────────

/// Element-wise sine with autograd.
#[pyfunction]
pub fn sin(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sin(&input.inner));
    PyTensor { inner }
}

/// Element-wise cosine with autograd.
#[pyfunction]
pub fn cos(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::cos(&input.inner));
    PyTensor { inner }
}

// ── Shape ops ────────────────────────────────────────────────────────────────

/// Flip the tensor along `axis` (reverse the elements).
///
/// Backward is a flip of the gradient along the same axis.
#[pyfunction]
pub fn flip(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::flip(&input.inner, axis));
    PyTensor { inner }
}

/// Conditional element-wise select.
///
/// `out[i] = on_true[i] if cond[i] != 0 else on_false[i]`
///
/// All three tensors must have the same shape.  Gradient flows through
/// `on_true` and `on_false`; `cond` receives zero gradient.
#[pyfunction]
pub fn where_cond(
    cond: &PyTensor,
    on_true: &PyTensor,
    on_false: &PyTensor,
    py: Python<'_>,
) -> PyTensor {
    let inner = py
        .allow_threads(|| coeus_autograd::where_cond(&cond.inner, &on_true.inner, &on_false.inner));
    PyTensor { inner }
}

/// Softmax along `dim` (numerically stable).
#[pyfunction]
pub fn softmax(input: &PyTensor, dim: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::softmax(&input.inner, dim as isize));
    PyTensor { inner }
}

// ── Constructors (additions) ──────────────────────────────────────────────────

/// Create a tensor filled with random samples from a normal distribution N(0, 1).
///
/// Uses `coeus_nn::init::normal_with_seed` with a time-derived seed.
#[pyfunction]
#[pyo3(signature = (shape, requires_grad = false))]
pub fn randn(shape: Vec<usize>, requires_grad: bool) -> PyTensor {
    use ::std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(12345);
    // Build a zero tensor then fill via normal_with_seed.
    let zeros_t = Tensor::<f64, MoiraiBackend>::zeros(shape);
    let mut v = Var::new(zeros_t, requires_grad);
    coeus_nn::init::normal_with_seed(&mut v, 0.0, 1.0, seed);
    PyTensor { inner: v }
}

/// Top-k values and indices along `dim`.
///
/// Returns `(values, indices)` where `values` is a `PyTensor` and `indices`
/// is a `PyTensor` with the original integer positions encoded as `f64`.
#[pyfunction]
#[pyo3(signature = (input, k, dim = 0))]
pub fn topk(input: &PyTensor, k: usize, dim: usize, py: Python<'_>) -> (PyTensor, PyTensor) {
    let backend = MoiraiBackend::new();
    // topk returns (Tensor<f64>, Tensor<i64>); convert i64 indices to f64 for uniform PyTensor.
    let (vals, idxs_i64) = py.allow_threads(|| coeus_ops::topk(&input.inner.tensor, k, dim, true));
    // Convert i64 index tensor → f64 tensor via element-wise cast.
    let idx_data: Vec<f64> = idxs_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let idx_f64 = Tensor::<f64, MoiraiBackend>::from_slice(idxs_i64.shape().to_vec(), &idx_data);
    (
        PyTensor {
            inner: Var::new(vals, false),
        },
        PyTensor {
            inner: Var::new(idx_f64, false),
        },
    )
}

/// Sort tensor along `dim`.
///
/// Returns `(sorted_values, argsort_indices)`.  If `descending=True`,
/// sorts in descending order.  Indices are encoded as `f64`.
#[pyfunction]
#[pyo3(signature = (input, dim = 0, descending = false))]
pub fn sort(
    input: &PyTensor,
    dim: usize,
    descending: bool,
    py: Python<'_>,
) -> (PyTensor, PyTensor) {
    let backend = MoiraiBackend::new();
    let (vals, idxs) =
        py.allow_threads(|| coeus_ops::sort(&input.inner.tensor, dim, descending, &backend));
    (
        PyTensor {
            inner: Var::new(vals, false),
        },
        PyTensor {
            inner: Var::new(idxs, false),
        },
    )
}

// ── Additional constructors ───────────────────────────────────────────────────

/// Create a tensor of zeros with the same shape as `input`.
#[pyfunction]
#[pyo3(signature = (input, requires_grad = false))]
pub fn zeros_like(input: &PyTensor, requires_grad: bool) -> PyTensor {
    PyTensor {
        inner: Var::new(
            Tensor::<f64, MoiraiBackend>::zeros(input.inner.tensor.shape().to_vec()),
            requires_grad,
        ),
    }
}

/// Create a tensor of ones with the same shape as `input`.
#[pyfunction]
#[pyo3(signature = (input, requires_grad = false))]
pub fn ones_like(input: &PyTensor, requires_grad: bool) -> PyTensor {
    PyTensor {
        inner: Var::new(
            Tensor::<f64, MoiraiBackend>::ones(input.inner.tensor.shape().to_vec()),
            requires_grad,
        ),
    }
}

/// Create a square identity matrix of size `n×n`.
#[pyfunction]
#[pyo3(signature = (n, requires_grad = false))]
pub fn eye(n: usize, requires_grad: bool) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = Tensor::<f64, MoiraiBackend>::eye_on(n, &backend);
    PyTensor {
        inner: Var::new(t, requires_grad),
    }
}

// ── Statistical ops ──────────────────────────────────────────────────────────

/// Global standard deviation over all elements.
///
/// Uses the unbiased (Bessel-corrected, N-1) formula unless `unbiased=False`.
/// Named `std_dev` in Rust to avoid conflict with the `std` crate name.
#[pyfunction]
#[pyo3(name = "std", signature = (input, unbiased = true))]
pub fn std_dev(input: &PyTensor, unbiased: bool, py: Python<'_>) -> PyResult<f64> {
    py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        let t = input.inner.tensor.to_contiguous_on(&backend);
        let xs = t.as_slice();
        if xs.is_empty() {
            return Err(PyValueError::new_err(
                "std: empty tensors have no standard deviation",
            ));
        }
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var_sum: f64 = xs.iter().map(|&x| (x - mean).powi(2)).sum();
        let denom = if unbiased && n > 1.0 { n - 1.0 } else { n };
        Ok((var_sum / denom).sqrt())
    })
}

/// Global variance over all elements.
///
/// Uses the unbiased (Bessel-corrected, N-1) formula unless `unbiased=False`.
/// Named `tensor_var` in Rust to avoid potential name conflicts.
#[pyfunction]
#[pyo3(name = "var", signature = (input, unbiased = true))]
pub fn tensor_var(input: &PyTensor, unbiased: bool, py: Python<'_>) -> PyResult<f64> {
    py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        let t = input.inner.tensor.to_contiguous_on(&backend);
        let xs = t.as_slice();
        if xs.is_empty() {
            return Err(PyValueError::new_err("var: empty tensors have no variance"));
        }
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var_sum: f64 = xs.iter().map(|&x| (x - mean).powi(2)).sum();
        let denom = if unbiased && n > 1.0 { n - 1.0 } else { n };
        Ok(var_sum / denom)
    })
}

/// L2 norm of all elements: `sqrt(sum(x^2))`.
#[pyfunction]
pub fn norm(input: &PyTensor, py: Python<'_>) -> f64 {
    py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        let t = input.inner.tensor.to_contiguous_on(&backend);
        t.as_slice().iter().map(|&x| x * x).sum::<f64>().sqrt()
    })
}

// ── Comparison / selection free functions ────────────────────────────────────

/// Element-wise equal: returns float mask tensor (1.0 = equal, 0.0 = not).
#[pyfunction]
pub fn eq(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let t = py.allow_threads(|| {
        tensor_cmp(&a.inner.tensor, &b.inner.tensor, |x, y| {
            if (x - y).abs() < f64::EPSILON * 8.0 {
                1.0
            } else {
                0.0
            }
        })
    })?;
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

/// Element-wise less-than: returns float mask tensor.
#[pyfunction]
pub fn lt(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let t = py.allow_threads(|| {
        tensor_cmp(&a.inner.tensor, &b.inner.tensor, |x, y| {
            if x < y {
                1.0
            } else {
                0.0
            }
        })
    })?;
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

/// Element-wise greater-than: returns float mask tensor.
#[pyfunction]
pub fn gt(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let t = py.allow_threads(|| {
        tensor_cmp(&a.inner.tensor, &b.inner.tensor, |x, y| {
            if x > y {
                1.0
            } else {
                0.0
            }
        })
    })?;
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

/// Conditional selection (free-function form matching `torch.where`).
///
/// `where(cond, on_true, on_false)` — all tensors must have the same shape.
/// `cond` is treated as boolean (non-zero = true).
#[pyfunction]
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn where_fn(
    cond: &PyTensor,
    on_true: &PyTensor,
    on_false: &PyTensor,
    py: Python<'_>,
) -> PyTensor {
    let inner = py
        .allow_threads(|| coeus_autograd::where_cond(&cond.inner, &on_true.inner, &on_false.inner));
    PyTensor { inner }
}
