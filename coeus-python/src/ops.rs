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

/// Element-wise reciprocal: 1/x.
#[pyfunction]
pub fn recip(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::recip(&input.inner));
    PyTensor { inner }
}

/// Element-wise signum: -1, 0, or 1.
#[pyfunction]
pub fn sign(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::sign(&input.inner));
    PyTensor { inner }
}

/// Element-wise floor.
#[pyfunction]
pub fn floor(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::floor(&input.inner));
    PyTensor { inner }
}

/// Element-wise ceil.
#[pyfunction]
pub fn ceil(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::ceil(&input.inner));
    PyTensor { inner }
}

/// Element-wise round to nearest integer.
#[pyfunction]
pub fn round(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::round(&input.inner));
    PyTensor { inner }
}

/// Element-wise truncation toward zero.
#[pyfunction]
pub fn trunc(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::trunc(&input.inner));
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

// ── Statistical ops ─────────────────────────────────────────────

/// Global standard deviation over all elements.
///
/// Uses the unbiased (Bessel-corrected, N-1) formula unless `unbiased=False`
/// — matches PyTorch's `torch.std` and JAX's `jnp.std` defaults. Pass
/// `axis` and `keepdim` for `torch.std(x, dim=…)`-style behavior.
///
/// Named `std_dev` in Rust to avoid conflict with the `std` crate name.
#[pyfunction]
#[pyo3(name = "std", signature = (input, unbiased = true, axis = None, keepdim = false))]
pub fn std_dev(
    input: &PyTensor,
    unbiased: bool,
    axis: Option<usize>,
    keepdim: bool,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let backend = MoiraiBackend::new();
    if input.inner.tensor.numel() == 0 {
        return Err(PyValueError::new_err(
            "std: empty tensors have no standard deviation",
        ));
    }
    if let Some(ax) = axis {
        validate_stat_axis("std", input, ax)?;
        let reduced = py
            .allow_threads(|| coeus_ops::std_dev_axis(&input.inner.tensor, ax, unbiased, &backend));
        return tensor_or_scalar_reduction(py, reduced, ax, keepdim);
    }
    let v = py.allow_threads(|| {
        coeus_ops::std_dev::<f64, MoiraiBackend>(&input.inner.tensor, unbiased, &backend)
    });
    scalar_object(py, v)
}

/// Global variance over all elements.
///
/// Uses the unbiased (Bessel-corrected, N-1) formula unless `unbiased=False`
/// — matches PyTorch's `torch.var` and JAX's `jnp.var` defaults.
///
/// Named `tensor_var` in Rust to avoid conflict with `Var<T,B>` autograd
/// wrapper naming.
#[pyfunction]
#[pyo3(name = "var", signature = (input, unbiased = true, axis = None, keepdim = false))]
pub fn tensor_var(
    input: &PyTensor,
    unbiased: bool,
    axis: Option<usize>,
    keepdim: bool,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let backend = MoiraiBackend::new();
    if input.inner.tensor.numel() == 0 {
        return Err(PyValueError::new_err("var: empty tensors have no variance"));
    }
    if let Some(ax) = axis {
        validate_stat_axis("var", input, ax)?;
        let reduced =
            py.allow_threads(|| coeus_ops::var_axis(&input.inner.tensor, ax, unbiased, &backend));
        return tensor_or_scalar_reduction(py, reduced, ax, keepdim);
    }
    let v = py.allow_threads(|| {
        coeus_ops::var::<f64, MoiraiBackend>(&input.inner.tensor, unbiased, &backend)
    });
    scalar_object(py, v)
}

/// L2 (Euclidean) norm of all elements: `sqrt(sum(x^2))`.
///
/// Matches `torch.linalg.vector_norm(x, ord=2)` with no shape argument. Use
/// [`vector_norm`] for ord-p variants (`ord=1` Manhattan, `ord=3`, ...).
#[pyfunction]
pub fn norm(input: &PyTensor, py: Python<'_>) -> f64 {
    py.allow_threads(|| {
        let backend = MoiraiBackend::new();
        coeus_ops::norm::<f64, MoiraiBackend>(&input.inner.tensor, &backend)
    })
}

/// `L_p` norm: `(Σ|xᵢ|^p)^(1/p)`.
///
/// Matches `torch.linalg.vector_norm(input, ord=p, dim=..., keepdim=...)`.
/// `ord` must be a finite positive number. Empty tensors surface as
/// `ValueError` (the Rust-core call panics on empty input rather than
/// silently returning zero).
#[pyfunction]
#[pyo3(name = "vector_norm", signature = (input, ord = 2.0, axis = None, keepdim = false))]
pub fn vector_norm(
    input: &PyTensor,
    ord: f64,
    axis: Option<usize>,
    keepdim: bool,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    // `keepdim` is part of the public torch.linalg.vector_norm signature;
    // the per-axis form is not yet implemented in Rust-core, so the
    // parameter is currently a no-op reserved for symmetry. Documented as
    // an "axis not supported" error below so the parameter is never
    // silently ignored.
    let _ = keepdim;
    if !ord.is_finite() || ord <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "vector_norm: ord must be a finite positive number, got {ord}"
        )));
    }
    let backend = MoiraiBackend::new();
    if input.inner.tensor.numel() == 0 {
        return Err(PyValueError::new_err(
            "vector_norm: empty tensors have no norm",
        ));
    }
    if let Some(ax) = axis {
        validate_stat_axis("vector_norm", input, ax)?;
        // Per-axis ord-p norm: fold per-slice on the CPU backend. The current
        // Rust-core surface only exposes a global `norm_p`; the per-axis form
        // composes via per-slice reshape until the `norm_p_axis` SSOT lands.
        return Err(PyValueError::new_err(
            "vector_norm: axis/keepdim not yet supported; pass ord without axis for a global Lp norm",
        ));
    }
    let v = py.allow_threads(|| {
        coeus_ops::norm_p::<f64, MoiraiBackend>(&input.inner.tensor, ord, &backend)
    });
    scalar_object(py, v)
}

fn validate_stat_axis(op: &str, input: &PyTensor, axis: usize) -> PyResult<()> {
    let shape = input.inner.tensor.shape();
    if axis >= shape.len() {
        return Err(PyValueError::new_err(format!(
            "{op}: axis {axis} out of range for rank {}",
            shape.len()
        )));
    }
    if shape[axis] == 0 {
        return Err(PyValueError::new_err(format!(
            "{op}: axis {axis} has zero elements"
        )));
    }
    Ok(())
}

fn tensor_or_scalar_reduction(
    py: Python<'_>,
    reduced: Tensor<f64, MoiraiBackend>,
    axis: usize,
    keepdim: bool,
) -> PyResult<Py<PyAny>> {
    if keepdim {
        return Ok(Py::new(
            py,
            PyTensor {
                inner: Var::new(reduced, false),
            },
        )?
        .into_any());
    }

    let mut shape = reduced.shape().to_vec();
    shape.remove(axis);
    if shape.is_empty() {
        let backend = MoiraiBackend::new();
        let value = reduced.to_contiguous_on(&backend).as_slice()[0];
        scalar_object(py, value)
    } else {
        let squeezed = reduced.reshape(shape);
        Ok(Py::new(
            py,
            PyTensor {
                inner: Var::new(squeezed, false),
            },
        )?
        .into_any())
    }
}

fn scalar_object(py: Python<'_>, value: f64) -> PyResult<Py<PyAny>> {
    Ok(value.into_pyobject(py)?.unbind().into_any())
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

/// Element-wise greater-or-equal: returns float mask tensor.
#[pyfunction]
pub fn ge(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let t = py.allow_threads(|| {
        tensor_cmp(&a.inner.tensor, &b.inner.tensor, |x, y| {
            if x >= y {
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

/// Element-wise less-or-equal: returns float mask tensor.
#[pyfunction]
pub fn le(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let t = py.allow_threads(|| {
        tensor_cmp(&a.inner.tensor, &b.inner.tensor, |x, y| {
            if x <= y {
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

/// Element-wise not-equal: returns float mask tensor.
#[pyfunction]
pub fn ne(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    let t = py.allow_threads(|| {
        tensor_cmp(&a.inner.tensor, &b.inner.tensor, |x, y| {
            if (x - y).abs() < f64::EPSILON * 8.0 {
                0.0
            } else {
                1.0
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

// ── Indexing ops ─────────────────────────────────────────────────────────────

/// Index-based element selection along `dim`.
///
/// `out[…, k, …] = input[…, index[…, k, …], …]`
/// where `index` contains non-negative integer values (as f64).
///
/// Backward: gradient flows back to `input` via `scatter_add`.
#[pyfunction]
pub fn gather(input: &PyTensor, dim: usize, index: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::gather(&input.inner, dim, &index.inner));
    PyTensor { inner }
}

/// Scatter-accumulate: `out = input` then `out[…, index[…,k,…], …] += src[…,k,…]`.
///
/// Returns a new tensor (does not mutate `input`). Non-differentiable.
#[pyfunction]
pub fn scatter_add(
    input: &PyTensor,
    dim: usize,
    index: &PyTensor,
    src: &PyTensor,
    py: Python<'_>,
) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = py.allow_threads(|| {
        coeus_ops::scatter_add(
            &input.inner.tensor,
            dim,
            &index.inner.tensor,
            &src.inner.tensor,
            &backend,
        )
    });
    PyTensor {
        inner: Var::new(t, false),
    }
}

/// Repeat each element along `dim` exactly `repeats` times (interleaved).
///
/// Matches `torch.repeat_interleave(input, repeats, dim)`.
#[pyfunction]
pub fn repeat_interleave(input: &PyTensor, repeats: usize, dim: usize, py: Python<'_>) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = py.allow_threads(|| {
        coeus_ops::repeat_interleave(&input.inner.tensor, repeats, dim, &backend)
    });
    PyTensor {
        inner: Var::new(t, false),
    }
}

// ── Shape extras ─────────────────────────────────────────────────────────────

/// Insert a size-1 dimension at `dim`.
///
/// Equivalent to `torch.unsqueeze(input, dim)` / `jnp.expand_dims(input, dim)`.
#[pyfunction]
pub fn unsqueeze(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim > input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "unsqueeze: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::unsqueeze(&input.inner, dim));
    Ok(PyTensor { inner })
}

/// Remove size-1 dimensions.
///
/// If `dim` is given, squeeze only that axis (must have size 1); otherwise
/// remove all size-1 dimensions (matches `torch.squeeze`/`jnp.squeeze`).
#[pyfunction]
#[pyo3(signature = (input, dim = None))]
pub fn squeeze(input: &PyTensor, dim: Option<usize>, py: Python<'_>) -> PyResult<PyTensor> {
    if let Some(axis) = dim {
        let shape = input.inner.tensor.shape();
        if axis >= shape.len() {
            return Err(PyValueError::new_err(format!(
                "squeeze: dim {axis} out of range for rank {}",
                shape.len()
            )));
        }
        if shape[axis] != 1 {
            return Err(PyValueError::new_err(format!(
                "squeeze: dim {axis} has extent {}, expected 1",
                shape[axis]
            )));
        }
    }
    let inner = py.allow_threads(|| coeus_autograd::squeeze(&input.inner, dim));
    Ok(PyTensor { inner })
}

/// Flatten contiguous dimensions `[start_dim, end_dim]` into one.
///
/// Equivalent to `torch.flatten(input, start_dim, end_dim)`.
/// Negative indices are not currently supported; use Python-side arithmetic
/// (`ndim + dim`) if needed.
#[pyfunction]
#[pyo3(signature = (input, start_dim = 0, end_dim = None))]
pub fn flatten(
    input: &PyTensor,
    start_dim: usize,
    end_dim: Option<usize>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let shape = input.inner.tensor.shape().to_vec();
    let ndim = shape.len();
    if ndim == 0 {
        let inner = py.allow_threads(|| coeus_autograd::reshape(&input.inner, vec![1]));
        return Ok(PyTensor { inner });
    }
    if start_dim >= ndim {
        return Err(PyValueError::new_err(format!(
            "flatten: start_dim {start_dim} out of range for rank {ndim}"
        )));
    }
    let end = end_dim.unwrap_or(ndim - 1);
    if end >= ndim {
        return Err(PyValueError::new_err(format!(
            "flatten: end_dim {end} out of range for rank {ndim}"
        )));
    }
    if end < start_dim {
        return Err(PyValueError::new_err(format!(
            "flatten: end_dim {end} precedes start_dim {start_dim}"
        )));
    }
    let flat: usize = shape[start_dim..=end].iter().product();
    let mut new_shape: Vec<usize> = shape[..start_dim].to_vec();
    new_shape.push(flat);
    new_shape.extend_from_slice(&shape[end + 1..]);
    let inner = py.allow_threads(move || coeus_autograd::reshape(&input.inner, new_shape));
    Ok(PyTensor { inner })
}

// ── Selection / argmax / argmin ───────────────────────────────────────────────

/// Return index of the maximum value along `dim` (keep-dim = True).
///
/// Result is a `f64` tensor of indices (no autograd; indices are integers).
/// Matches `torch.argmax(input, dim, keepdim=True)`.
#[pyfunction]
pub fn argmax(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "argmax: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let backend = MoiraiBackend::new();
    let idx_i64 =
        py.allow_threads(|| coeus_ops::argmax::<f64, MoiraiBackend>(&input.inner.tensor, dim));
    let data: Vec<f64> = idx_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let t = Tensor::<f64, MoiraiBackend>::from_slice(idx_i64.shape().to_vec(), &data);
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

/// Return index of the minimum value along `dim` (keep-dim = True).
///
/// Matches `torch.argmin(input, dim, keepdim=True)`.
#[pyfunction]
pub fn argmin(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim >= input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "argmin: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let backend = MoiraiBackend::new();
    let idx_i64 =
        py.allow_threads(|| coeus_ops::argmin::<f64, MoiraiBackend>(&input.inner.tensor, dim));
    let data: Vec<f64> = idx_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let t = Tensor::<f64, MoiraiBackend>::from_slice(idx_i64.shape().to_vec(), &data);
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

// ── broadcast_to / masked_fill / nonzero ─────────────────────────────────────

/// Materialize `input` into `target_shape` by repeating along singleton dims.
///
/// `target_shape` must have the same rank as `input`; each dimension must
/// either match or the input dimension must be 1 (broadcast from size 1).
/// Backward: sum the output gradient over all broadcast dimensions.
///
/// Matches `torch.broadcast_to(input, shape)` / `jnp.broadcast_to(input, shape)`.
#[pyfunction]
pub fn broadcast_to(
    input: &PyTensor,
    target_shape: Vec<usize>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if target_shape.len() != input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "broadcast_to: target rank {} must match input rank {}",
            target_shape.len(),
            input.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::broadcast_to(&input.inner, target_shape));
    Ok(PyTensor { inner })
}

/// Replace elements of `input` with `value` wherever `mask` is non-zero (tracked).
///
/// `mask` must have the same shape as `input`.
/// Backward: gradient is zero at masked positions (mask zeroed by same pattern).
///
/// Matches `torch.masked_fill(input, mask.bool(), value)`.
#[pyfunction]
pub fn masked_fill(
    input: &PyTensor,
    mask: &PyTensor,
    value: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if input.inner.tensor.shape() != mask.inner.tensor.shape() {
        return Err(PyValueError::new_err(format!(
            "masked_fill: input shape {:?} must match mask shape {:?}",
            input.inner.tensor.shape(),
            mask.inner.tensor.shape()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::masked_fill(&input.inner, &mask.inner, value));
    Ok(PyTensor { inner })
}

/// Return the ND coordinates of all non-zero elements as a `[N, ndim]` tensor.
///
/// Each row is the coordinate of one non-zero element in row-major order.
/// Returns an empty `[0, ndim]` tensor when all elements are zero.
/// Non-differentiable (indices are not differentiable).
///
/// Matches `torch.nonzero(input, as_tuple=False)`.
#[pyfunction]
pub fn nonzero(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let backend = MoiraiBackend::new();
    let t = py
        .allow_threads(|| coeus_ops::nonzero::<f64, MoiraiBackend>(&input.inner.tensor, &backend));
    PyTensor {
        inner: Var::new(t, false),
    }
}

// ── Triangular masking / roll ────────────────────────────────────────────────

/// Lower-triangular mask on the last two dimensions.
///
/// Elements above diagonal `k` are zeroed. Backward masks the gradient with
/// the same triangular mask (tracked).
/// Matches `torch.tril(input, diagonal=k)`.
#[pyfunction]
#[pyo3(signature = (input, k = 0))]
pub fn tril(input: &PyTensor, k: i64, py: Python<'_>) -> PyResult<PyTensor> {
    let ndim = input.inner.tensor.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err(format!(
            "tril: requires at least 2-D input, got {ndim}-D"
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::tril(&input.inner, k as isize));
    Ok(PyTensor { inner })
}

/// Upper-triangular mask on the last two dimensions.
///
/// Elements below diagonal `k` are zeroed (tracked).
/// Matches `torch.triu(input, diagonal=k)`.
#[pyfunction]
#[pyo3(signature = (input, k = 0))]
pub fn triu(input: &PyTensor, k: i64, py: Python<'_>) -> PyResult<PyTensor> {
    let ndim = input.inner.tensor.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err(format!(
            "triu: requires at least 2-D input, got {ndim}-D"
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::triu(&input.inner, k as isize));
    Ok(PyTensor { inner })
}

/// Circular shift along `dims` by `shifts` (tracked).
///
/// Matches `torch.roll(input, shifts, dims)`.
#[pyfunction]
pub fn roll(
    input: &PyTensor,
    shifts: Vec<i64>,
    dims: Vec<usize>,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if shifts.len() != dims.len() {
        return Err(PyValueError::new_err(format!(
            "roll: shifts ({}) and dims ({}) must have equal length",
            shifts.len(),
            dims.len()
        )));
    }
    let ndim = input.inner.tensor.ndim();
    for &d in &dims {
        if d >= ndim {
            return Err(PyValueError::new_err(format!(
                "roll: dim {d} out of range for {ndim}-D tensor"
            )));
        }
    }
    let shifts_isize: Vec<isize> = shifts.iter().map(|&s| s as isize).collect();
    let inner = py.allow_threads(move || coeus_autograd::roll(&input.inner, &shifts_isize, &dims));
    Ok(PyTensor { inner })
}

// ── Spatial resize ────────────────────────────────────────────────────────────

/// Resize a spatial tensor.
///
/// Input shape:
/// - 1-D: `[N, C, L]`    → `size = [new_L]`
/// - 2-D: `[N, C, H, W]` → `size = [new_H, new_W]`
///
/// `mode` must be one of `"nearest"` or `"bilinear"`.
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
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "interpolate: expected 3-D or 4-D input, got {ndim}-D"
            )));
        }
    };
    Ok(PyTensor {
        inner: Var::new(t, false),
    })
}

// ── Functional nn ops (stateless) ────────────────────────────────────────────

/// Functional linear transform: `output = input @ weight.T + bias`.
///
/// Matches `torch.nn.functional.linear(input, weight, bias)`.
/// - `input`: `[..., in_features]`
/// - `weight`: `[out_features, in_features]`
/// - `bias`: optional `[out_features]`
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
    PyTensor { inner }
}

/// Functional layer normalization over the last `norm_shape` elements.
///
/// Matches `torch.nn.functional.layer_norm(input, norm_shape, weight, bias, eps)`.
/// - `weight`: optional scale `[norm_shape]`.
/// - `bias`: optional shift `[norm_shape]`.
/// - `eps`: stability epsilon (default `1e-5`).
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
    let inner = py.allow_threads(move || {
        use coeus_nn::normalization::LayerNorm;
        use coeus_nn::Module;
        let backend = coeus_core::MoiraiBackend::new();
        let weight_var = w.unwrap_or_else(|| {
            coeus_autograd::Var::new(coeus_tensor::Tensor::ones_on([norm_shape], &backend), false)
        });
        let bias_var = b.unwrap_or_else(|| {
            coeus_autograd::Var::new(
                coeus_tensor::Tensor::zeros_on([norm_shape], &backend),
                false,
            )
        });
        let ln = LayerNorm::from_parts(weight_var, bias_var, eps);
        ln.forward(&x)
    });
    PyTensor { inner }
}

/// Functional dropout: zero each element with probability `p` during training.
///
/// Matches `torch.nn.functional.dropout(input, p, training)`.
/// When `training=False` (the default), returns the input unchanged.
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
    PyTensor { inner }
}
