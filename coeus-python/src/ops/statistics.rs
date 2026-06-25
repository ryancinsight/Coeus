use crate::tensor::PyTensor;
use coeus_core::{ComputeBackend, MoiraiBackend};
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pyfunction]
pub fn norm(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::norm(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(name = "vector_norm", signature = (input, ord = 2.0, axis = None, keepdim = false))]
pub fn vector_norm(
    input: &PyTensor,
    ord: f64,
    axis: Option<usize>,
    keepdim: bool,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if !ord.is_finite() || ord <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "vector_norm: ord must be a finite positive number, got {ord}"
        )));
    }
    let n = input.inner.tensor.numel();
    if n == 0 {
        return Err(PyValueError::new_err(
            "vector_norm: empty tensors have no norm",
        ));
    }
    if let Some(ax) = axis {
        validate_stat_axis("vector_norm", input, ax)?;
        let inner = py.allow_threads(|| coeus_autograd::norm_p_axis(&input.inner, ord, ax));
        let squeezed = if keepdim {
            inner
        } else {
            let mut shape = inner.tensor.shape().to_vec();
            shape.remove(ax);
            if shape.is_empty() {
                shape = vec![1];
            }
            coeus_autograd::reshape(&inner, shape)
        };
        return Ok(PyTensor { inner: squeezed });
    }
    let inner = py.allow_threads(|| coeus_autograd::norm_p(&input.inner, ord));
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
#[pyo3(signature = (input, k, dim = 0, largest = true))]
pub fn topk(
    input: &PyTensor,
    k: usize,
    dim: usize,
    largest: bool,
    py: Python<'_>,
) -> (PyTensor, PyTensor) {
    let backend = MoiraiBackend::new();
    let (vals, idxs_i64) =
        py.allow_threads(|| coeus_ops::topk(&input.inner.tensor, k, dim, largest));
    let idx_data: Vec<f64> = idxs_i64
        .to_contiguous_on(&backend)
        .as_slice()
        .iter()
        .map(|&x| x as f64)
        .collect();
    let idx_f64 = Tensor::<f64, MoiraiBackend>::from_slice(idxs_i64.shape().to_vec(), &idx_data);
    (
        PyTensor {
            inner: coeus_autograd::Var::new(vals, false),
        },
        PyTensor {
            inner: coeus_autograd::Var::new(idx_f64, false),
        },
    )
}

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
            inner: coeus_autograd::Var::new(vals, false),
        },
        PyTensor {
            inner: coeus_autograd::Var::new(idxs, false),
        },
    )
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
                inner: coeus_autograd::Var::new(reduced, false),
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
                inner: coeus_autograd::Var::new(squeezed, false),
            },
        )?
        .into_any())
    }
}

fn scalar_object(py: Python<'_>, value: f64) -> PyResult<Py<PyAny>> {
    Ok(value.into_pyobject(py)?.unbind().into_any())
}

/// Clip the global gradient norm of a list of parameters.
///
/// Rescales gradients so that their global Lp norm does not exceed `max_norm`.
/// Returns the global norm before clipping.
///
/// Equivalent to `torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)`.
#[pyfunction]
#[pyo3(signature = (parameters, max_norm, norm_type = 2.0))]
pub fn clip_grad_norm_(
    parameters: Vec<pyo3::Py<PyTensor>>,
    max_norm: f64,
    norm_type: f64,
    py: Python<'_>,
) -> PyResult<f64> {
    if max_norm <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "clip_grad_norm_: max_norm must be positive, got {max_norm}"
        )));
    }
    // Collect all gradient slices.
    let mut grad_data: Vec<Vec<f64>> = Vec::new();
    for p in &parameters {
        let p_ref = p.bind(py).borrow();
        if let Some(g) = p_ref.inner.grad() {
            let cont = g.to_contiguous();
            grad_data.push(cont.as_slice().to_vec());
        }
    }
    if grad_data.is_empty() {
        return Ok(0.0);
    }
    // Compute global norm.
    let global_norm = if (norm_type - 2.0).abs() < 1e-9 {
        let sum_sq: f64 = grad_data
            .iter()
            .flat_map(|g| g.iter())
            .map(|&v| v * v)
            .sum();
        sum_sq.sqrt()
    } else {
        let sum_p: f64 = grad_data
            .iter()
            .flat_map(|g| g.iter())
            .map(|&v| v.abs().powf(norm_type))
            .sum();
        sum_p.powf(1.0 / norm_type)
    };
    // Scale if norm exceeds max_norm.
    if global_norm > max_norm {
        let scale = max_norm / (global_norm + 1e-6);
        for p in &parameters {
            let p_ref = p.bind(py).borrow();
            if p_ref.inner.grad.is_none() {
                continue;
            }
            let grad_buf = p_ref.inner.grad.as_ref().unwrap();
            let grad_tensor = grad_buf.write();
            let backend = MoiraiBackend::new();
            // Apply scale in-place via host round-trip.
            let numel = grad_tensor.numel();
            let mut host = vec![0.0f64; numel];
            backend.copy_to_host(grad_tensor.storage(), &mut host);
            for v in &mut host {
                *v *= scale;
            }
            backend.copy_to_device(&host, grad_tensor.storage_mut());
        }
    }
    Ok(global_norm)
}

/// Clip each gradient's values element-wise to `[-clip_value, clip_value]`.
///
/// Equivalent to `torch.nn.utils.clip_grad_value_(parameters, clip_value)`.
#[pyfunction]
pub fn clip_grad_value_(
    parameters: Vec<pyo3::Py<PyTensor>>,
    clip_value: f64,
    py: Python<'_>,
) -> PyResult<()> {
    if clip_value <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "clip_grad_value_: clip_value must be positive, got {clip_value}"
        )));
    }
    for p in &parameters {
        let p_ref = p.bind(py).borrow();
        if p_ref.inner.grad.is_none() {
            continue;
        }
        let grad_buf = p_ref.inner.grad.as_ref().unwrap();
        let grad_tensor = grad_buf.write();
        let backend = MoiraiBackend::new();
        // Clamp in-place via copy-to-host + clamp + copy-back.
        let numel = grad_tensor.numel();
        let mut host = vec![0.0f64; numel];
        backend.copy_to_host(grad_tensor.storage(), &mut host);
        for v in &mut host {
            *v = v.clamp(-clip_value, clip_value);
        }
        backend.copy_to_device(&host, grad_tensor.storage_mut());
    }
    Ok(())
}
