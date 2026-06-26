use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn reshape(input: &PyTensor, shape: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::reshape(&input.inner, shape));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn permute(input: &PyTensor, dims: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::permute(&input.inner, &dims));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn t(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::transpose_2d(&input.inner));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn flip(input: &PyTensor, axis: usize, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::flip(&input.inner, axis));
    PyTensor::from_var(inner)
}

#[pyfunction]
pub fn unsqueeze(input: &PyTensor, dim: usize, py: Python<'_>) -> PyResult<PyTensor> {
    if dim > input.inner.tensor.ndim() {
        return Err(PyValueError::new_err(format!(
            "unsqueeze: dim {dim} out of range for rank {}",
            input.inner.tensor.ndim()
        )));
    }
    let inner = py.allow_threads(|| coeus_autograd::unsqueeze(&input.inner, dim));
    Ok(PyTensor::from_var(inner))
}

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
    Ok(PyTensor::from_var(inner))
}

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
        return Ok(PyTensor::from_var(inner));
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
    Ok(PyTensor::from_var(inner))
}

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
    Ok(PyTensor::from_var(inner))
}

#[pyfunction]
pub fn tile(input: &PyTensor, reps: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_autograd::tile(&input.inner, &reps));
    PyTensor::from_var(inner)
}

#[pyfunction]
#[pyo3(signature = (tensors, indexing = "ij"))]
pub fn meshgrid(
    tensors: Vec<pyo3::Py<PyTensor>>,
    indexing: &str,
    py: Python<'_>,
) -> PyResult<Vec<PyTensor>> {
    if !matches!(indexing, "ij" | "xy") {
        return Err(PyValueError::new_err(format!(
            "meshgrid: indexing must be \"ij\" or \"xy\", got {indexing:?}"
        )));
    }
    for (i, t) in tensors.iter().enumerate() {
        if t.bind(py).borrow().inner.tensor.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "meshgrid: tensor {i} must be 1-D, got {}-D",
                t.bind(py).borrow().inner.tensor.ndim()
            )));
        }
    }
    let rust_tensors: Vec<Tensor<f64, MoiraiBackend>> = tensors
        .iter()
        .map(|t| t.bind(py).borrow().inner.tensor.clone())
        .collect();
    let backend = MoiraiBackend::new();
    let grids = py.allow_threads(|| {
        let refs: Vec<&Tensor<f64, MoiraiBackend>> = rust_tensors.iter().collect();
        coeus_ops::meshgrid(&refs, indexing, &backend)
    });
    Ok(grids
        .into_iter()
        .map(|t| PyTensor {
            inner: coeus_autograd::Var::new(t, false),
        })
        .collect())
}

/// Broadcast a list of tensors to a common shape.
///
/// Returns a list of tensors all expanded to the same broadcastable shape.
/// Equivalent to `torch.broadcast_tensors(*tensors)`.
#[pyfunction]
pub fn broadcast_tensors(
    tensors: Vec<pyo3::Py<PyTensor>>,
    py: Python<'_>,
) -> PyResult<Vec<PyTensor>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }
    // Compute the broadcast output shape from all inputs.
    let shapes: Vec<Vec<usize>> = tensors
        .iter()
        .map(|t| t.bind(py).borrow().inner.tensor.shape().to_vec())
        .collect();
    // Fold over shapes to find the broadcast shape.
    let mut out_shape = shapes[0].clone();
    for shape in &shapes[1..] {
        let ndim_out = out_shape.len().max(shape.len());
        let mut new_shape = vec![1usize; ndim_out];
        let out_pad = ndim_out - out_shape.len();
        let shape_pad = ndim_out - shape.len();
        for i in 0..ndim_out {
            let a = if i >= out_pad {
                out_shape[i - out_pad]
            } else {
                1
            };
            let b = if i >= shape_pad {
                shape[i - shape_pad]
            } else {
                1
            };
            if a != b && a != 1 && b != 1 {
                return Err(PyValueError::new_err(format!(
                    "broadcast_tensors: shapes {:?} and {:?} are incompatible",
                    out_shape, shape
                )));
            }
            new_shape[i] = a.max(b);
        }
        out_shape = new_shape;
    }
    // Expand each tensor to the broadcast shape.
    let results = tensors
        .iter()
        .map(|t| {
            let t_ref = t.bind(py).borrow();
            let src_ndim = t_ref.inner.tensor.ndim();
            // Prepend ones to match output ndim.
            let padded_shape: Vec<usize> = (0..out_shape.len())
                .map(|i| {
                    let pad = out_shape.len() - src_ndim;
                    if i < pad {
                        1
                    } else {
                        t_ref.inner.tensor.shape()[i - pad]
                    }
                })
                .collect();
            // Expand to broadcast shape.
            let expand_shape = out_shape.clone();
            let var = if padded_shape == expand_shape {
                t_ref.inner.clone()
            } else {
                // Reshape to padded then add zeros of target shape to broadcast.
                let reshaped = coeus_autograd::reshape(&t_ref.inner, padded_shape);
                let zeros_v = coeus_autograd::Var::new(
                    coeus_tensor::Tensor::<f64, MoiraiBackend>::zeros(expand_shape),
                    false,
                );
                coeus_autograd::add(&reshaped, &zeros_v)
            };
            PyTensor::from_var(var)
        })
        .collect();
    Ok(results)
}
