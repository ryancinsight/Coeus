use crate::tensor::PyTensor;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Element-wise comparison: calls `f(a_elem, b_elem)` for each pair of elements
/// from contiguous CPU tensors of the same shape, returns a mask tensor.
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
        inner: coeus_autograd::Var::new(t, false),
    })
}

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
        inner: coeus_autograd::Var::new(t, false),
    })
}

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
        inner: coeus_autograd::Var::new(t, false),
    })
}

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
        inner: coeus_autograd::Var::new(t, false),
    })
}

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
        inner: coeus_autograd::Var::new(t, false),
    })
}

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
        inner: coeus_autograd::Var::new(t, false),
    })
}

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
    PyTensor::from_var(inner)
}
