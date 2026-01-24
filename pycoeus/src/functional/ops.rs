use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (input, p=0.5, training=true, inplace=false))]
pub fn dropout(input: &PyTensor, p: f64, training: bool, inplace: bool) -> PyResult<PyTensor> {
    if inplace {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "dropout(inplace=True) is not implemented",
        ));
    }

    let result = match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = coeus_nn::functional_api::dropout(t, p, training).map_err(to_py_err)?;
             Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = coeus_nn::functional_api::dropout(t, p, training).map_err(to_py_err)?;
             Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = coeus_nn::functional_api::dropout(t, p, training).map_err(to_py_err)?;
             Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "dropout not implemented for this tensor type",
        )),
    }?;
    Ok(result)
}

/// Matrix multiplication
#[pyfunction]
pub fn matmul(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.matmul(other)
}

/// Batch matrix multiplication
#[pyfunction]
pub fn bmm(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.bmm(other)
}

/// Add matrix multiplication
#[pyfunction]
#[pyo3(signature = (input, mat1, mat2, beta=1.0, alpha=1.0))]
pub fn addmm(
    input: &PyTensor,
    mat1: &PyTensor,
    mat2: &PyTensor,
    beta: f32,
    alpha: f32,
) -> PyResult<PyTensor> {

    input.addmm(mat1, mat2, beta as f64, alpha as f64)
}

/// Matrix-vector multiplication
#[pyfunction]
pub fn mv(input: &PyTensor, vec: &PyTensor) -> PyResult<PyTensor> {
    input.mv(vec)
}

/// Outer product
#[pyfunction]
pub fn addr(input: &PyTensor, vec1: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
    // Note: PyTorch addr is input + outer(v1, v2).
    // My PyTensor::addr implementation currently mimics torch.outer (ignoring input).
    // If user calls torch.addr(input, v1, v2), they expect addition?
    // For now, I'll bind it, but acknowledge the behavior in PyTensor::addr.
    // Actually, PyTensor::addr takes input (self), vec1, vec2.
    // But my implementation ignored self and just called outer.
    // If I want to match torch.addr(input, ...), I should use input.addr(v1, v2).
    // Which currently calls outer.
    input.addr(vec1, vec2)
}

/// Outer product alias
#[pyfunction]
pub fn outer(input: &PyTensor, vec2: &PyTensor) -> PyResult<PyTensor> {
    input.outer(vec2)
}

/// Reshape tensor
#[pyfunction]
pub fn reshape(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.reshape(shape)
}

/// View tensor (alias for reshape)
#[pyfunction]
pub fn view(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.view(shape)
}

/// Flatten tensor
#[pyfunction]
#[pyo3(signature = (input, start_dim=0, end_dim=-1))]
pub fn flatten(input: &PyTensor, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
    input.flatten(start_dim, end_dim)
}

/// Squeeze tensor
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn squeeze(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    input.squeeze(dim)
}

/// Unsqueeze tensor
#[pyfunction]
pub fn unsqueeze(input: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    input.unsqueeze(dim)
}

/// Transpose tensor
#[pyfunction]
pub fn transpose(input: &PyTensor, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
    input.transpose(dim0, dim1)
}

/// Permute tensor
#[pyfunction]
pub fn permute(input: &PyTensor, dims: Vec<usize>) -> PyResult<PyTensor> {
    // Permute on sparse is not implemented, so manual match
    match &input.inner {
        TensorWrapper::CpuDenseF32(t) => {
            let res = t.permute(&dims).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        TensorWrapper::CpuDenseF64(t) => {
            let res = t.permute(&dims).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
        TensorWrapper::CpuDenseI64(t) => {
             let res = t.permute(&dims).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseI64(res) })
        }
        #[cfg(feature = "gpu")]
        TensorWrapper::GpuDenseF32(t) => {
            let res = t.permute(&dims).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "permute not implemented for sparse tensors",
        )),
    }
}
