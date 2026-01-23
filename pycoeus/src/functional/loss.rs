use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use dtype::float::Float32;
use pyo3::prelude::*;

#[pyfunction]
pub fn mse_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
            let res = coeus_nn::functional_api::mse_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
            let res = coeus_nn::functional_api::mse_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
        #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
             let res = coeus_nn::functional_api::mse_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "mse_loss not implemented for these types (requires float)",
        )),
    }
}

#[pyfunction]
pub fn l1_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
            let res = coeus_nn::functional_api::l1_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
            let res = coeus_nn::functional_api::l1_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
        #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
             let res = coeus_nn::functional_api::l1_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "l1_loss not implemented for these types (requires float)",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, target, beta=1.0))]
pub fn smooth_l1_loss(input: &PyTensor, target: &PyTensor, beta: f32) -> PyResult<PyTensor> {
    match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
            let res = coeus_nn::functional_api::smooth_l1_loss(a, b, Float32::new(beta))
                .map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(res),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "smooth_l1_loss only implemented for F32",
        )),
    }
}

#[pyfunction]
pub fn binary_cross_entropy(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
            let res = coeus_nn::functional_api::binary_cross_entropy(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
            let res = coeus_nn::functional_api::binary_cross_entropy(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
         #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
             let res = coeus_nn::functional_api::binary_cross_entropy(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "binary_cross_entropy not implemented for these types (requires float)",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0))]
pub fn cross_entropy(
    input: &PyTensor,
    target: &PyTensor,
    weight: Option<&Bound<'_, PyAny>>,
    ignore_index: i64,
    reduction: &str,
    label_smoothing: f64,
) -> PyResult<PyTensor> {
    if weight.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(weight=...) is not implemented",
        ));
    }

    if ignore_index != -100 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(ignore_index!= -100) is not implemented",
        ));
    }

    if reduction != "mean" {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(reduction!= 'mean') is not implemented",
        ));
    }

    if label_smoothing != 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(label_smoothing!=0.0) is not implemented",
        ));
    }

    let result = match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(i), TensorWrapper::CpuDenseF32(t)) => {
            TensorWrapper::CpuDenseF32(
                coeus_nn::functional_api::cross_entropy(i, t).map_err(to_py_err)?,
            )
        }
        (TensorWrapper::CpuDenseF64(i), TensorWrapper::CpuDenseF64(t)) => {
            TensorWrapper::CpuDenseF64(
                coeus_nn::functional_api::cross_entropy(i, t).map_err(to_py_err)?,
            )
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "cross_entropy not implemented for this dtype",
            ))
        }
    };
    PyTensor { inner: result }.squeeze(None)
}

/// Binary cross-entropy with logits loss
#[pyfunction]
pub fn bce_with_logits_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
            let res = coeus_nn::functional_api::bce_with_logits_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
            let res = coeus_nn::functional_api::bce_with_logits_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
         #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
             let res = coeus_nn::functional_api::bce_with_logits_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "bce_with_logits_loss not implemented for these types (requires float)",
        )),
    }?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, ignore_index=-100, reduction="mean"))]
pub fn nll_loss(
    input: &PyTensor,
    target: &PyTensor,
    weight: Option<&Bound<'_, PyAny>>,
    ignore_index: i64,
    reduction: &str,
) -> PyResult<PyTensor> {
    if weight.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss(weight=...) is not implemented",
        ));
    }

    if ignore_index != -100 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss(ignore_index!= -100) is not implemented",
        ));
    }

    if reduction != "mean" {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss(reduction!= 'mean') is not implemented",
        ));
    }

    let result = match (&input.inner, &target.inner) {
        (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
            let res = coeus_nn::functional_api::nll_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
        }
        (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
            let res = coeus_nn::functional_api::nll_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
        }
         #[cfg(feature = "gpu")]
        (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
             let res = coeus_nn::functional_api::nll_loss(a, b).map_err(to_py_err)?;
            Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss not implemented for these types (requires same float dtype)",
        )),
    }?;
    result.squeeze(None)
}
