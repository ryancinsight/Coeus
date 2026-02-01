use crate::tensor::{to_py_err, PyTensor, TensorWrapper};
use pyo3::prelude::*;

#[pyfunction]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    match (&input.inner, &weight.inner) {
        (TensorWrapper::CpuDenseF32(i), TensorWrapper::CpuDenseF32(w)) => {
            let b = match bias {
                Some(bt) => {
                    if let TensorWrapper::CpuDenseF32(inner_b) = &bt.inner {
                        Some(inner_b)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Bias must have same dtype as input",
                        ));
                    }
                }
                None => None,
            };
            let result = coeus_nn::functional_api::linear(i, w, b).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF32(result),
            })
        }
        (TensorWrapper::CpuDenseF64(i), TensorWrapper::CpuDenseF64(w)) => {
            let b = match bias {
                Some(bt) => {
                    if let TensorWrapper::CpuDenseF64(inner_b) = &bt.inner {
                        Some(inner_b)
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Bias must have same dtype as input",
                        ));
                    }
                }
                None => None,
            };
            let result = coeus_nn::functional_api::linear(i, w, b).map_err(to_py_err)?;
            Ok(PyTensor {
                inner: TensorWrapper::CpuDenseF64(result),
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "linear not implemented for this dtype mixture",
        )),
    }
}
