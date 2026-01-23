use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::dispatch_tensor_mut;
use pyo3::prelude::*;
use dtype::float::{Float32, Float64};

#[pymethods]
impl PyTensor {
    pub fn zero_(&mut self) -> PyResult<()> {
        dispatch_tensor_mut!(self, inner => inner.zero_());
        Ok(())
    }

    pub fn fill_(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                let val = Float32(value as f32);
                for x in inner.as_mut_slice() {
                    *x = val;
                }
            }
            TensorWrapper::CpuDenseF64(inner) => {
                let val = Float64(value);
                for x in inner.as_mut_slice() {
                    *x = val;
                }
            }
            TensorWrapper::CpuDenseI64(inner) => {
                let val = dtype::int::Int64(value as i64);
                for x in inner.as_mut_slice() {
                    *x = val;
                }
            }
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "GPU fill_ not implemented",
                ))
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("fill_ not implemented for sparse"))
        }
        Ok(())
    }

    #[pyo3(name = "add_")]
    pub fn add_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                for x in inner.as_mut_slice() {
                    *x = Float32(x.get() + value as f32);
                }
            }
            TensorWrapper::CpuDenseF64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = Float64(x.get() + value);
                }
            }
            TensorWrapper::CpuDenseI64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = dtype::int::Int64(x.get() + value as i64);
                }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("add_ not implemented for sparse"))
        }
        Ok(())
    }

    #[pyo3(name = "mul_")]
    pub fn mul_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                inner.mul_scalar_(Float32(value as f32)).map_err(to_py_err)?;
            }
            TensorWrapper::CpuDenseF64(inner) => {
                inner.mul_scalar_(Float64(value)).map_err(to_py_err)?;
            }
            TensorWrapper::CpuDenseI64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = dtype::int::Int64(x.get() * value as i64);
                }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("mul_ not implemented for sparse"))
        }
        Ok(())
    }

    #[pyo3(name = "sub_")]
    pub fn sub_inplace(&mut self, value: f64) -> PyResult<()> {
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                for x in inner.as_mut_slice() {
                    *x = Float32(x.get() - value as f32);
                }
            }
            TensorWrapper::CpuDenseF64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = Float64(x.get() - value);
                }
            }
            TensorWrapper::CpuDenseI64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = dtype::int::Int64(x.get() - value as i64);
                }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("sub_ not implemented for sparse"))
        }
        Ok(())
    }

    #[pyo3(name = "div_")]
    pub fn div_inplace(&mut self, value: f64) -> PyResult<()> {
        if value == 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyZeroDivisionError, _>("division by zero"));
        }
        match &mut self.inner {
            TensorWrapper::CpuDenseF32(inner) => {
                for x in inner.as_mut_slice() {
                    *x = Float32(x.get() / value as f32);
                }
            }
            TensorWrapper::CpuDenseF64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = Float64(x.get() / value);
                }
            }
            TensorWrapper::CpuDenseI64(inner) => {
                for x in inner.as_mut_slice() {
                    *x = dtype::int::Int64(x.get() / value as i64);
                }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("div_ not implemented for sparse"))
        }
        Ok(())
    }
}

