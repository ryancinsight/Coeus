//! Dtype reporting and casts.

use super::PyTensor;
use crate::error::map_backend_error;
use coeus_autograd::Var;
use eunomia::{convert::IntegerTarget, FloatElement};
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn float(&self) -> Self {
        self.clone()
    }

    fn double(&self) -> Self {
        self.clone()
    }

    fn long(&self) -> PyResult<Self> {
        let contiguous = self
            .inner
            .tensor
            .to_contiguous()
            .map_err(map_backend_error)?;
        let data: Vec<f64> = contiguous
            .as_slice()
            .iter()
            .map(|&v| f64::from_integer(i64::from_truncated(v)))
            .collect();
        let shape = self.inner.tensor.shape().to_vec();
        let t = coeus_tensor::Tensor::from_slice(shape, &data).map_err(map_backend_error)?;
        Var::new(t, false)
            .map(Self::from_var)
            .map_err(map_backend_error)
    }

    fn int(&self) -> PyResult<Self> {
        self.long()
    }

    fn half(&self) -> PyResult<Self> {
        let contiguous = self
            .inner
            .tensor
            .to_contiguous()
            .map_err(map_backend_error)?;
        let data: Vec<f64> = contiguous
            .as_slice()
            .iter()
            .map(|&v| f64::from(eunomia::F16::from_f64(v).to_f32()))
            .collect();
        let shape = self.inner.tensor.shape().to_vec();
        let t = coeus_tensor::Tensor::from_slice(shape, &data).map_err(map_backend_error)?;
        Var::new(t, false)
            .map(Self::from_var)
            .map_err(map_backend_error)
    }

    fn to(&self, dtype: &str) -> PyResult<Self> {
        match dtype {
            "float" | "float32" | "float64" | "double" => Ok(self.float()),
            "long" | "int64" => self.long(),
            "int" | "int32" => self.int(),
            "half" | "float16" => self.half(),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to: unknown dtype '{other}'; supported: float, double, long, int, half, float16, float32, float64, int32, int64"
            ))),
        }
    }

    fn type_as(&self, _other: &PyTensor) -> Self {
        self.clone()
    }
}
