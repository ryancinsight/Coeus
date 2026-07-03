// ── PyTensor dtype cast operations ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
    fn float(&self) -> Self {
        self.clone()
    }

    fn double(&self) -> Self {
        self.clone()
    }

    fn long(&self) -> Self {
        let data: Vec<f64> = self
            .inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|&v| (v as i64) as f64)
            .collect();
        let shape = self.inner.tensor.shape().to_vec();
        let t = coeus_tensor::Tensor::from_slice(shape, &data);
        Self {
            inner: coeus_autograd::Var::new(t, false),
        }
    }

    fn int(&self) -> Self {
        self.long()
    }

    fn half(&self) -> Self {
        let data: Vec<f64> = self
            .inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|&v| f64::from(half::f16::from_f64(v)))
            .collect();
        let shape = self.inner.tensor.shape().to_vec();
        let t = coeus_tensor::Tensor::from_slice(shape, &data);
        Self {
            inner: coeus_autograd::Var::new(t, false),
        }
    }

    fn to(&self, dtype: &str) -> PyResult<Self> {
        match dtype {
            "float" | "float32" | "float64" | "double" => Ok(self.float()),
            "long" | "int64" => Ok(self.long()),
            "int" | "int32" => Ok(self.int()),
            "half" | "float16" => Ok(self.half()),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to: unknown dtype '{other}'; supported: float, double, long, int, half, float16, float32, float64, int32, int64"
            ))),
        }
    }

    fn type_as(&self, _other: &PyTensor) -> Self {
        self.clone()
    }
}
