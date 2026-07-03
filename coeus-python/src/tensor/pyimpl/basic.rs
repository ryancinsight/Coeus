// ── PyTensor constructor, properties, and basic protocol methods ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape = None, requires_grad = false))]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let tensor = coeus_tensor::Tensor::from_slice(shape, &data);
        Ok(Self {
            inner: coeus_autograd::Var::new(tensor, requires_grad),
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.tensor.shape().to_vec()
    }

    #[getter]
    fn data(&self) -> Vec<f64> {
        self.inner.tensor.to_contiguous().as_slice().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.tensor.ndim()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.grad.is_some()
    }

    fn item(&self) -> PyResult<f64> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        if numel != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "item(): tensor has {numel} elements, expected 1"
            )));
        }
        let contiguous = self.inner.tensor.to_contiguous();
        Ok(contiguous.as_slice()[0])
    }

    fn tolist(&self) -> Vec<f64> {
        self.data()
    }

    fn clone_tensor(&self) -> Self {
        self.clone()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.tensor.is_contiguous()
    }

    #[pyo3(name = "numel")]
    fn numel_method(&self) -> usize {
        self.inner.tensor.shape().iter().product()
    }

    // ── Python protocol / dunder methods ──

    fn __repr__(&self) -> String {
        let shape = self.shape();
        let requires_grad = self.inner.grad.is_some();
        let data = self.inner.tensor.to_contiguous();
        let vals = data.as_slice();
        let max_display = 8;
        let data_str = if vals.is_empty() {
            "[]".to_string()
        } else if vals.len() <= max_display {
            let formatted: Vec<String> = vals
                .iter()
                .map(|&v| {
                    if v == (v as i64) as f64 && v.abs() < 1e6 {
                        format!("{:.1}", v)
                    } else {
                        format!("{:.4}", v)
                    }
                })
                .collect();
            format!("[{}]", formatted.join(", "))
        } else {
            let first: Vec<String> = vals[..3].iter().map(|&v| format!("{:.4}", v)).collect();
            let last: Vec<String> = vals[vals.len() - 2..]
                .iter()
                .map(|&v| format!("{:.4}", v))
                .collect();
            format!("[{}, ..., {}]", first.join(", "), last.join(", "))
        };
        if requires_grad {
            format!("Tensor({data_str}, shape={shape:?}, requires_grad=True)")
        } else {
            format!("Tensor({data_str}, shape={shape:?})")
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __bool__(&self) -> PyResult<bool> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        if numel == 0 {
            return Ok(false);
        }
        if numel != 1 {
            return Ok(true);
        }
        Ok(self.inner.tensor.to_contiguous().as_slice()[0] != 0.0)
    }

    fn __float__(&self) -> PyResult<f64> {
        self.item()
    }

    fn __int__(&self) -> PyResult<i64> {
        self.item().map(|v| v as i64)
    }

    fn __len__(&self) -> PyResult<usize> {
        let shape = self.inner.tensor.shape();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__len__: tensor is 0-dimensional",
            ));
        }
        Ok(shape[0])
    }

    // ── Arithmetic operators ──

    fn __add__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::add(&self.inner, &other.inner));
        Ok(Self::from_var(inner))
    }

    fn __sub__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sub(&self.inner, &other.inner));
        Ok(Self::from_var(inner))
    }

    fn __mul__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::mul(&self.inner, &other.inner));
        Ok(Self::from_var(inner))
    }

    fn __truediv__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::div(&self.inner, &other.inner));
        Ok(Self::from_var(inner))
    }

    fn __matmul__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::matmul(&self.inner, &other.inner));
        Ok(Self::from_var(inner))
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::neg(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn __rmul__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&self.inner, scalar));
        Ok(Self::from_var(inner))
    }

    fn __radd__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_add(&self.inner, scalar));
        Ok(Self::from_var(inner))
    }
}
