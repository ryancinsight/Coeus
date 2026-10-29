//! Constructor and the read-only tensor properties.

use super::PyTensor;
use crate::error::map_backend_error;
use coeus_autograd::Var;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape = None, requires_grad = false))]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let tensor = coeus_tensor::Tensor::from_slice(shape, &data).map_err(map_backend_error)?;
        let inner = Var::new(tensor, requires_grad).map_err(map_backend_error)?;
        Ok(Self { inner })
    }

    #[getter]
    pub(crate) fn shape(&self) -> Vec<usize> {
        self.inner.tensor.shape().to_vec()
    }

    #[getter]
    fn data(&self) -> PyResult<Vec<f64>> {
        self.inner
            .tensor
            .to_contiguous()
            .map(|tensor| tensor.as_slice().to_vec())
            .map_err(map_backend_error)
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.tensor.ndim()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.grad.is_some()
    }

    pub(crate) fn item(&self) -> PyResult<f64> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        if numel != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "item(): tensor has {numel} elements, expected 1"
            )));
        }
        let contiguous = self
            .inner
            .tensor
            .to_contiguous()
            .map_err(map_backend_error)?;
        Ok(contiguous.as_slice()[0])
    }

    fn tolist(&self) -> PyResult<Vec<f64>> {
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
}
