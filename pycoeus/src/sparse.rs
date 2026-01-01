use crate::tensor::PyTensor;
use crate::tensor_error;
use backend::CpuBackend;
use dtype::float::Float32;
use pyo3::prelude::*;
use storage::{CooStorage, CsrStorage};
use tensor::Tensor;

/// CSR Sparse Tensor wrapper for Python
#[pyclass(name = "SparseCsrTensor", module = "_coeus")]
#[derive(Clone)]
pub struct PySparseCsrTensor {
    pub inner: Tensor<CpuBackend<Float32>, CsrStorage<Float32>, Float32>,
}

#[pymethods]
impl PySparseCsrTensor {
    #[new]
    fn new(
        data: Vec<f32>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: Vec<usize>,
    ) -> PyResult<Self> {
        let float_data: Vec<Float32> = data.into_iter().map(Float32).collect();
        let storage =
            CsrStorage::new(float_data, indices, indptr, &shape).map_err(|e| tensor_error!(e))?;
        let tensor = Tensor::from_storage(storage, CpuBackend::default());
        Ok(PySparseCsrTensor { inner: tensor })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Multiply CSR tensor with another CSR tensor
    fn matmul(&self, other: &PySparseCsrTensor) -> PyResult<PyCooTensor> {
        let result = self
            .inner
            .sparse_matmul(&other.inner)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyCooTensor { inner: result })
    }

    /// Convert to dense PyTensor
    fn to_dense(&self) -> PyResult<PyTensor> {
        let dense_tensor = self
            .inner
            .to_dense_generic()
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor {
            inner: dense_tensor,
        })
    }
}

/// COO Sparse Tensor wrapper for Python
#[pyclass(name = "CooTensor", module = "_coeus")]
#[derive(Clone)]
pub struct PyCooTensor {
    pub inner: Tensor<CpuBackend<Float32>, CooStorage<Float32>, Float32>,
}

#[pymethods]
impl PyCooTensor {
    #[new]
    fn new(
        data: Vec<f32>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: Vec<usize>,
    ) -> PyResult<Self> {
        let float_data: Vec<Float32> = data.into_iter().map(Float32).collect();
        let storage = CooStorage::new(float_data, row_indices, col_indices, &shape)
            .map_err(|e| tensor_error!(e))?;
        let tensor = Tensor::from_storage(storage, CpuBackend::default());
        Ok(PyCooTensor { inner: tensor })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Add two COO tensors
    fn add(&self, other: &PyCooTensor) -> PyResult<PyCooTensor> {
        let result = self
            .inner
            .sparse_add(&other.inner)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyCooTensor { inner: result })
    }

    /// Convert to dense PyTensor
    fn to_dense(&self) -> PyResult<PyTensor> {
        let dense_tensor = self
            .inner
            .to_dense_generic()
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor {
            inner: dense_tensor,
        })
    }
}
