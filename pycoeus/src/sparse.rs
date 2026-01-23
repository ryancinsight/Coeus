//! Python bindings for sparse tensor operations.
//!
//! This module exposes sparse tensor types (CSR and COO) to Python via PyO3,
//! using TensorWrapper for dense tensor conversions.

use crate::tensor::{PyTensor, TensorWrapper};
use crate::tensor_error;
use backend::CpuBackend;
use dtype::float::Float32;
use pyo3::prelude::*;
use storage::{CooStorage, CsrStorage};
use tensor::Tensor;

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PySparseCsrTensor>()?;
    m.add_class::<PyCooTensor>()?;
    Ok(())
}

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
    fn matmul(&self, other: &PySparseCsrTensor) -> PyResult<PySparseCsrTensor> {
        // Coeus sparse_matmul on CSR returns CSR
        let result = self
            .inner
            .sparse_matmul(&other.inner)
            .map_err(|e| tensor_error!(e))?;
        Ok(PySparseCsrTensor { inner: result })
    }

    /// Convert to dense PyTensor
    fn to_dense(&self) -> PyResult<PyTensor> {
        let dense_tensor = self
            .inner
            .to_dense_generic()
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(dense_tensor),
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

    /// Add two COO tensors (converts to CSR for optimized addition)
    fn add(&self, other: &PyCooTensor) -> PyResult<PySparseCsrTensor> {
        let self_csr = self.inner.storage().to_csr().map_err(|e| tensor_error!(e))?;
        let other_csr = other.inner.storage().to_csr().map_err(|e| tensor_error!(e))?;
        
        let self_tensor = Tensor::from_storage(self_csr, self.inner.backend().clone());
        let other_tensor = Tensor::from_storage(other_csr, other.inner.backend().clone());
        
        let result = self_tensor
            .sparse_add(&other_tensor)
            .map_err(|e| tensor_error!(e))?;
        Ok(PySparseCsrTensor { inner: result })
    }

    /// Convert to dense PyTensor
    fn to_dense(&self) -> PyResult<PyTensor> {
        let dense_tensor = self
            .inner
            .to_dense_generic()
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor {
            inner: TensorWrapper::CpuDenseF32(dense_tensor),
        })
    }
}
