use pyo3::prelude::*;
use pyo3::pyclass;
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

/// Tensor wrapper for Python
#[pyclass(name = "Tensor", module = "_coeus")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor<CpuBackend, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let backend = CpuBackend::default();
        let float_data: Vec<Float32> = data.into_iter().map(|x| Float32(x)).collect();
        let tensor = Tensor::from_vec(float_data, &shape).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tensor creation failed: {:?}", e))
        })?;
        Ok(PyTensor { inner: tensor })
    }

    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner + &other.inner;
        Ok(PyTensor { inner: result })
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner * &other.inner;
        Ok(PyTensor { inner: result })
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        self.inner = self.inner.clone().requires_grad_(requires_grad);
        Ok(())
    }

    /// Set the number of threads for CPU operations (static method)
    #[staticmethod]
    fn set_num_threads(_num_threads: usize) -> PyResult<()> {
        // TODO: Implement when backend supports it
        Ok(())
    }

    /// Get the current number of threads for CPU operations (static method)
    #[staticmethod]
    fn get_num_threads() -> PyResult<usize> {
        // TODO: Implement when backend supports it
        Ok(1)
    }
}

/// Device enum
#[pyclass(name = "Device", module = "_coeus")]
pub enum Device {
    CPU,
    CUDA,
}
