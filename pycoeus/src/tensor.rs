use coeus_autograd::ops::backward;
use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use pyo3::prelude::*;
use pyo3::pyclass;
use numpy;
use numpy::PyArrayMethods;

// Import the new error handling macros (exported at crate root)
use crate::tensor_error;

/// Tensor wrapper for Python
#[pyclass(name = "Tensor", module = "_coeus")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let _backend: CpuBackend<Float32> = CpuBackend::default();
        let float_data: Vec<Float32> = data.into_iter().map(|x| Float32(x)).collect();
        let tensor = Tensor::from_vec(float_data, &shape).map_err(|e| {
            tensor_error!(e)
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

    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.matmul(&other.inner).map_err(|e| {
            tensor_error!(e)
        })?;
        Ok(PyTensor { inner: result })
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        let result = self.inner.transpose(dim0, dim1).map_err(|e| {
            tensor_error!(e)
        })?;
        Ok(PyTensor { inner: result })
    }

    fn reshape(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        let result = self.inner.reshape(&shape).map_err(|e| {
            tensor_error!(e)
        })?;
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

    fn backward(&self) -> PyResult<()> {
        backward(&self.inner).map_err(|e| {
            tensor_error!(e)
        })?;
        Ok(())
    }

    /// Set the number of threads for CPU operations (static method)
    /// Note: Threading control is not yet implemented in the backend.
    /// This is a placeholder that will be activated when CPU threading
    /// support is added to the tensor backend.
    #[staticmethod]
    fn set_num_threads(_num_threads: usize) -> PyResult<()> {
        // Placeholder: CPU threading not yet implemented in backend
        Ok(())
    }

    /// Get the current number of threads for CPU operations (static method)
    /// Note: Currently returns 1 as threading is not implemented.
    /// This will return the actual thread count when CPU threading
    /// support is added to the tensor backend.
    #[staticmethod]
    fn get_num_threads() -> PyResult<usize> {
        // Placeholder: CPU threading not yet implemented in backend
        Ok(1)
    }

    /// Create a tensor filled with zeros
    #[staticmethod]
    pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::zeros(&shape).map_err(|e| {
            tensor_error!(e)
        })?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor filled with ones
    #[staticmethod]
    pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::ones(&shape).map_err(|e| {
            tensor_error!(e)
        })?;
        Ok(PyTensor { inner: tensor })
    }

    /// Implement __array__ method for NumPy compatibility
    /// This allows direct conversion to NumPy arrays via np.array(tensor)
    #[pyo3(signature = (*, dtype=None, copy=None))]
    fn __array__(&self, py: Python, dtype: Option<PyObject>, copy: Option<bool>) -> PyResult<Py<PyAny>> {
        // Get tensor data and shape
        let shape = self.inner.shape().dims().to_vec();

        // Extract raw float data from the tensor storage
        // Convert Float32 values to f32 for NumPy compatibility
        let data: Vec<f32> = self.inner.as_slice().iter().map(|&x| x.get()).collect();

        // Create NumPy array from the data and reshape to correct shape
        let array = numpy::PyArray::from_vec(py, data);
        let reshaped = array.reshape(shape)
            .map_err(|e| tensor_error!(e))?;

        Ok(reshaped.unbind().into())
    }
}

// TODO: Implement full buffer protocol support (PyBufferProtocol)
// The buffer protocol enables zero-copy memory views between Rust tensors and NumPy
// For now, the __array__ method provides the primary NumPy compatibility

/// Device enum
#[pyclass(name = "Device", module = "_coeus")]
pub enum Device {
    CPU,
    CUDA,
}
