use autograd::ops::backward;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;
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

    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner - &other.inner;
        Ok(PyTensor { inner: result })
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner * &other.inner;
        Ok(PyTensor { inner: result })
    }

    fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = &self.inner / &other.inner;
        Ok(PyTensor { inner: result })
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::neg(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__add__(other)
    }

    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__sub__(other)
    }

    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__mul__(other)
    }

    fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__truediv__(other)
    }

    fn pow(&self, exponent: &PyTensor) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::pow(&self.inner, &exponent.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn abs(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::abs(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn exp(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::exp(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn log(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::log(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn sqrt(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::sqrt(&self.inner).map_err(|e| tensor_error!(e))?;
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

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn sum(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.sum(dim.as_deref(), keepdim).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.mean(dim.as_deref(), keepdim).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn max(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.max(dim.as_deref(), keepdim).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn min(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.min(dim.as_deref(), keepdim).map_err(|e| tensor_error!(e))?;
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
        backward(&self.inner, None, false, false).map_err(|e| {
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

    /// Create an uninitialized tensor (actually zeros in this implementation for safety)
    #[staticmethod]
    pub fn empty(shape: Vec<usize>) -> PyResult<PyTensor> {
        Self::zeros(shape)
    }

    /// Create a tensor filled with a constant value
    #[staticmethod]
    pub fn full(shape: Vec<usize>, fill_value: f32) -> PyResult<PyTensor> {
        let tensor = Tensor::from_vec(vec![Float32(fill_value); shape.iter().product()], &shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a 1D tensor with values from [start, end) with step
    #[staticmethod]
    #[pyo3(signature = (start, end=None, step=1.0))]
    pub fn arange(start: f32, end: Option<f32>, step: f32) -> PyResult<PyTensor> {
        let (real_start, real_end) = match end {
            Some(e) => (start, e),
            None => (0.0, start),
        };
        let mut data = Vec::new();
        let mut curr = real_start;
        while curr < real_end {
            data.push(Float32(curr));
            curr += step;
        }
        let len = data.len();
        let tensor = Tensor::from_vec(data, &[len]).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a 1D tensor with `steps` values from `start` to `end` inclusive
    #[staticmethod]
    #[pyo3(signature = (start, end, steps=100))]
    pub fn linspace(start: f32, end: f32, steps: usize) -> PyResult<PyTensor> {
        if steps == 0 {
            return Ok(PyTensor { inner: Tensor::from_vec(vec![], &[0]).map_err(|e| tensor_error!(e))? });
        }
        if steps == 1 {
            return Ok(PyTensor { inner: Tensor::from_vec(vec![Float32(start)], &[1]).map_err(|e| tensor_error!(e))? });
        }
        let step = (end - start) / (steps - 1) as f32;
        let data: Vec<Float32> = (0..steps).map(|i| Float32(start + i as f32 * step)).collect();
        let tensor = Tensor::from_vec(data, &[steps]).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a 1D tensor with `steps` values from `base^start` to `base^end` inclusive
    #[staticmethod]
    #[pyo3(signature = (start, end, steps=100, base=10.0))]
    pub fn logspace(start: f32, end: f32, steps: usize, base: f32) -> PyResult<PyTensor> {
        let lin = Self::linspace(start, end, steps)?;
        let data: Vec<Float32> = lin.inner.as_slice().iter().map(|&x| Float32(base.powf(x.get()))).collect();
        let tensor = Tensor::from_vec(data, &[steps]).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Implement __array__ method for NumPy compatibility
    /// This allows direct conversion to NumPy arrays via np.array(tensor)
    #[pyo3(signature = (*, dtype=None, copy=None))]
    fn __array__(&self, py: Python, dtype: Option<Py<PyAny>>, copy: Option<bool>) -> PyResult<Py<PyAny>> {
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
