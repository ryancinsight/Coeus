use coeus_tensor::Tensor as RustTensor;
use coeus_backend::CpuBackend;
use pyo3::prelude::*;
use pyo3::{PyErr, exceptions::PyRuntimeError};

// TensorError to PyErr conversion is handled inline to avoid orphan rule violations

// Suppress PyO3 false positives for useless conversions
// PyO3 requires explicit Ok() wrapper for PyResult return types
#[allow(clippy::useless_conversion)]
#[allow(clippy::empty_line_after_outer_attr)]

/// Device enumeration for PyTorch compatibility
#[derive(Clone, Debug, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum Device {
    #[pyo3(name = "cpu")]
    Cpu,
    #[pyo3(name = "cuda")]
    Cuda,
}

#[pymethods]
impl Device {
    /// Get the device type as string (PyTorch compatibility)
    #[pyo3(name = "type")]
    fn device_type(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda => "cuda".to_string(),
        }
    }

    /// Get string representation
    fn __str__(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda => "cuda".to_string(),
        }
    }

    /// Get repr representation
    fn __repr__(&self) -> String {
        match self {
            Device::Cpu => "device(type='cpu')".to_string(),
            Device::Cuda => "device(type='cuda')".to_string(),
        }
    }
}

/// Simplified PyTorch-compatible Tensor class
/// Note: This is a basic implementation focusing on core functionality.
/// Full autograd support will be added in future iterations.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTensor {
    /// The underlying Rust tensor (wrapped to avoid thread safety issues)
    pub tensor: RustTensor<f32, CpuBackend>,
    /// Whether gradients should be computed (autograd framework ready)
    pub requires_grad: bool,
    /// Device information
    pub device: Device,
}

#[pymethods]
#[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
impl PyTensor {
    /// Create a new tensor from data and shape
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let backend = CpuBackend::default();
        let tensor = RustTensor::from_vec(backend, data, shape).map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor,
            requires_grad: false,
            device: Device::Cpu,
        })
    }

    /// Create a tensor filled with zeros
    #[staticmethod]
    pub fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let backend = CpuBackend::default();
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        let tensor = RustTensor::<f32, CpuBackend>::from_vec(backend, data, shape).map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor {
            tensor,
            requires_grad: false,
            device: Device::Cpu,
        })
    }

    /// Create a tensor filled with ones
    #[staticmethod]
    pub fn ones(shape: Vec<usize>) -> PyResult<Self> {
        let backend = CpuBackend::default();
        let size = shape.iter().product::<usize>();
        let data = vec![1.0f32; size];
        let tensor = RustTensor::<f32, CpuBackend>::from_vec(backend, data, shape).map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor {
            tensor,
            requires_grad: false,
            device: Device::Cpu,
        })
    }

    /// Create a tensor with random normal distribution
    #[staticmethod]
    pub fn randn(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = coeus_utils::random::randn(shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor {
            tensor,
            requires_grad: false,
            device: Device::Cpu,
        })
    }

    /// Extract the inner tensor from a PyTensor (for Python -> Rust conversion)
    /// Note: This method is for internal use and may change
    pub fn extract_inner_tensor(&self) -> PyResult<Self> {
        Ok(PyTensor {
            tensor: self.tensor.clone(),
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    /// Zero the gradients (PyTorch compatibility)
    pub fn zero_grad(&mut self) {
        // For now, this is a no-op since we don't have full autograd implementation
        // In a complete implementation, this would zero the gradient tensor
    }

    /// Get the gradient tensor (PyTorch compatibility)
    pub fn grad(&self) -> Option<PyTensor> {
        // For now, return None since we don't have full autograd implementation
        // In a complete implementation, this would return the gradient tensor
        None
    }

    /// Get the tensor shape
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }

    /// Get the tensor data as a vector
    pub fn data(&self) -> PyResult<Vec<f32>> {
        Ok(self.tensor.data().to_vec())
    }

    /// Get whether gradients are required (autograd compatibility)
    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Add two tensors (__add__ operator)
    pub fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        let result_tensor = (&self.tensor + &other.tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result_tensor,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        })
    }

    /// Backward pass for autograd (placeholder)
    pub fn backward(&self) -> PyResult<()> {
        // Placeholder - full autograd implementation would go here
        Ok(())
    }

    /// Matrix multiplication (__matmul__ operator)
    pub fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        let result_tensor = (&self.tensor * &other.tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(PyTensor {
            tensor: result_tensor,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        })
    }

    /// Update the tensor data (used by optimizers)
    pub fn update_data(&mut self, new_data: Vec<f32>) -> PyResult<()> {
        if new_data.len() != self.tensor.data().len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "New data length must match tensor size",
            ));
        }

        // Create a new tensor with the updated data and same shape
        let backend = CpuBackend::default();
        let new_tensor = RustTensor::from_vec(backend, new_data, self.tensor.shape().to_vec())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tensor creation failed: {}", e)))?;
        self.tensor = new_tensor;
        Ok(())
    }

    /// Set the number of threads for CPU operations
    #[staticmethod]
    pub fn set_num_threads(_num_threads: usize) -> PyResult<()> {
        // Placeholder - rayon thread pool management would go here
        // For now, this is a no-op
        Ok(())
    }

    /// Get the number of threads for CPU operations
    #[staticmethod]
    pub fn get_num_threads() -> PyResult<usize> {
        // Placeholder - return default thread count
        Ok(1)
    }

    /// Set the random seed
    #[staticmethod]
    pub fn manual_seed(_seed: u64) -> PyResult<()> {
        // Placeholder - random seed management would go here
        Ok(())
    }

    /// Check if CUDA is available
    #[staticmethod]
    pub fn cuda_is_available() -> PyResult<bool> {
        // GPU support not yet implemented
        Ok(false)
    }
}

impl PyTensor {
    /// Create a PyTensor from an existing Rust tensor
    pub fn from_rust_tensor(tensor: coeus_tensor::Tensor<f32, CpuBackend>) -> Self {
        PyTensor {
            tensor,
            requires_grad: false,
            device: Device::Cpu,
        }
    }
}

// Removed orphan rule violating From implementation
// Error handling done inline in PyO3 methods
