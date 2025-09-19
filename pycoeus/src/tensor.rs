use coeus_tensor::Tensor as RustTensor;
use numpy::{IxDyn, PyArray, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

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
#[derive(Clone)]
pub struct PyTensor {
    /// The underlying Rust tensor (wrapped to avoid thread safety issues)
    pub tensor: RustTensor<f32>,
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
        let tensor = RustTensor::from_vec(data, shape);

        Ok(PyTensor {
            tensor,
            requires_grad: false,
            device: Device::Cpu,
        })
    }

    /// Create a new tensor from NumPy array (efficient copy)
    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=None))]
    pub fn from_numpy(array: PyObject, requires_grad: Option<bool>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let requires_grad = requires_grad.unwrap_or(false);

            // Check if it's a numpy array
            if let Ok(pyarray) = array.downcast_bound::<PyArray<f32, IxDyn>>(py) {
                // Get the array buffer directly and copy data efficiently
                let buffer = pyarray.readonly();
                let data_slice = buffer.as_slice()?;
                let shape: Vec<usize> = pyarray.shape().to_vec();

                // Copy data from slice to Vec (efficient memcpy)
                let data: Vec<f32> = data_slice.to_vec();

                // Create tensor from copied data
                let mut tensor = RustTensor::from_vec(data, shape);

                if requires_grad {
                    tensor.set_requires_grad(true);
                    return Ok(PyTensor {
                        tensor,
                        requires_grad: true,
                        device: Device::Cpu,
                    });
                }

                Ok(PyTensor {
                    tensor,
                    requires_grad: false,
                    device: Device::Cpu,
                })
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected numpy array",
                ))
            }
        })
    }

    /// Create a new tensor from data and shape with gradient tracking
    #[staticmethod]
    #[pyo3(signature = (data, shape, requires_grad=None))]
    pub fn from_data(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let mut tensor = RustTensor::from_vec(data, shape);
        let requires_grad = requires_grad.unwrap_or(false);

        if requires_grad {
            tensor.set_requires_grad(true);
        }

        Ok(PyTensor {
            tensor,
            requires_grad,
            device: Device::Cpu,
        })
    }

    /// Get tensor data as Python list
    #[allow(clippy::useless_conversion)]
    pub fn data(&self) -> PyResult<Vec<f32>> {
        Ok(self.tensor.data().to_vec())
    }

    /// Update tensor data in-place
    pub fn update_data(&mut self, new_data: Vec<f32>) -> PyResult<()> {
        if new_data.len() != self.tensor.numel() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Data length mismatch: expected {}, got {}",
                self.tensor.numel(),
                new_data.len()
            )));
        }

        // Update the tensor data in-place
        let data_slice = self.tensor.data_mut();
        data_slice.copy_from_slice(&new_data);

        Ok(())
    }

    /// Get tensor shape
    #[allow(clippy::useless_conversion)]
    pub fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.tensor.shape().to_vec())
    }

    /// Get number of dimensions
    #[allow(clippy::useless_conversion)]
    fn dim(&self) -> PyResult<usize> {
        Ok(self.tensor.shape().len())
    }

    /// Get total number of elements
    #[allow(clippy::useless_conversion)]
    fn numel(&self) -> PyResult<usize> {
        Ok(self.tensor.shape().iter().product())
    }

    /// Enable gradient computation
    fn requires_grad_(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        self.tensor.set_requires_grad(requires_grad);
    }

    /// Check if gradients are required
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        // Zero gradients on the underlying Rust tensor
        self.tensor.zero_grad();
    }

    /// Get gradient tensor
    pub fn grad(&self) -> Option<PyTensor> {
        self.tensor.grad().map(|grad_tensor| PyTensor {
            tensor: grad_tensor,
            requires_grad: false, // Gradients don't require gradients by default
            device: self.device.clone(),
        })
    }

    /// Set gradient tensor
    fn set_grad(&mut self, grad_tensor: &PyTensor) -> PyResult<()> {
        self.tensor
            .set_grad(grad_tensor.tensor.clone())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to set gradient: {:?}",
                    e
                ))
            })
    }

    /// Get device
    fn device(&self) -> Device {
        self.device.clone()
    }

    /// Perform backward pass to compute gradients
    fn backward(&mut self) -> PyResult<()> {
        self.tensor.backward().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Backward pass failed: {:?}",
                e
            ))
        })
    }

    /// Move tensor to CPU
    #[allow(clippy::useless_conversion)]
    fn cpu(&self) -> PyResult<Self> {
        // For now, assume tensor is already on CPU
        Ok(PyTensor {
            tensor: self.tensor.clone(),
            requires_grad: self.requires_grad,
            device: Device::Cpu,
        })
    }

    /// Move tensor to CUDA (future implementation)
    #[allow(clippy::useless_conversion)]
    fn cuda(&self) -> PyResult<Self> {
        // Future: implement GPU transfer
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "CUDA support not yet implemented",
        ))
    }

    /// Detach tensor from computational graph
    fn detach(&self) -> PyResult<PyTensor> {
        // Create a new tensor without gradient tracking
        Ok(PyTensor {
            tensor: self.tensor.clone(), // Clone the underlying tensor
            requires_grad: false,        // Disable gradient tracking
            device: self.device.clone(),
        })
    }

    /// Clone the tensor
    fn clone(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: self.tensor.clone(),
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    /// Get scalar value (panics if tensor is not scalar)
    fn item(&self) -> PyResult<f32> {
        if !self.tensor.shape().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Can only call item() on scalar tensors",
            ));
        }

        Ok(self.tensor.data()[0])
    }

    /// Element-wise absolute value
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn abs(&self) -> PyResult<PyTensor> {
        let abs_tensor = self.tensor.abs();

        Ok(PyTensor {
            tensor: abs_tensor,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    // Arithmetic operations
    #[allow(clippy::useless_conversion)]
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.tensor + &other.tensor).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Tensor operation failed: {e:?}"
            ))
        })?;

        let new_tensor = PyTensor {
            tensor: result,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        };

        // Autograd graph registration is handled automatically by the underlying Rust tensor
        Ok(new_tensor)
    }

    #[allow(clippy::useless_conversion)]
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.tensor - &other.tensor).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Tensor operation failed: {e:?}"
            ))
        })?;

        let new_tensor = PyTensor {
            tensor: result,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        };

        // Autograd graph registration is handled automatically by the underlying Rust tensor
        Ok(new_tensor)
    }

    #[allow(clippy::useless_conversion)]
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.tensor * &other.tensor).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Tensor operation failed: {e:?}"
            ))
        })?;

        let new_tensor = PyTensor {
            tensor: result,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        };

        // Autograd graph registration is handled automatically by the underlying Rust tensor
        Ok(new_tensor)
    }

    #[allow(clippy::useless_conversion)]
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn __truediv__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = (&self.tensor / &other.tensor).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Tensor operation failed: {e:?}"
            ))
        })?;

        let new_tensor = PyTensor {
            tensor: result,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        };

        // Autograd graph registration is handled automatically by the underlying Rust tensor
        Ok(new_tensor)
    }

    /// Matrix multiplication (@ operator)
    #[allow(clippy::useless_conversion)]
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn __matmul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.tensor.matmul(&other.tensor).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Matrix multiplication failed: {:?}",
                e
            ))
        })?;

        let new_tensor = PyTensor {
            tensor: result,
            requires_grad: self.requires_grad || other.requires_grad,
            device: self.device.clone(),
        };

        // Autograd graph registration is handled automatically by the underlying Rust tensor
        Ok(new_tensor)
    }

    /// Matrix multiplication (matmul method)
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__matmul__(other)
    }

    // Method versions of operations
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__add__(other)
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__sub__(other)
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__mul__(other)
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.__truediv__(other)
    }

    // Mathematical operations
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn pow(&self, exponent: f32) -> PyResult<PyTensor> {
        let result = self.tensor.pow(exponent);

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    // __pow__ operator not implemented yet - use pow() method

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn exp(&self) -> PyResult<PyTensor> {
        let result = self.tensor.exp();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn log(&self) -> PyResult<PyTensor> {
        let result = self.tensor.log();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn sin(&self) -> PyResult<PyTensor> {
        let result = self.tensor.sin();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn cos(&self) -> PyResult<PyTensor> {
        let result = self.tensor.cos();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    // Reduction operations
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn sum(&self) -> PyResult<PyTensor> {
        let result = self.tensor.sum();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn mean(&self) -> PyResult<PyTensor> {
        let result = self
            .tensor
            .mean()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    // Shape manipulation
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyTensor> {
        let result = self.tensor.reshape(new_shape).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {e:?}"))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn transpose(&self, _dim0: usize, _dim1: usize) -> PyResult<PyTensor> {
        let result = self.tensor.t().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Transpose failed: {e:?}"))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    /// Expand tensor to new shape (broadcasting)
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn expand(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        // For now, use reshape if dimensions match, otherwise return error
        let current_numel = self.numel()?;
        let target_numel = shape.iter().product::<usize>();

        if current_numel == target_numel {
            let result = self.tensor.reshape(shape).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Reshape error: {e:?}"))
            })?;
            Ok(PyTensor {
                tensor: result,
                requires_grad: self.requires_grad,
                device: self.device.clone(),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot expand tensor to shape with different number of elements",
            ))
        }
    }

    /// Unsqueeze tensor (add dimension of size 1)
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn unsqueeze(&self, dim: usize) -> PyResult<PyTensor> {
        let result = self.tensor.unsqueeze(dim).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Unsqueeze error: {e:?}"))
        })?;
        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    /// Squeeze tensor (remove dimensions of size 1)
    #[pyo3(signature = (dim=None))]
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn squeeze(&self, dim: Option<usize>) -> PyResult<PyTensor> {
        let shape = self.shape()?;
        let result = if let Some(d) = dim {
            // Check if dimension exists and has size 1
            if d >= shape.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Dimension out of range",
                ));
            }
            if shape[d] != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Cannot squeeze dimension with size != 1",
                ));
            }

            // Remove the dimension by reshaping
            let mut new_shape = shape;
            new_shape.remove(d);
            self.tensor.reshape(new_shape).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Reshape error: {e:?}"))
            })?
        } else {
            // Remove all dimensions of size 1
            let new_shape: Vec<usize> = shape.into_iter().filter(|&s| s != 1).collect();
            self.tensor.reshape(new_shape).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Reshape error: {e:?}"))
            })?
        };

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    // Additional mathematical operations
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn sqrt(&self) -> PyResult<PyTensor> {
        let result = self.tensor.sqrt();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn tanh(&self) -> PyResult<PyTensor> {
        let result = self.tensor.tanh();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn relu(&self) -> PyResult<PyTensor> {
        let result = self.tensor.relu();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn sigmoid(&self) -> PyResult<PyTensor> {
        let result = self.tensor.sigmoid();

        Ok(PyTensor {
            tensor: result,
            requires_grad: self.requires_grad,
            device: self.device.clone(),
        })
    }

    // Comparison operations - simplified for now
    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn eq(&self, other: &PyTensor) -> PyResult<bool> {
        // Simplified equality check - compare shapes and first few elements
        if self.shape()? != other.shape()? {
            return Ok(false);
        }
        let self_data = self.data()?;
        let other_data = other.data()?;
        if !self_data.is_empty() && !other_data.is_empty() {
            Ok((self_data[0] - other_data[0]).abs() < 1e-6)
        } else {
            Ok(true)
        }
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn lt(&self, _other: &PyTensor) -> PyResult<bool> {
        // Simplified comparison - not fully implemented
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Tensor comparison operations not yet fully implemented",
        ))
    }

    #[allow(clippy::useless_conversion)] // PyO3 requires Ok() wrapper for PyResult return type - clippy false positive
    fn gt(&self, _other: &PyTensor) -> PyResult<bool> {
        // Simplified comparison - not fully implemented
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Tensor comparison operations not yet fully implemented",
        ))
    }

    // Creation functions (will be added later)

    // String representation
    fn __str__(&self) -> String {
        format!(
            "Tensor({}, shape={:?}, requires_grad={})",
            self.tensor
                .data()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            self.tensor.shape(),
            self.requires_grad
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}
