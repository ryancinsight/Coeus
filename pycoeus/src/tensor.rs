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
use pyo3::types::{PySlice, PyTuple};
use std::convert::TryFrom;
use tensor::ops::comparison;

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

    /// Extract a scalar value from a single-element tensor
    fn item(&self) -> PyResult<f32> {
        if self.inner.shape().size() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "item() can only be called on single-element tensors"
            ));
        }
        Ok(self.inner.as_slice()[0].get())
    }

    /// Convert tensor to NumPy array
    fn numpy(&self, py: Python) -> PyResult<PyObject> {
        self.__array__(py, None, None)
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

    // Comparison Operators

    fn __eq__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = comparison::eq(&self.inner, &other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn __ne__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = comparison::ne(&self.inner, &other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn __lt__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = comparison::lt(&self.inner, &other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn __le__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = comparison::le(&self.inner, &other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn __gt__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = comparison::gt(&self.inner, &other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn __ge__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = comparison::ge(&self.inner, &other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    // Indexing

    fn __getitem__(&self, index: Bound<PyAny>) -> PyResult<PyTensor> {
        // Handle integer indexing (basic implementation)
        if let Ok(idx) = index.extract::<i32>() {
             // For 1D tensor, this is a single element selection, which strictly speaks returns a 0-d tensor in PyTorch
             // implementation detail: use fancy index for consistency
             let result = self.inner.fancy_index(&[idx]).map_err(|e| tensor_error!(e))?;
             return Ok(PyTensor { inner: result });
        }
        
        // Handle list of integers (fancy indexing)
        if let Ok(indices) = index.extract::<Vec<i32>>() {
             let result = self.inner.fancy_index(&indices).map_err(|e| tensor_error!(e))?;
             return Ok(PyTensor { inner: result });
        }

        // Handle slice (advanced slicing) - simplified 1D support for now
        if let Ok(slice) = index.downcast::<PySlice>() {
             let indices = slice.indices(self.inner.len() as isize)?;
             let start = indices.start as i32;
             let stop = indices.stop as i32;
             let step = indices.step as i32;
             
             // Convert to start/end/step format for advanced_slice
             // Note: internal implementation expects [(start, end, step)] per dim
             // This is a naive implementation assuming 1D for the slice or first dim
             // Ideally we need to parse multi-dim slices from PyTuple
             
             // Using fancy indexing with generated range for simplicity in this iteration if advanced_slice usage is complex
             // OR map to advanced_slice
             let params = &[(Some(start), Some(stop), step)]; 
             // Need to handle if tensor is > 1D, advanced_slice expects slice per dim
             // For full support we need to parse PyTuple
             
             // Fallback to advanced_slice if 1D
             if self.inner.shape().dims().len() == 1 {
                 let result = self.inner.advanced_slice(params).map_err(|e| tensor_error!(e))?;
                 return Ok(PyTensor { inner: result });
             } else {
                 return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Slicing currently only fully supported for 1D tensors in this iteration"
                 ));
             }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported index type. Currently supports int, list[int], and slice (1D)."
        ))
    }

    fn __setitem__(&mut self, index: Bound<PyAny>, value: Bound<PyAny>) -> PyResult<()> {
        let values: Vec<Float32> = if let Ok(val_tensor) = value.extract::<PyTensor>() {
            val_tensor.inner.as_slice().to_vec()
        } else if let Ok(val_float) = value.extract::<f32>() {
            // Will be repeated to match target size
            vec![Float32(val_float)]
        } else if let Ok(val_int) = value.extract::<i32>() {
             vec![Float32(val_int as f32)]
        } else {
             return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported value type. Expected Tensor, int, or float."
            ));
        };

        // Helper to check and expand scalar
        let expand_values = |target_len: usize, vals: &[Float32]| -> Result<Vec<Float32>, PyErr> {
             if vals.len() == 1 {
                 Ok(vec![vals[0]; target_len])
             } else if vals.len() == target_len {
                 Ok(vals.to_vec())
             } else {
                 Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Shape mismatch in assignment: target size {}, values size {}", target_len, vals.len())
                ))
             }
        };

        // Handle integer indexing
        if let Ok(idx) = index.extract::<i32>() {
             let expanded = expand_values(1, &values)?;
             self.inner.fancy_assign(&[idx], &expanded).map_err(|e| tensor_error!(e))?;
             return Ok(());
        }

        // Handle list of integers
        if let Ok(indices) = index.extract::<Vec<i32>>() {
             let expanded = expand_values(indices.len(), &values)?;
             self.inner.fancy_assign(&indices, &expanded).map_err(|e| tensor_error!(e))?;
             return Ok(());
        }

        // Handle slice
        if let Ok(slice) = index.downcast::<PySlice>() {
             let tensor_len = self.inner.len();
             let indices = slice.indices(tensor_len as isize)?;
             let start = indices.start as i32;
             let stop = indices.stop as i32;
             let step = indices.step as i32;

             // Calculate number of steps
             let steps = indices.slicelength as usize;
             
             let params = &[(Some(start), Some(stop), step)];
             
             // Check if 1D
             if self.inner.shape().dims().len() == 1 {
                 let expanded = expand_values(steps, &values)?;
                 self.inner.advanced_assign(params, &expanded).map_err(|e| tensor_error!(e))?;
                 return Ok(());
             } else {
                 return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Slicing assignment currently only fully supported for 1D tensors in this iteration"
                 ));
             }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported index type. Currently supports int, list[int], and slice (1D)."
        ))
    }
    
    fn clone(&self) -> PyTensor {
        PyTensor { inner: self.inner.clone() }
    }
    
    fn detach(&self) -> PyTensor {
        // Create a new tensor sharing data but detached from graph
        let mut new_tensor = PyTensor { inner: self.inner.clone() };
        let _ = new_tensor.requires_grad_(false);
        new_tensor
    }
    
    fn cpu(&self) -> PyTensor {
        // Already on CPU
        self.clone()
    }
    
    fn cuda(&self) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "CUDA backend not yet implemented"
        ))
    }
    
    #[pyo3(signature = (device=None, dtype=None))]
    fn to(&self, device: Option<&Bound<'_, PyAny>>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        // Placeholder implementation
        if let Some(_d) = device {
             // Check if it's "cuda" -> error
        }
        // Dtype conversion not fully implemented yet
        Ok(self.clone())
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

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.argmax(dim, keepdim).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.argmin(dim, keepdim).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    fn size(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
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
