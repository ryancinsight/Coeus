use autograd::ops::backward;
use backend::CpuBackend;
use dtype::float::Float32;
use numpy;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::pyclass;
use storage::DenseStorage;
use tensor::Tensor;

// Import the new error handling macros (exported at crate root)
use crate::tensor_error;
use pyo3::types::PySlice;
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
        let float_data: Vec<Float32> = data.into_iter().map(Float32).collect();
        let tensor = Tensor::from_vec(float_data, &shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Extract a scalar value from a single-element tensor
    fn item(&self) -> PyResult<f32> {
        if self.inner.shape().size() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "item() can only be called on single-element tensors",
            ));
        }
        Ok(self.inner.as_slice()[0].get())
    }

    /// Convert tensor to NumPy array
    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
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
        let result = tensor::ops::arithmetic::pow(&self.inner, &exponent.inner)
            .map_err(|e| tensor_error!(e))?;
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

    fn rsqrt(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::rsqrt(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn sin(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::sin(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn cos(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::cos(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn acos(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::acos(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn atan(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::atan(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn erf(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::erf(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn exp2(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::exp2(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn log10(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::log10(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn log2(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::log2(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn tan(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::tan(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn asin(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::asin(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn sinh(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::sinh(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn cosh(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::cosh(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn tanh(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::tanh(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn floor(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::floor(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn ceil(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::ceil(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn round(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::round(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn trunc(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::trunc(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    fn sign(&self) -> PyResult<PyTensor> {
        let result = tensor::ops::arithmetic::sign(&self.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (min=None, max=None))]
    fn clamp(&self, min: Option<f32>, max: Option<f32>) -> PyResult<PyTensor> {
        let min_val = min.map(Float32);
        let max_val = max.map(Float32);
        let result = tensor::ops::arithmetic::clamp(&self.inner, min_val, max_val)
            .map_err(|e| tensor_error!(e))?;
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
            let result = self
                .inner
                .fancy_index(&[idx])
                .map_err(|e| tensor_error!(e))?;
            return Ok(PyTensor { inner: result });
        }

        // Handle list of integers (fancy indexing)
        if let Ok(indices) = index.extract::<Vec<i32>>() {
            let result = self
                .inner
                .fancy_index(&indices)
                .map_err(|e| tensor_error!(e))?;
            return Ok(PyTensor { inner: result });
        }

        // Handle slice (advanced slicing) - simplified 1D support for now
        if let Ok(slice) = index.cast::<PySlice>() {
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
                let result = self
                    .inner
                    .advanced_slice(params)
                    .map_err(|e| tensor_error!(e))?;
                return Ok(PyTensor { inner: result });
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Slicing currently only fully supported for 1D tensors in this iteration",
                ));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported index type. Currently supports int, list[int], and slice (1D).",
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
                "Unsupported value type. Expected Tensor, int, or float.",
            ));
        };

        // Helper to check and expand scalar
        let expand_values = |target_len: usize, vals: &[Float32]| -> Result<Vec<Float32>, PyErr> {
            if vals.len() == 1 {
                Ok(vec![vals[0]; target_len])
            } else if vals.len() == target_len {
                Ok(vals.to_vec())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Shape mismatch in assignment: target size {}, values size {}",
                    target_len,
                    vals.len()
                )))
            }
        };

        // Handle integer indexing
        if let Ok(idx) = index.extract::<i32>() {
            let expanded = expand_values(1, &values)?;
            self.inner
                .fancy_assign(&[idx], &expanded)
                .map_err(|e| tensor_error!(e))?;
            return Ok(());
        }

        // Handle list of integers
        if let Ok(indices) = index.extract::<Vec<i32>>() {
            let expanded = expand_values(indices.len(), &values)?;
            self.inner
                .fancy_assign(&indices, &expanded)
                .map_err(|e| tensor_error!(e))?;
            return Ok(());
        }

        // Handle slice
        if let Ok(slice) = index.cast::<PySlice>() {
            let tensor_len = self.inner.len();
            let indices = slice.indices(tensor_len as isize)?;
            let start = indices.start as i32;
            let stop = indices.stop as i32;
            let step = indices.step as i32;

            // Calculate number of steps
            let steps = indices.slicelength;

            let params = &[(Some(start), Some(stop), step)];

            // Check if 1D
            if self.inner.shape().dims().len() == 1 {
                let expanded = expand_values(steps, &values)?;
                self.inner
                    .advanced_assign(params, &expanded)
                    .map_err(|e| tensor_error!(e))?;
                return Ok(());
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Slicing assignment currently only fully supported for 1D tensors in this iteration"
                 ));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported index type. Currently supports int, list[int], and slice (1D).",
        ))
    }

    fn clone(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.clone(),
        }
    }

    fn detach(&self) -> PyTensor {
        // Create a new tensor sharing data but detached from graph
        let mut new_tensor = PyTensor {
            inner: self.inner.clone(),
        };
        let _ = new_tensor.requires_grad_(false);
        new_tensor
    }

    fn cpu(&self) -> PyTensor {
        // Already on CPU
        self.clone()
    }

    fn cuda(&self) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "CUDA backend not yet implemented",
        ))
    }

    #[pyo3(signature = (device=None, dtype=None))]
    fn to(
        &self,
        py: Python,
        device: Option<Py<PyAny>>,
        dtype: Option<Py<PyAny>>,
    ) -> PyResult<PyTensor> {
        if let Some(device_obj) = device {
            let device_bound = device_obj.bind(py);

            if let Ok(device_enum) = device_bound.extract::<Device>() {
                match device_enum {
                    Device::CPU => {}
                    Device::CUDA => {
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "CUDA backend not yet implemented",
                        ));
                    }
                }
            } else if let Ok(device_str) = device_bound.extract::<String>() {
                let device_str = device_str.to_ascii_lowercase();
                match device_str.as_str() {
                    "cpu" => {}
                    "cuda" | "gpu" => {
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "CUDA backend not yet implemented",
                        ));
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unsupported device: {device_str}"
                        )));
                    }
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "device must be a Device or str",
                ));
            }
        }

        if let Some(dtype_obj) = dtype {
            let dtype_bound = dtype_obj.bind(py);
            let dtype_str = dtype_bound.str()?.to_str()?.to_ascii_lowercase();
            let is_f32 = dtype_str.contains("float32")
                || dtype_str.contains("torch.float32")
                || dtype_str.contains("f4")
                || dtype_str == "float";

            if !is_f32 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    format!("dtype conversion not implemented (requested: {dtype_str})"),
                ));
            }
        }

        Ok(self.clone())
    }
    pub(crate) fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .matmul(&other.inner)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    pub(crate) fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        let result = self
            .inner
            .transpose(dim0, dim1)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    pub(crate) fn reshape(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        let result = self.inner.reshape(&shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    pub(crate) fn view(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        self.reshape(shape)
    }

    pub(crate) fn flatten(&self, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
        // Simplified flatten, usually flatten(0, -1) -> 1D
        // But PyTorch supports flattening a range of dims.
        // For now, implementing simple global flatten if args are default-ish
        // To support full range flattening we need to calculate target shape.

        let dims = self.inner.shape().dims();
        let ndim = dims.len();

        // Handle negative end_dim
        let end = if end_dim < 0 {
            (ndim as isize + end_dim + 1) as usize
        } else {
            (end_dim + 1) as usize
        };

        if start_dim >= end {
            return Ok(self.clone());
        }

        // Calculate new shape
        let mut new_shape: Vec<isize> = dims.iter().take(start_dim).map(|&d| d as isize).collect();

        let flattened_size = dims
            .iter()
            .take(end)
            .skip(start_dim)
            .copied()
            .product::<usize>();
        new_shape.push(flattened_size as isize);

        new_shape.extend(dims.iter().skip(end).map(|&d| d as isize));

        self.reshape(new_shape)
    }

    pub(crate) fn squeeze(&self, dim: Option<usize>) -> PyResult<PyTensor> {
        let dims = self.inner.shape().dims();
        let mut new_shape = Vec::new();

        match dim {
            Some(d) => {
                for (i, &s) in dims.iter().enumerate() {
                    if i != d || s != 1 {
                        new_shape.push(s as isize);
                    }
                }
            }
            None => {
                for &s in dims {
                    if s != 1 {
                        new_shape.push(s as isize);
                    }
                }
            }
        }
        self.reshape(new_shape)
    }

    pub(crate) fn unsqueeze(&self, dim: usize) -> PyResult<PyTensor> {
        let dims = self.inner.shape().dims();
        let ndim = dims.len();

        // dim can be up to ndim (to append)
        if dim > ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension out of range (expected to be in range of [0, {}], but got {})",
                ndim, dim
            )));
        }

        let mut new_shape: Vec<isize> = dims.iter().take(dim).map(|&d| d as isize).collect();
        new_shape.push(1);
        new_shape.extend(dims.iter().take(ndim).skip(dim).map(|&d| d as isize));

        self.reshape(new_shape)
    }

    fn mm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }

    pub(crate) fn permute(&self, dims: Vec<usize>) -> PyResult<PyTensor> {
        let result = self.inner.permute(&dims).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    pub(crate) fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.bmm(&other.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (mat1, mat2, beta=1.0, alpha=1.0))]
    pub(crate) fn addmm(
        &self,
        mat1: &PyTensor,
        mat2: &PyTensor,
        beta: f32,
        alpha: f32,
    ) -> PyResult<PyTensor> {
        let result = self
            .inner
            .addmm(&mat1.inner, &mat2.inner, Float32(beta), Float32(alpha))
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn sum(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self
            .inner
            .sum(dim.as_deref(), keepdim)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self
            .inner
            .mean(dim.as_deref(), keepdim)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn max(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self
            .inner
            .max(dim.as_deref(), keepdim)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn min(&self, dim: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self
            .inner
            .min(dim.as_deref(), keepdim)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self
            .inner
            .argmax(dim, keepdim)
            .map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self
            .inner
            .argmin(dim, keepdim)
            .map_err(|e| tensor_error!(e))?;
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
        backward(&self.inner, None, false, false).map_err(|e| tensor_error!(e))?;
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
        let tensor = Tensor::zeros(&shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor filled with ones
    #[staticmethod]
    pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::ones(&shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor with random values from a standard normal distribution
    #[staticmethod]
    pub fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::randn(&shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor with random values from a uniform distribution [0, 1)
    #[staticmethod]
    pub fn rand(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::rand(&shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor with random integers from [low, high)
    #[staticmethod]
    pub fn randint(low: i64, high: i64, shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::randint(low, high, &shape).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor filled with zeros with the same shape as input
    #[staticmethod]
    pub fn zeros_like(input: &PyTensor) -> PyResult<PyTensor> {
        let tensor = Tensor::zeros_like(&input.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor filled with ones with the same shape as input
    #[staticmethod]
    pub fn ones_like(input: &PyTensor) -> PyResult<PyTensor> {
        let tensor = Tensor::ones_like(&input.inner).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a tensor filled with a constant value with the same shape as input
    #[staticmethod]
    pub fn full_like(input: &PyTensor, fill_value: f32) -> PyResult<PyTensor> {
        let tensor =
            Tensor::full_like(&input.inner, Float32(fill_value)).map_err(|e| tensor_error!(e))?;
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
        let tensor = Tensor::from_vec(vec![Float32(fill_value); shape.iter().product()], &shape)
            .map_err(|e| tensor_error!(e))?;
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
            return Ok(PyTensor {
                inner: Tensor::from_vec(vec![], &[0]).map_err(|e| tensor_error!(e))?,
            });
        }
        if steps == 1 {
            return Ok(PyTensor {
                inner: Tensor::from_vec(vec![Float32(start)], &[1])
                    .map_err(|e| tensor_error!(e))?,
            });
        }
        let step = (end - start) / (steps - 1) as f32;
        let data: Vec<Float32> = (0..steps)
            .map(|i| Float32(start + i as f32 * step))
            .collect();
        let tensor = Tensor::from_vec(data, &[steps]).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Create a 1D tensor with `steps` values from `base^start` to `base^end` inclusive
    #[staticmethod]
    #[pyo3(signature = (start, end, steps=100, base=10.0))]
    pub fn logspace(start: f32, end: f32, steps: usize, base: f32) -> PyResult<PyTensor> {
        let lin = Self::linspace(start, end, steps)?;
        let data: Vec<Float32> = lin
            .inner
            .as_slice()
            .iter()
            .map(|&x| Float32(base.powf(x.get())))
            .collect();
        let tensor = Tensor::from_vec(data, &[steps]).map_err(|e| tensor_error!(e))?;
        Ok(PyTensor { inner: tensor })
    }

    /// Implement __array__ method for NumPy compatibility
    /// This allows direct conversion to NumPy arrays via np.array(tensor)
    #[pyo3(signature = (*, dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python,
        dtype: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if matches!(copy, Some(false)) {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "copy=False requires zero-copy buffer protocol support",
            ));
        }

        if let Some(dtype_obj) = dtype {
            let dtype_bound = dtype_obj.bind(py);
            let dtype_str = dtype_bound.str()?.to_str()?.to_ascii_lowercase();
            let is_f32 = dtype_str.contains("float32")
                || dtype_str.contains("torch.float32")
                || dtype_str.contains("f4")
                || dtype_str == "float";

            if !is_f32 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    format!("dtype conversion not implemented (requested: {dtype_str})"),
                ));
            }
        }

        // Get tensor data and shape
        let shape = self.inner.shape().dims().to_vec();

        // Extract raw float data from the tensor storage
        // Convert Float32 values to f32 for NumPy compatibility
        let data: Vec<f32> = self.inner.as_slice().iter().map(|&x| x.get()).collect();

        // Create NumPy array from the data and reshape to correct shape
        let array = numpy::PyArray::from_vec(py, data);
        let reshaped = array.reshape(shape).map_err(|e| tensor_error!(e))?;

        Ok(reshaped.unbind().into())
    }
}

/// Device enum
#[pyclass(name = "Device", module = "_coeus")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA,
}
