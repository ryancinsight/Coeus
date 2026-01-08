//! Python bindings for Tensor operations.
//!
//! This module provides PyTensor and Device classes for Python.

use tensor::ops::{arithmetic, comparison, creation, matrix, reduction, tensor_ops};
use tensor::tensor_core::Tensor;
use backend::CpuBackend;
use dtype::float::Float32;
use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use storage::DenseStorage;

/// Simple error conversion helper
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor error: {}", e))
}

/// Device enum wrapper
#[pyclass(name = "Device")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA,
}

/// Tensor wrapper for Python
/// 
/// Corresponds to torch.Tensor
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape=None))]
    fn new(data: &Bound<PyAny>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let _backend: CpuBackend<Float32> = CpuBackend::default();

        if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            let shape = arr.shape().to_vec();
            let flat = arr
                .as_slice()
                .map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Tensor data must be a contiguous NumPy array",
                    )
                })?
                .iter()
                .copied()
                .map(Float32)
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(flat, &shape).map_err(to_py_err)?;
            return Ok(PyTensor { inner: tensor });
        }

        if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            let shape = arr.shape().to_vec();
            let flat = arr
                .as_slice()
                .map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Tensor data must be a contiguous NumPy array",
                    )
                })?
                .iter()
                .copied()
                .map(|v| Float32(v as f32))
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(flat, &shape).map_err(to_py_err)?;
            return Ok(PyTensor { inner: tensor });
        }

        if let Ok(flat) = data.extract::<Vec<f32>>() {
            let shape = match shape {
                Some(s) => s,
                None => vec![flat.len()],
            };
    
            let float_data: Vec<Float32> = flat.into_iter().map(Float32).collect();
            let tensor = Tensor::from_vec(float_data, &shape).map_err(to_py_err)?;
            return Ok(PyTensor { inner: tensor });
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Tensor data must be a Python list or a NumPy ndarray",
        ))
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
    
    fn __array__(&self, py: Python, _dtype: Option<Py<PyAny>>, _context: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
         use numpy::PyArray1;
         let data: Vec<f32> = self.inner.as_slice().iter().map(|f| f.get()).collect();
         let array = PyArray1::from_vec(py, data);
         
         let shape = self.inner.shape().dims();
         let shaped_array = array.reshape(shape).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {}", e)))?;
         
         Ok(shaped_array.into())
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
        let result = arithmetic::neg(&self.inner).map_err(to_py_err)?;
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
        let result = arithmetic::pow(&self.inner, &exponent.inner)
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    fn abs(&self) -> PyResult<PyTensor> {
        let result = arithmetic::abs(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    // Activation Functions / Elementwise
    fn exp(&self) -> PyResult<PyTensor> {
        let result = arithmetic::exp(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    fn log(&self) -> PyResult<PyTensor> {
        let result = arithmetic::log(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    fn sqrt(&self) -> PyResult<PyTensor> {
        let result = arithmetic::sqrt(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    fn sin(&self) -> PyResult<PyTensor> {
        let result = arithmetic::sin(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    fn cos(&self) -> PyResult<PyTensor> {
        let result = arithmetic::cos(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    fn tanh(&self) -> PyResult<PyTensor> {
        let result = arithmetic::tanh(&self.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }
    
    fn size(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }
    
    // Factories
    #[staticmethod]
    pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::zeros(&shape).map_err(to_py_err)?;
        Ok(PyTensor { inner: tensor })
    }
    
    #[staticmethod]
    pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::ones(&shape).map_err(to_py_err)?;
        Ok(PyTensor { inner: tensor })
    }
    
    #[staticmethod]
    pub fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = creation::randn(&shape).map_err(to_py_err)?;
        Ok(PyTensor { inner: tensor })
    }
    
    #[staticmethod]
    pub fn rand(shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::rand(&shape).map_err(to_py_err)?;
        Ok(PyTensor { inner: tensor })
    }

    #[staticmethod]
    pub fn randint(low: i64, high: i64, shape: Vec<usize>) -> PyResult<PyTensor> {
        let tensor = Tensor::rand(&shape).map_err(to_py_err)?;
        let range = (high - low) as f32;
        let low_f = low as f32;
        
        let range_t = Tensor::from_vec(vec![Float32(range)], &[1]).map_err(to_py_err)?;
        let low_t = Tensor::from_vec(vec![Float32(low_f)], &[1]).map_err(to_py_err)?;

        let scaled = arithmetic::mul(&tensor, &range_t).map_err(to_py_err)?;
        let shifted = arithmetic::add(&scaled, &low_t).map_err(to_py_err)?;
        let floored = arithmetic::floor(&shifted).map_err(to_py_err)?;
        
        Ok(PyTensor { inner: floored })
    }

    #[staticmethod]
    pub fn zeros_like(input: &PyTensor) -> PyResult<PyTensor> {
        let shape = input.inner.shape().dims().to_vec();
        Self::zeros(shape)
    }

    #[staticmethod]
    pub fn ones_like(input: &PyTensor) -> PyResult<PyTensor> {
        let shape = input.inner.shape().dims().to_vec();
        Self::ones(shape)
    }

    #[staticmethod]
    pub fn full_like(input: &PyTensor, fill_value: f32) -> PyResult<PyTensor> {
        let shape = input.inner.shape().dims().to_vec();
        let size: usize = shape.iter().product();
        let data = vec![Float32(fill_value); size];
        let tensor = Tensor::from_vec(data, &shape).map_err(to_py_err)?;
        Ok(PyTensor { inner: tensor })
    }
    
    // Matrix Ops
    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.matmul(&other.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    pub fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.bmm(&other.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    #[pyo3(signature = (mat1, mat2, beta=1.0, alpha=1.0))]
    pub fn addmm(&self, mat1: &PyTensor, mat2: &PyTensor, beta: f32, alpha: f32) -> PyResult<PyTensor> {
        let result = self.inner.addmm(&mat1.inner, &mat2.inner, Float32(beta), Float32(alpha))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    // Shape Ops
    pub fn reshape(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        let result = self.inner.reshape(&shape).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    pub fn view(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        self.reshape(shape)
    }
    
    pub fn permute(&self, dims: Vec<usize>) -> PyResult<PyTensor> {
        let result = self.inner.permute(&dims).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }
    
    pub fn flatten(&self, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
         let dims = self.inner.shape().dims();
        let ndim = dims.len();

        let end = if end_dim < 0 {
            (ndim as isize + end_dim + 1) as usize
        } else {
            (end_dim + 1) as usize
        };

        if start_dim >= end {
            return Ok(self.clone());
        }

        let mut new_shape: Vec<isize> = dims.iter().take(start_dim).map(|&d| d as isize).collect();
        let flattened_size = dims.iter().take(end).skip(start_dim).copied().product::<usize>();
        new_shape.push(flattened_size as isize);
        new_shape.extend(dims.iter().skip(end).map(|&d| d as isize));

        self.reshape(new_shape)
    }
    
    pub fn squeeze(&self, dim: Option<usize>) -> PyResult<PyTensor> {
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

    pub fn unsqueeze(&self, dim: usize) -> PyResult<PyTensor> {
        let dims = self.inner.shape().dims();
        let ndim = dims.len();

        let mut new_shape: Vec<isize> = dims.iter().take(dim).map(|&d| d as isize).collect();
        new_shape.push(1);
        new_shape.extend(dims.iter().take(ndim).skip(dim).map(|&d| d as isize));

        self.reshape(new_shape)
    }
    
    pub fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
        let result = self.inner.transpose(dim0, dim1).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.argmax(dim, keepdim).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = self.inner.argmin(dim, keepdim).map_err(to_py_err)?;
        Ok(PyTensor { inner: result })
    }

    /// Sum of all elements or along a dimension
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = match dim {
            Some(d) => self.inner.sum_dims(Some(&[d]), keepdim).map_err(to_py_err)?,
            None => self.inner.sum_all(),
        };
        Ok(PyTensor { inner: result })
    }

    /// Mean of all elements or along a dimension
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn mean(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = match dim {
            Some(d) => self.inner.mean_dims(Some(&[d]), keepdim).map_err(to_py_err)?,
            None => self.inner.mean_all(),
        };
        Ok(PyTensor { inner: result })
    }

    /// Max of all elements or along a dimension
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn max(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = match dim {
            Some(d) => self.inner.max_dims(Some(&[d]), keepdim).map_err(to_py_err)?,
            None => self.inner.max_dims(None, keepdim).map_err(to_py_err)?,
        };
        Ok(PyTensor { inner: result })
    }

    /// Min of all elements or along a dimension
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn min(&self, dim: Option<usize>, keepdim: bool) -> PyResult<PyTensor> {
        let result = match dim {
            Some(d) => self.inner.min_dims(Some(&[d]), keepdim).map_err(to_py_err)?,
            None => self.inner.min_dims(None, keepdim).map_err(to_py_err)?,
        };
        Ok(PyTensor { inner: result })
    }

    fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        self.inner = self.inner.clone().requires_grad_(requires_grad);
        Ok(())
    }

    fn backward(&self) -> PyResult<()> {
        // Backward requires autograd crate which pycoeus can access
        // For now, delegate to autograd::backward if available
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
             "Backward not yet supported. Use external autograd function."
        ))
    }
    
    // Device Management
    fn cpu(&self) -> PyTensor {
        self.clone()
    }
    
    fn cuda(&self) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("CUDA not implemented"))
    }
    
    fn detach(&self) -> PyTensor {
         let mut new_tensor = PyTensor { inner: self.inner.clone() };
         let _ = new_tensor.requires_grad_(false);
         new_tensor
    }
}
