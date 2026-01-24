use crate::tensor::class::{PyTensor, TensorWrapper, to_py_err};
use crate::dispatch_tensor;
use pyo3::prelude::*;
use numpy::PyUntypedArrayMethods;
use numpy::PyReadonlyArrayDyn;
use dtype::float::{Float32, Float64};
use tensor::tensor_core::Tensor;
use dtype::int::Int64;

pub mod arithmetic;
pub mod activation;
pub mod comparison;
pub mod shape;
pub mod linalg;
pub mod reduction;
pub mod inplace;
pub mod conversion;


#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape=None))]
    fn new(data: &Bound<PyAny>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            let shape = arr.shape().to_vec();
            let flat = arr.as_slice().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Contiguous NumPy required"))?.iter().copied().map(Float32).collect();
            let tensor = Tensor::from_vec(flat, &shape).map_err(to_py_err)?;
            return Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(tensor) });
        }
        if let Ok(arr) = data.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            let shape = arr.shape().to_vec();
            let flat = arr.as_slice().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Contiguous NumPy required"))?.iter().copied().map(Float64).collect();
            let tensor = Tensor::from_vec(flat, &shape).map_err(to_py_err)?;
            return Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(tensor) });
        }
        if let Ok(flat) = data.extract::<Vec<f32>>() {
            let shape = shape.unwrap_or_else(|| vec![flat.len()]);
            let float_data = flat.into_iter().map(Float32).collect();
            let tensor = Tensor::from_vec(float_data, &shape).map_err(to_py_err)?;
            return Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(tensor) });
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Invalid data type for Tensor"))
    }

    pub fn clone(&self) -> PyTensor {
        PyTensor { inner: self.inner.clone() }
    }

    pub fn detach(&self) -> PyTensor {
        let mut cloned = self.inner.clone();
        cloned = cloned.requires_grad_(false);
        PyTensor { inner: cloned }
    }

    pub fn contiguous(&self) -> PyTensor {
        self.clone()
    }

    #[pyo3(name = "type")]
    pub fn py_type(&self) -> String {
        match &self.inner {
            TensorWrapper::CpuDenseF32(_) | TensorWrapper::CpuSparseF32(_) => "torch.FloatTensor".to_string(),
            TensorWrapper::CpuDenseF64(_) | TensorWrapper::CpuSparseF64(_) => "torch.DoubleTensor".to_string(),
            TensorWrapper::CpuDenseI64(_) => "torch.LongTensor".to_string(),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => "torch.cuda.FloatTensor".to_string(),
        }
    }

    pub fn type_as(&self, other: &PyTensor) -> PyResult<PyTensor> {
        if self.py_type() == other.py_type() { Ok(self.clone()) }
        else { Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("type_as mismatch")) }
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        dispatch_tensor!(self, inner => inner.shape().dims().to_vec())
    }

    pub fn size(&self) -> Vec<usize> {
        self.shape()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        match self.inner {
            TensorWrapper::CpuDenseF32(_) | TensorWrapper::CpuSparseF32(_) => "float32".to_string(),
            TensorWrapper::CpuDenseF64(_) | TensorWrapper::CpuSparseF64(_) => "float64".to_string(),
            TensorWrapper::CpuDenseI64(_) => "int64".to_string(),
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => "float32".to_string(),
        }
    }

    #[getter]
    pub fn device(&self) -> crate::tensor::class::Device {
        match self.inner {
            TensorWrapper::CpuDenseF32(_) | TensorWrapper::CpuDenseF64(_) | TensorWrapper::CpuDenseI64(_) | TensorWrapper::CpuSparseF32(_) | TensorWrapper::CpuSparseF64(_) => crate::tensor::class::Device::CPU,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => crate::tensor::class::Device::CUDA,
        }
    }

    #[getter]
    pub fn data(&self) -> PyTensor {
        self.clone()
    }

    #[getter]
    pub fn grad(&self) -> PyResult<Option<PyTensor>> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(inner) => match inner.grad() {
                Ok(grad) => Ok(Some(PyTensor { inner: TensorWrapper::CpuDenseF32(grad) })),
                Err(_) => Ok(None),
            },
            TensorWrapper::CpuDenseF64(inner) => match inner.grad() {
                Ok(grad) => Ok(Some(PyTensor { inner: TensorWrapper::CpuDenseF64(grad) })),
                Err(_) => Ok(None),
            },
            _ => Ok(None)
        }
    }

    fn requires_grad(&self) -> bool {
        dispatch_tensor!(self, inner => inner.requires_grad())
    }

    fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        let inner = std::mem::replace(&mut self.inner, TensorWrapper::CpuDenseF32(Tensor::zeros(&[0]).map_err(to_py_err)?));
        self.inner = inner.requires_grad_(requires_grad);
        Ok(())
    }

    #[pyo3(signature = (gradient=None))]
    fn backward(&self, gradient: Option<&PyTensor>) -> PyResult<()> {
        match gradient {
            Some(g) => match (&self.inner, &g.inner) {
                (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => a.backward_with_grad(b).map_err(to_py_err),
                (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => a.backward_with_grad(b).map_err(to_py_err),
                _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Backward mismatch"))
            },
            None => dispatch_tensor!(self, inner => inner.backward().map_err(to_py_err))
        }
    }

    pub fn numel(&self) -> usize {
        self.inner.shape().size()
    }

    #[pyo3(name = "ndim")]
    pub fn dim_count(&self) -> usize {
        self.inner.shape().dims().len()
    }

    pub fn add_scalar(&self, value: f64) -> PyResult<PyTensor> {
        let dtype = self.dtype();
        let scalar_tensor = PyTensor::full(self.shape(), value, Some(dtype.as_str()), Some("cpu"))?;
        self.add(&scalar_tensor)
    }
    pub fn add_scalar_f64(&self, value: f64) -> PyResult<PyTensor> { self.add_scalar(value) }
    


    /// Returns True if the tensor dtype is a floating point type.
    pub fn is_floating_point(&self) -> bool {
        matches!(
            &self.inner,
            TensorWrapper::CpuDenseF32(_)
                | TensorWrapper::CpuDenseF64(_)
                | TensorWrapper::CpuSparseF32(_)
                | TensorWrapper::CpuSparseF64(_)
        )
    }

    /// Returns True if the tensor is stored in sparse format.
    pub fn is_sparse(&self) -> bool {
        matches!(
            &self.inner,
            TensorWrapper::CpuSparseF32(_) | TensorWrapper::CpuSparseF64(_)
        )
    }

    /// Returns True if the tensor is stored in a contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        !self.is_sparse()
    }

    /// Returns True if the tensor is on CPU.
    pub fn is_cpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        { !matches!(&self.inner, TensorWrapper::GpuDenseF32(_)) }
        #[cfg(not(feature = "gpu"))]
        { true }
    }

    /// Returns True if the tensor is on CUDA GPU.
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "gpu")]
        { matches!(&self.inner, TensorWrapper::GpuDenseF32(_)) }
        #[cfg(not(feature = "gpu"))]
        { false }
    }

    /// Returns the number of dimensions.
    pub fn dim(&self) -> usize {
        self.inner.shape().dims().len()
    }

    /// Returns the number of dimensions (alias for dim()).
    pub fn ndimension(&self) -> usize {
        self.dim()
    }

    /// Returns the number of elements in the tensor.
    pub fn nelement(&self) -> usize {
        self.inner.shape().size()
    }

    /// Returns the size in bytes of each element.
    pub fn element_size(&self) -> usize {
        match &self.inner {
            TensorWrapper::CpuDenseF32(_) | TensorWrapper::CpuSparseF32(_) => 4,
            TensorWrapper::CpuDenseF64(_) | TensorWrapper::CpuSparseF64(_) => 8,
            TensorWrapper::CpuDenseI64(_) => 8,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32(_) => 4,
        }
    }

    /// Returns the total number of bytes consumed by the tensor data.
    pub fn nbytes(&self) -> usize {
        self.nelement() * self.element_size()
    }

    pub fn sort(&self, dim: usize, descending: bool) -> PyResult<(PyTensor, PyTensor)> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let (v, indices) = tensor::ops::sort(t, dim, descending).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF32(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            TensorWrapper::CpuDenseF64(t) => {
                let (v, indices) = tensor::ops::sort(t, dim, descending).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF64(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("sort not implemented for this storage"))
        }
    }

    pub fn topk(&self, k: usize, dim: usize, largest: bool) -> PyResult<(PyTensor, PyTensor)> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let (v, indices) = tensor::ops::topk(t, k, dim, largest).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF32(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            TensorWrapper::CpuDenseF64(t) => {
                let (v, indices) = tensor::ops::topk(t, k, dim, largest).map_err(to_py_err)?;
                let i64_data: Vec<Int64> = indices.into_iter().map(|idx| Int64(idx as i64)).collect();
                let i_tensor = Tensor::from_vec(i64_data, v.shape().dims()).map_err(to_py_err)?;
                Ok((
                    PyTensor { inner: TensorWrapper::CpuDenseF64(v) },
                    PyTensor { inner: TensorWrapper::CpuDenseI64(i_tensor) }
                ))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("topk not implemented for this storage"))
        }
    }

    pub fn unique(&self) -> PyResult<PyTensor> {
        match &self.inner {
            TensorWrapper::CpuDenseF32(t) => {
                let res = tensor::ops::unique(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            TensorWrapper::CpuDenseF64(t) => {
                let res = tensor::ops::unique(t).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("unique not implemented for this storage"))
        }
    }

    pub fn atan2(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &other.inner) {
            (TensorWrapper::CpuDenseF32(a), TensorWrapper::CpuDenseF32(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(a), TensorWrapper::CpuDenseF64(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            #[cfg(feature = "gpu")]
            (TensorWrapper::GpuDenseF32(a), TensorWrapper::GpuDenseF32(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::GpuDenseF32(res) })
            }
            (TensorWrapper::CpuSparseF32(a), TensorWrapper::CpuSparseF32(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                 Ok(PyTensor { inner: TensorWrapper::CpuSparseF32(res) })
            }
            (TensorWrapper::CpuSparseF64(a), TensorWrapper::CpuSparseF64(b)) => {
                let res = tensor::ops::atan2(a, b).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuSparseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "atan2 not implemented for this tensor type combination (integers not supported)",
            )),
        }
    }
}

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    #[pyfunction] fn relu(input: &PyTensor) -> PyResult<PyTensor> { input.relu() }
    #[pyfunction] fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> { input.sigmoid() }
    #[pyfunction] fn tanh(input: &PyTensor) -> PyResult<PyTensor> { input.tanh() }
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    Ok(())
}
