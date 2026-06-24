// ── Python tensor wrapper ──

use coeus_autograd::Var;
use coeus_tensor::Tensor;
use pyo3::prelude::*;

/// Python-exposed tensor class wrapping autograd variables.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Var<f64>,
}

#[pymethods]
impl PyTensor {
    /// Create a tensor from a list of data and an optional shape.
    #[new]
    #[pyo3(signature = (data, shape = None, requires_grad = false))]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let tensor = Tensor::from_slice(shape, &data);
        Ok(Self {
            inner: Var::new(tensor, requires_grad),
        })
    }

    /// Shape getter.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.tensor.shape().to_vec()
    }

    /// Data getter.
    #[getter]
    fn data(&self) -> Vec<f64> {
        self.inner.tensor.to_contiguous().as_slice().to_vec()
    }

    /// Data setter.
    #[setter]
    fn set_data(&mut self, data: Vec<f64>) -> PyResult<()> {
        let shape = self.inner.tensor.shape().to_vec();
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data length {} does not match tensor shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            )));
        }
        self.inner.tensor = Tensor::from_slice(shape, &data);
        Ok(())
    }

    /// Gradient getter.
    #[getter]
    fn grad(&self) -> Option<Vec<f64>> {
        self.inner
            .grad()
            .map(|g| g.to_contiguous().as_slice().to_vec())
    }

    /// Run backward pass (releasing the GIL).
    fn backward(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.backward());
        Ok(())
    }

    /// Element-wise add (+).
    fn __add__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::add(&self.inner, &other.inner));
        Ok(Self { inner })
    }

    /// Element-wise sub (-).
    fn __sub__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sub(&self.inner, &other.inner));
        Ok(Self { inner })
    }

    /// Element-wise mul (*).
    fn __mul__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::mul(&self.inner, &other.inner));
        Ok(Self { inner })
    }

    /// Element-wise div (/).
    fn __truediv__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::div(&self.inner, &other.inner));
        Ok(Self { inner })
    }

    /// Matrix multiplication (@).
    fn __matmul__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::matmul(&self.inner, &other.inner));
        Ok(Self { inner })
    }

    /// Element-wise exponential.
    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::exp(&self.inner));
        Ok(Self { inner })
    }

    /// Element-wise natural logarithm.
    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log(&self.inner));
        Ok(Self { inner })
    }

    /// Sum along the specified axis.
    fn sum_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sum_axis(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Mean along the specified axis.
    fn mean_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::mean_axis(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Zero-copy reshape.
    fn reshape(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::reshape(&self.inner, shape));
        Ok(Self { inner })
    }

    /// Zero-copy permute.
    fn permute(&self, dims: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::permute(&self.inner, &dims));
        Ok(Self { inner })
    }

    /// Zero-copy squeeze of dimensions of size 1.
    #[pyo3(signature = (axis = None))]
    fn squeeze(&self, axis: Option<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::squeeze(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Zero-copy unsqueeze inserting a dimension of size 1.
    fn unsqueeze(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::unsqueeze(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Zero-copy transpose of a 2D tensor.
    fn t(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::transpose_2d(&self.inner));
        Ok(Self { inner })
    }

    /// Zero-copy transpose swapping dim0 and dim1.
    fn transpose(&self, dim0: usize, dim1: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::transpose(&self.inner, dim0, dim1));
        Ok(Self { inner })
    }

    /// Contiguous copy of the tensor.
    fn contiguous(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::contiguous(&self.inner));
        Ok(Self { inner })
    }

    /// Cumulative sum along `dim`.
    fn cumsum(&self, dim: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cumsum(&self.inner, dim));
        Ok(Self { inner })
    }

    /// Element-wise negation (unary −).
    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::neg(&self.inner));
        Ok(Self { inner })
    }

    /// Element-wise absolute value.
    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::abs(&self.inner));
        Ok(Self { inner })
    }

    /// Element-wise square root.
    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sqrt(&self.inner));
        Ok(Self { inner })
    }

    /// Element-wise power: `self ** exp`.
    ///
    /// `exp` is a scalar `f64` applied uniformly to all elements.
    fn pow(&self, exp: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::pow(&self.inner, exp));
        Ok(Self { inner })
    }

    /// Python `**` operator: `self ** exp`.
    fn __pow__(&self, exp: f64, _modulo: Option<i64>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::pow(&self.inner, exp));
        Ok(Self { inner })
    }

    /// Element-wise clamp to `[min_val, max_val]`.
    ///
    /// Gradient is 1 inside the clamp range and 0 at saturated positions.
    fn clamp(&self, min_val: f64, max_val: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::clamp(&self.inner, min_val, max_val));
        Ok(Self { inner })
    }

    /// Multiply all elements by a scalar (no intermediate broadcast tensor).
    fn scale(&self, s: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&self.inner, s));
        Ok(Self { inner })
    }

    /// Maximum along `axis`, output shape has that axis = 1.
    ///
    /// Backward: indicator gradient at the argmax position (tied maxima split equally).
    fn max_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::max_axis(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Minimum along `axis`, output shape has that axis = 1.
    ///
    /// Backward: indicator gradient at the argmin position (tied minima split equally).
    fn min_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::min_axis(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Numerically stable log-sum-exp along `axis`.
    ///
    /// `lse(x, axis) = log(sum(exp(x − max(x, axis)), axis)) + max(x, axis)`
    ///
    /// Output shape has `axis` reduced to size 1. Gradient equals softmax(x) along `axis`.
    fn log_sum_exp(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log_sum_exp(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Tracked element-wise sine.
    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sin(&self.inner));
        Ok(Self { inner })
    }

    /// Tracked element-wise cosine.
    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cos(&self.inner));
        Ok(Self { inner })
    }

    /// Flip the tensor along `axis`.
    fn flip(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::flip(&self.inner, axis));
        Ok(Self { inner })
    }

    /// Extract a scalar value from a single-element tensor.
    ///
    /// Raises `ValueError` if the tensor does not have exactly one element.
    fn item(&self) -> PyResult<f64> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        if numel != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "item(): tensor has {numel} elements, expected 1"
            )));
        }
        let contiguous = self.inner.tensor.to_contiguous();
        Ok(contiguous.as_slice()[0])
    }

    /// Total number of elements.
    fn numel(&self) -> usize {
        self.inner.tensor.shape().iter().product()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.tensor.ndim()
    }

    /// Zero the accumulated gradient.
    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    /// Repr representation.
    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, requires_grad={})",
            self.shape(),
            self.inner.grad.is_some()
        )
    }
}

/// Python-exposed StateDict class wrapping weight/bias checkpoints.
#[pyclass(name = "StateDict")]
pub struct PyStateDict {
    pub inner: coeus_tensor::checkpoint::StateDict<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyStateDict {
    #[new]
    fn new() -> Self {
        Self {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Insert a tensor into the state dict.
    fn insert(&mut self, name: String, tensor: &PyTensor) {
        self.inner.insert(name, tensor.inner.tensor.clone());
    }

    /// Get a tensor by name.
    fn get(&self, name: &str) -> Option<PyTensor> {
        self.inner.get(name).map(|t| PyTensor {
            inner: coeus_autograd::Var::new(t.clone(), false),
        })
    }

    /// Save state dict to a file path.
    fn save(&self, path: &str) -> PyResult<()> {
        let mut file = std::fs::File::create(path)?;
        self.inner.save(&mut file)?;
        Ok(())
    }

    /// Load state dict from a file path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let mut file = std::fs::File::open(path)?;
        let inner = coeus_tensor::checkpoint::StateDict::load(&mut file)?;
        Ok(Self { inner })
    }

    /// Repr representation.
    fn __repr__(&self) -> String {
        format!(
            "StateDict(keys={:?})",
            self.inner.tensors.keys().collect::<Vec<_>>()
        )
    }
}
