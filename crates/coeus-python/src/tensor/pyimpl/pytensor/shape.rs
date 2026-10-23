//! Shape manipulation and layout views.

use super::PyTensor;
use crate::error::map_backend_error;
use coeus_autograd::Var;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn reshape(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::reshape(x, shape))
    }

    fn permute(&self, dims: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::permute(x, &dims))
    }

    #[pyo3(signature = (axis = None))]
    fn squeeze(&self, axis: Option<usize>, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::squeeze(x, axis))
    }

    fn unsqueeze(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::unsqueeze(x, axis))
    }

    fn t(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::transpose_2d)
    }

    fn transpose(&self, dim0: usize, dim1: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::transpose(x, dim0, dim1))
    }

    fn contiguous(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::contiguous)
    }

    #[pyo3(signature = (start_dim = 0, end_dim = -1))]
    fn flatten(&self, start_dim: i64, end_dim: i64, py: Python<'_>) -> PyResult<Self> {
        let ndim = self.inner.tensor.ndim() as i64;
        let start = if start_dim < 0 {
            (ndim + start_dim).max(0)
        } else {
            start_dim
        } as usize;
        let end = if end_dim < 0 {
            (ndim + end_dim).max(0)
        } else {
            end_dim.min(ndim - 1)
        } as usize;
        let shape = self.inner.tensor.shape();
        let mut new_shape: Vec<usize> = shape[..start].to_vec();
        new_shape.push(shape[start..=end].iter().product());
        new_shape.extend_from_slice(&shape[end + 1..]);
        self.unary_dispatch(py, |x| coeus_autograd::reshape(x, new_shape))
    }

    fn view(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::reshape(x, shape))
    }

    fn expand(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let src = self.inner.tensor.shape().to_vec();
        if src.len() != shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expand: ndim mismatch: src={} target={}",
                src.len(),
                shape.len()
            )));
        }
        for (s, t) in src.iter().zip(shape.iter()) {
            if *s != 1 && *s != *t {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "expand: incompatible dim: src={s} target={t}"
                )));
            }
        }
        let inner = py.allow_threads(|| {
            let zeros = coeus_tensor::Tensor::<f64, coeus_core::MoiraiBackend>::zeros(shape)?;
            let zeros_v = Var::new(zeros, false)?;
            coeus_autograd::add(&self.inner, &zeros_v)
        });
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    fn broadcast_to(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        self.expand(shape, py)
    }

    fn flip(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::flip(x, axis))
    }

    fn repeat(&self, reps: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::tile(&self.inner, &reps));
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    #[getter]
    #[expect(non_snake_case, reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097")]
    fn T(&self, py: Python<'_>) -> PyResult<Self> {
        let ndim = self.inner.tensor.ndim();
        if ndim != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tensor.T: expected 2-D tensor, got {ndim}-D"
            )));
        }
        let inner = py.allow_threads(|| coeus_autograd::permute(&self.inner, &[1, 0]));
        inner.map(Self::from_var).map_err(map_backend_error)
    }
}
