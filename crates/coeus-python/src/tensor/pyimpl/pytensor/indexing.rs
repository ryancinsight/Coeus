//! Indexing, assignment, and iteration.

use super::PyTensor;
use crate::error::map_backend_error;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn __getitem__(&self, index: &pyo3::Bound<'_, pyo3::PyAny>, py: Python<'_>) -> PyResult<Self> {
        let shape = self.inner.tensor.shape();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "__getitem__: cannot index a 0-dimensional tensor",
            ));
        }

        if let Ok(i) = index.extract::<i64>() {
            let n = shape[0] as i64;
            let normalized = if i < 0 { n + i } else { i };
            if !(0..n).contains(&normalized) {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "__getitem__: index {i} out of range for dim 0 size {n}"
                )));
            }
            let idx = normalized as usize;
            let ranges: Vec<(usize, usize)> = shape
                .iter()
                .enumerate()
                .map(|(d, &s)| if d == 0 { (idx, idx + 1) } else { (0, s) })
                .collect();
            let inner = py.allow_threads(|| {
                let sliced = coeus_autograd::slice(&self.inner, &ranges)?;
                coeus_autograd::squeeze(&sliced, Some(0))
            });
            return inner.map(Self::from_var).map_err(map_backend_error);
        }

        if let Ok(sl) = index.downcast::<pyo3::types::PySlice>() {
            let n = shape[0];
            let slice_len = isize::try_from(n).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "__getitem__: dim 0 size {n} exceeds Python slice bounds"
                ))
            })?;
            let indices = sl.indices(slice_len)?;
            if indices.step != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "__getitem__: slice step {} is unsupported; expected 1",
                    indices.step
                )));
            }
            let start = indices.start.max(0) as usize;
            let stop = (indices.stop.max(0) as usize).min(n);
            if start >= stop {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "__getitem__: empty slice [{start}:{stop}]"
                )));
            }
            let ranges: Vec<(usize, usize)> = shape
                .iter()
                .enumerate()
                .map(|(d, &s)| if d == 0 { (start, stop) } else { (0, s) })
                .collect();
            let inner = py.allow_threads(|| coeus_autograd::slice(&self.inner, &ranges));
            return inner.map(Self::from_var).map_err(map_backend_error);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "__getitem__: index must be an int or a slice",
        ))
    }

    fn __setitem__(
        &mut self,
        index: &pyo3::Bound<'_, pyo3::PyAny>,
        value: &pyo3::Bound<'_, pyo3::PyAny>,
        py: Python<'_>,
    ) -> PyResult<()> {
        self.reject_tracked_mutation("__setitem__")?;
        let shape = self.inner.tensor.shape().to_vec();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "__setitem__: cannot index a 0-dimensional tensor",
            ));
        }
        let n = shape[0];

        let idx = if let Ok(i) = index.extract::<i64>() {
            let normalized = if i < 0 { n as i64 + i } else { i };
            if normalized < 0 || normalized as usize >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "__setitem__: index {i} out of range for dim 0 size {n}"
                )));
            }
            normalized as usize
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__setitem__: index must be an int",
            ));
        };

        let row_numel: usize = shape[1..].iter().product::<usize>().max(1);
        let fill_data: Vec<f64> = if let Ok(v) = value.extract::<f64>() {
            vec![v; row_numel]
        } else if let Ok(t) = value.extract::<PyTensor>() {
            let cont = t.inner.tensor.to_contiguous().map_err(map_backend_error)?;
            cont.as_slice().to_vec()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__setitem__: value must be a float or Tensor",
            ));
        };

        if fill_data.len() != row_numel {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "__setitem__: value has {} elements but row requires {}",
                fill_data.len(),
                row_numel
            )));
        }

        let _ = py;
        let numel: usize = shape.iter().product();
        let mut host = vec![0.0f64; numel];
        use coeus_core::ComputeBackend;
        let backend = coeus_core::MoiraiBackend::new();
        backend
            .copy_to_host(self.inner.tensor.storage(), &mut host)
            .map_err(map_backend_error)?;
        let start = idx * row_numel;
        host[start..start + row_numel].copy_from_slice(&fill_data);
        self.inner.tensor =
            coeus_tensor::Tensor::from_slice(shape, &host).map_err(map_backend_error)?;
        Ok(())
    }

    fn __iter__(&self) -> PyResult<crate::tensor::PyTensorIterator> {
        let length = self.inner.tensor.shape().first().copied().ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("__iter__: tensor is 0-dimensional")
        })?;
        Ok(crate::tensor::PyTensorIterator {
            tensor: self.clone(),
            current: 0,
            length,
        })
    }
}
