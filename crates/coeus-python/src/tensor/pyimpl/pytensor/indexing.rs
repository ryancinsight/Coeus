//! Indexing, assignment, and iteration.

use super::PyTensor;
use crate::error::map_backend_error;
use pyo3::prelude::*;

/// Normalize a Python index against `len`, refusing one outside the axis.
///
/// Python counts from `-len` (that element) through `len - 1`, so a negative
/// index wraps once and any other out-of-range value gets `None` rather than a
/// wrapped `usize`.
fn resolve_index(index: i64, len: usize) -> Option<usize> {
    let signed_len = i64::try_from(len).ok()?;
    let normalized = if index < 0 { signed_len + index } else { index };
    let resolved = usize::try_from(normalized).ok()?;
    (resolved < len).then_some(resolved)
}

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
            let Some(idx) = resolve_index(i, shape[0]) else {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "__getitem__: index {i} out of range for dim 0 size {}",
                    shape[0]
                )));
            };
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
            // `PySlice::indices` reports bounds within `[-slice_len, slice_len]`,
            // and `slice_len` came from the axis length, so a non-negative bound
            // fits `usize` and needs no further check.
            let start = usize::try_from(indices.start.max(0))
                .expect("invariant: a non-negative slice bound fits usize");
            let stop = usize::try_from(indices.stop.max(0))
                .expect("invariant: a non-negative slice bound fits usize")
                .min(n);
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
            let Some(idx) = resolve_index(i, n) else {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "__setitem__: index {i} out of range for dim 0 size {n}"
                )));
            };
            idx
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
