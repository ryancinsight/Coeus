//! Python protocol methods: representation, truthiness, numeric conversion, length.

use super::PyTensor;
use crate::error::map_backend_error;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn __repr__(&self) -> PyResult<String> {
        let shape = self.shape();
        let requires_grad = self.inner.grad.is_some();
        let data = self
            .inner
            .tensor
            .to_contiguous()
            .map_err(map_backend_error)?;
        let vals = data.as_slice();
        let max_display = 8;
        let data_str = if vals.is_empty() {
            "[]".to_string()
        } else if vals.len() <= max_display {
            let formatted: Vec<String> = vals
                .iter()
                .map(|&v| {
                    if v == (v as i64) as f64 && v.abs() < 1e6 {
                        format!("{:.1}", v)
                    } else {
                        format!("{:.4}", v)
                    }
                })
                .collect();
            format!("[{}]", formatted.join(", "))
        } else {
            let first: Vec<String> = vals[..3].iter().map(|&v| format!("{:.4}", v)).collect();
            let last: Vec<String> = vals[vals.len() - 2..]
                .iter()
                .map(|&v| format!("{:.4}", v))
                .collect();
            format!("[{}, ..., {}]", first.join(", "), last.join(", "))
        };
        if requires_grad {
            Ok(format!(
                "Tensor({data_str}, shape={shape:?}, requires_grad=True)"
            ))
        } else {
            Ok(format!("Tensor({data_str}, shape={shape:?})"))
        }
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// Python truthiness: only a single-element tensor has a truth value.
    ///
    /// An empty tensor has no value to test and a multi-element tensor has
    /// more than one, so both are refused rather than answered — torch words
    /// these as "Boolean value of Tensor with no values / more than one value
    /// is ambiguous". torch raises `RuntimeError`; this binding's contract
    /// errors are `ValueError`, matching `item()`'s element-count error, so
    /// code that catches either keeps working and code that relied on the
    /// invented answer (`False`/`True`) now sees the refusal instead of a
    /// silent skip.
    ///
    /// # Errors
    /// A `ValueError` when the tensor does not hold exactly one element, or a
    /// backend failure while reading that element.
    fn __bool__(&self) -> PyResult<bool> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        match numel {
            1 => self
                .inner
                .tensor
                .to_contiguous()
                .map(|tensor| tensor.as_slice()[0] != 0.0)
                .map_err(map_backend_error),
            0 => Err(pyo3::exceptions::PyValueError::new_err(
                "__bool__: Boolean value of Tensor with no values is ambiguous",
            )),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "__bool__: Boolean value of Tensor with more than one value is ambiguous \
                 ({numel} elements)"
            ))),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        self.item()
    }

    fn __int__(&self) -> PyResult<i64> {
        self.item().map(|v| v as i64)
    }

    fn __len__(&self) -> PyResult<usize> {
        let shape = self.inner.tensor.shape();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__len__: tensor is 0-dimensional",
            ));
        }
        Ok(shape[0])
    }
}
