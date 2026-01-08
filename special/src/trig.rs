//! Miscellaneous math functions.
//!
//! Provides logit, expit (sigmoid), sinc, and other common ML math ops.

use backend::Backend;
use coeus_error::{Error, Result, StorageError, TensorError};
use dtype::DataType;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use storage::DenseStorage;
use tensor::Tensor;

pub trait Misc<B, T>: Sized {
    fn logit(&self, eps: Option<f64>) -> Result<Self>;
    fn expit(&self) -> Result<Self>;
    fn sinc(&self) -> Result<Self>;
}

impl<B, T> Misc<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive + ToPrimitive,
{
    fn logit(&self, eps: Option<f64>) -> Result<Self> {
        let eps = eps.unwrap_or(0.0);
        self.apply_unary(|x| {
            let x = if eps > 0.0 {
                x.clamp(eps, 1.0 - eps)
            } else {
                x
            };
            (x / (1.0 - x)).ln()
        })
    }

    fn expit(&self) -> Result<Self> {
        // Sigmoid: 1 / (1 + exp(-x))
        self.apply_unary(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn sinc(&self) -> Result<Self> {
        // Sinc: sin(pi*x) / (pi*x)
        self.apply_unary(|x| {
            if x == 0.0 {
                1.0
            } else {
                let pix = std::f64::consts::PI * x;
                pix.sin() / pix
            }
        })
    }
}

/// Helper trait for unary application
trait UnaryApply<T> {
    fn apply_unary<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(f64) -> f64,
        Self: Sized;
}

impl<B, T> UnaryApply<T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive + ToPrimitive,
{
    fn apply_unary<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(f64) -> f64,
    {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "Misc requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = f(x_f);
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "Misc result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }
}
