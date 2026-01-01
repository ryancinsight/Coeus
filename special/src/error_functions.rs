//! Error functions.
//!
//! Wrappers around `statrs` error function implementations.

use backend::Backend;
use coeus_error::{Error, Result, StorageError, TensorError};
use dtype::DataType;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use storage::DenseStorage;
use tensor::Tensor;

pub trait Erf<B, T>: Sized {
    fn erf(&self) -> Result<Self>;
    fn erfc(&self) -> Result<Self>;
    fn erfinv(&self) -> Result<Self>;
    fn ndtr(&self) -> Result<Self>;
}

impl<B, T> Erf<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive + ToPrimitive,
{
    fn erf(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "erf requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = statrs::function::erf::erf(x_f);
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "erf result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn erfc(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "erfc requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = statrs::function::erf::erfc(x_f);
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "erfc result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn erfinv(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "erfinv requires lossless f64 conversion".to_string(),
                ))
            })?;
            if !x_f.is_finite() || x_f <= -1.0 || x_f >= 1.0 {
                return Err(Error::Tensor(TensorError::OperationFailed(
                    "erfinv domain is (-1, 1) and finite".to_string(),
                )));
            }
            let y_f = erfinv_f64(x_f);
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "erfinv result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn ndtr(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "ndtr requires lossless f64 conversion".to_string(),
                ))
            })?;
            let z = x_f / std::f64::consts::SQRT_2;
            let y_f = 0.5 * (1.0 + statrs::function::erf::erf(z));
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "ndtr result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }
}

fn erfinv_f64(x: f64) -> f64 {
    let a = 0.147;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ln = (1.0 - x * x).ln();
    let t = 2.0 / (std::f64::consts::PI * a) + ln / 2.0;
    let inside = t * t - ln / a;
    let mut y = sign * (inside.sqrt() - t).sqrt();
    let sqrt_pi = std::f64::consts::PI.sqrt();
    for _ in 0..3 {
        let err = statrs::function::erf::erf(y) - x;
        let deriv = (2.0 / sqrt_pi) * (-y * y).exp();
        if !err.is_finite() || !deriv.is_finite() || deriv == 0.0 {
            break;
        }
        y -= err / deriv;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::cpu::CpuBackend;
    use dtype::float::Float32;

    type Ten = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

    fn approx(a: f32, b: f32, atol: f32) {
        let d = (a - b).abs();
        assert!(d <= atol, "diff {d} > {atol}");
    }

    #[test]
    fn ndtr_known_values() {
        let x0 = match Ten::from_vec(vec![Float32::new(0.0)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let y0 = match x0.ndtr() {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(y0.as_slice()[0].get(), 0.5, 1e-6);

        let x1 = match Ten::from_vec(vec![Float32::new(1.0)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let y1 = match x1.ndtr() {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(y1.as_slice()[0].get(), 0.841_344_746_f32, 1e-5);
    }
}
