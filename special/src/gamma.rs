//! Gamma functions.
//!
//! Wrappers around `statrs` gamma implementations applied element-wise.

use backend::Backend;
use coeus_error::{Error, Result, StorageError, TensorError};
use dtype::DataType;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use storage::DenseStorage;
use tensor::Tensor;

pub trait Gamma<B, T>: Sized {
    fn gamma(&self) -> Result<Self>;
    fn lgamma(&self) -> Result<Self>;
    fn digamma(&self) -> Result<Self>;
    fn polygamma(&self, n: usize) -> Result<Self>;
    fn beta(&self, other: &Self) -> Result<Self>;
}

impl<B, T> Gamma<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive + ToPrimitive,
{
    fn gamma(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "gamma requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = statrs::function::gamma::gamma(x_f);
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "gamma result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn lgamma(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "lgamma requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = statrs::function::gamma::ln_gamma(x_f);
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "lgamma result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn digamma(&self) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "digamma requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = digamma_f64(x_f)?;
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "digamma result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn polygamma(&self, n: usize) -> Result<Self> {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "polygamma requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = polygamma_f64(n, x_f)?;
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "polygamma result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }

    fn beta(&self, other: &Self) -> Result<Self> {
        if self.shape().dims() != other.shape().dims() {
            return Err(Error::Tensor(TensorError::ShapeMismatch(
                "beta requires identical shapes".to_string(),
            )));
        }

        let mut data = Vec::with_capacity(self.as_slice().len());
        for (&x, &y) in self.as_slice().iter().zip(other.as_slice().iter()) {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "beta requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = y.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "beta requires lossless f64 conversion".to_string(),
                ))
            })?;
            let v = statrs::function::gamma::ln_gamma(x_f) + statrs::function::gamma::ln_gamma(y_f)
                - statrs::function::gamma::ln_gamma(x_f + y_f);
            let b = v.exp();
            let out = T::from_f64(b).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "beta result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(out);
        }

        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }
}

fn digamma_f64(x: f64) -> Result<f64> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x.is_infinite() {
        return Ok(x);
    }

    if x <= 0.0 {
        let frac = x.fract();
        if frac == 0.0 {
            return Err(Error::Tensor(TensorError::OperationFailed(
                "digamma has poles at non-positive integers".to_string(),
            )));
        }
        let sin = (std::f64::consts::PI * x).sin();
        if sin == 0.0 {
            return Err(Error::Tensor(TensorError::OperationFailed(
                "digamma undefined at poles".to_string(),
            )));
        }
        let cot = (std::f64::consts::PI * x).cos() / sin;
        return Ok(digamma_f64(1.0 - x)? - std::f64::consts::PI * cot);
    }

    let mut y = 0.0;
    let mut z = x;
    while z < 6.0 {
        y -= 1.0 / z;
        z += 1.0;
    }

    let inv = 1.0 / z;
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    let inv8 = inv4 * inv4;
    y += z.ln() - 0.5 * inv - (1.0 / 12.0) * inv2 + (1.0 / 120.0) * inv4 - (1.0 / 252.0) * inv6
        + (1.0 / 240.0) * inv8;
    Ok(y)
}

fn polygamma_f64(n: usize, x: f64) -> Result<f64> {
    if n == 0 {
        return digamma_f64(x);
    }
    if !x.is_finite() {
        return Ok(if x.is_sign_positive() { 0.0 } else { f64::NAN });
    }
    if x <= 0.0 {
        return Err(Error::Tensor(TensorError::OperationFailed(
            "polygamma for n>=1 requires x > 0".to_string(),
        )));
    }

    let mut sum = 0.0f64;
    let mut k = 0usize;
    let pref = factorial_f64(n) * if (n + 1).is_multiple_of(2) { 1.0 } else { -1.0 };
    let p = (n + 1) as i32;
    let eps = 1e-14f64;

    loop {
        let denom = x + (k as f64);
        let term = 1.0 / denom.powi(p);
        sum += term;
        if term < eps {
            break;
        }
        k += 1;
        if k > 10_000_000 {
            return Err(Error::Tensor(TensorError::OperationFailed(
                "polygamma series did not converge".to_string(),
            )));
        }
    }
    Ok(pref * sum)
}

fn factorial_f64(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * (k as f64))
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
    fn digamma_known_values() {
        let x1 = match Ten::from_vec(vec![Float32::new(1.0)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let y1 = match x1.digamma() {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(y1.as_slice()[0].get(), -0.577_215_7_f32, 1e-6);

        let x2 = match Ten::from_vec(vec![Float32::new(0.5)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let y2 = match x2.digamma() {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(y2.as_slice()[0].get(), -1.963_51_f32, 1e-6);
    }

    #[test]
    fn polygamma_trigamma_at_one() {
        let x = match Ten::from_vec(vec![Float32::new(1.0)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let y = match x.polygamma(1) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(y.as_slice()[0].get(), 1.644_934_f32, 1e-5);
    }

    #[test]
    fn beta_known_values() {
        let a = match Ten::from_vec(vec![Float32::new(1.0)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let b = match Ten::from_vec(vec![Float32::new(1.0)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let out = match a.beta(&b) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(out.as_slice()[0].get(), 1.0, 1e-6);

        let a = match Ten::from_vec(vec![Float32::new(0.5)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let b = match Ten::from_vec(vec![Float32::new(0.5)], &[1]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let out = match a.beta(&b) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx(out.as_slice()[0].get(), std::f32::consts::PI, 1e-5);
    }
}
