//! Bessel functions.
//!
//! Bessel functions for tensor elements.

use backend::Backend;
use coeus_error::{Error, Result, StorageError, TensorError};
use dtype::DataType;
use libm;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use storage::DenseStorage;
use tensor::Tensor;

pub trait Bessel<B, T>: Sized {
    fn bessel_j0(&self) -> Result<Self>;
    fn bessel_j1(&self) -> Result<Self>;
    fn bessel_y0(&self) -> Result<Self>;
    fn bessel_y1(&self) -> Result<Self>;
    fn bessel_i0(&self) -> Result<Self>;
    fn bessel_i1(&self) -> Result<Self>;
}

impl<B, T> Bessel<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive + ToPrimitive,
{
    fn bessel_j0(&self) -> Result<Self> {
        self.apply_unary(|x| Ok(libm::j0(x)))
    }

    fn bessel_j1(&self) -> Result<Self> {
        self.apply_unary(|x| Ok(libm::j1(x)))
    }

    fn bessel_y0(&self) -> Result<Self> {
        self.apply_unary(|x| {
            if !x.is_finite() || x <= 0.0 {
                return Err(Error::Tensor(TensorError::OperationFailed(
                    "bessel_y0 domain is (0, +inf) and finite".to_string(),
                )));
            }
            Ok(libm::y0(x))
        })
    }

    fn bessel_y1(&self) -> Result<Self> {
        self.apply_unary(|x| {
            if !x.is_finite() || x <= 0.0 {
                return Err(Error::Tensor(TensorError::OperationFailed(
                    "bessel_y1 domain is (0, +inf) and finite".to_string(),
                )));
            }
            Ok(libm::y1(x))
        })
    }

    fn bessel_i0(&self) -> Result<Self> {
        self.apply_unary(i0_f64)
    }

    fn bessel_i1(&self) -> Result<Self> {
        self.apply_unary(i1_f64)
    }
}

/// Helper trait for unary application to avoid code duplication
trait UnaryApply<T> {
    fn apply_unary<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(f64) -> Result<f64>,
        Self: Sized;
}

impl<B, T> UnaryApply<T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive + ToPrimitive,
{
    fn apply_unary<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(f64) -> Result<f64>,
    {
        let mut data = Vec::with_capacity(self.as_slice().len());
        for &x in self.as_slice() {
            let x_f = x.to_f64().ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "Bessel requires lossless f64 conversion".to_string(),
                ))
            })?;
            let y_f = f(x_f)?;
            let y = T::from_f64(y_f).ok_or_else(|| {
                Error::Tensor(TensorError::DataTypeMismatch(
                    "Bessel result requires f64->T conversion".to_string(),
                ))
            })?;
            data.push(y);
        }
        let storage = DenseStorage::from_vec(data, self.shape().dims())
            .map_err(|e| Error::Storage(StorageError::InvalidShape(format!("{e}"))))?;
        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }
}

fn i0_f64(x: f64) -> Result<f64> {
    if !x.is_finite() {
        return Err(Error::Tensor(TensorError::OperationFailed(
            "bessel_i0 requires finite input".to_string(),
        )));
    }
    let ax = x.abs();
    if ax < 3.75 {
        let t = ax / 3.75;
        let t2 = t * t;
        Ok(1.0
            + t2 * (3.515_622_9
                + t2 * (3.089_942_4
                    + t2 * (1.206_749_2
                        + t2 * (0.265_973_2 + t2 * (0.036_076_8 + t2 * 0.004_581_3))))))
    } else {
        let t = 3.75 / ax;
        let poly = 0.398_942_28
            + t * (0.013_285_92
                + t * (0.002_253_19
                    + t * (-0.001_575_65
                        + t * (0.009_162_81
                            + t * (-0.020_577_06
                                + t * (0.026_355_37 + t * (-0.016_476_33 + t * 0.003_923_77)))))));
        Ok((ax.exp() / ax.sqrt()) * poly)
    }
}

fn i1_f64(x: f64) -> Result<f64> {
    if !x.is_finite() {
        return Err(Error::Tensor(TensorError::OperationFailed(
            "bessel_i1 requires finite input".to_string(),
        )));
    }
    let ax = x.abs();
    let out = if ax < 3.75 {
        let t = ax / 3.75;
        let t2 = t * t;
        ax * (0.5
            + t2 * (0.878_905_94
                + t2 * (0.514_988_69
                    + t2 * (0.150_849_34
                        + t2 * (0.026_587_33 + t2 * (0.003_015_32 + t2 * 0.000_324_11))))))
    } else {
        let t = 3.75 / ax;
        let poly = 0.398_942_28
            + t * (-0.039_880_24
                + t * (-0.003_620_18
                    + t * (0.001_638_01
                        + t * (-0.010_315_55
                            + t * (0.022_829_67
                                + t * (-0.028_953_12 + t * (0.017_876_54 - t * 0.004_200_59)))))));
        (ax.exp() / ax.sqrt()) * poly
    };
    Ok(if x < 0.0 { -out } else { out })
}
