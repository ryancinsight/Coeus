//! Window functions for signal processing.
//!
//! Provides generation of standard window functions (Hann, Hamming, Blackman, etc.).

use backend::Backend;
use coeus_error::Result;
use dtype::DataType;
use num_traits::{Float, FromPrimitive};
use std::f64::consts::PI;
use storage::DenseStorage;
use tensor::Tensor;

/// Window generation trait
pub trait WindowFunc<B, T> {
    fn hann_window(window_length: usize, periodic: bool) -> Result<Self>
    where
        Self: Sized;
    fn hamming_window(window_length: usize, periodic: bool) -> Result<Self>
    where
        Self: Sized;
    fn blackman_window(window_length: usize, periodic: bool) -> Result<Self>
    where
        Self: Sized;
    fn bartlett_window(window_length: usize, periodic: bool) -> Result<Self>
    where
        Self: Sized;
}

impl<B, T> WindowFunc<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Float + FromPrimitive,
{
    fn hann_window(window_length: usize, periodic: bool) -> Result<Self> {
        let n = if periodic {
            window_length + 1
        } else {
            window_length
        };

        let mut data = Vec::with_capacity(window_length);

        if window_length == 1 {
            data.push(T::one());
        } else {
            for i in 0..window_length {
                let factor = 2.0 * PI * (i as f64) / ((n - 1) as f64);
                let val = 0.5 * (1.0 - factor.cos());
                let converted = T::from_f64(val).ok_or_else(|| {
                    coeus_error::TensorError::OperationFailed(
                        "Failed to convert hann window coefficient to target dtype".to_string(),
                    )
                })?;
                data.push(converted);
            }
        }

        let storage = DenseStorage::from_vec(data, &[window_length])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()))?;
        Ok(Tensor::from_storage(storage, B::default()))
    }

    fn hamming_window(window_length: usize, periodic: bool) -> Result<Self> {
        let n = if periodic {
            window_length + 1
        } else {
            window_length
        };

        let mut data = Vec::with_capacity(window_length);

        if window_length == 1 {
            data.push(T::one());
        } else {
            for i in 0..window_length {
                let factor = 2.0 * PI * (i as f64) / ((n - 1) as f64);
                let val = 0.54 - 0.46 * factor.cos();
                let converted = T::from_f64(val).ok_or_else(|| {
                    coeus_error::TensorError::OperationFailed(
                        "Failed to convert hamming window coefficient to target dtype".to_string(),
                    )
                })?;
                data.push(converted);
            }
        }

        let storage = DenseStorage::from_vec(data, &[window_length])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()))?;
        Ok(Tensor::from_storage(storage, B::default()))
    }

    fn blackman_window(window_length: usize, periodic: bool) -> Result<Self> {
        let n = if periodic {
            window_length + 1
        } else {
            window_length
        };

        let mut data = Vec::with_capacity(window_length);

        if window_length == 1 {
            data.push(T::one());
        } else {
            for i in 0..window_length {
                let phase = 2.0 * PI * (i as f64) / ((n - 1) as f64);
                let val = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                let converted = T::from_f64(val).ok_or_else(|| {
                    coeus_error::TensorError::OperationFailed(
                        "Failed to convert blackman window coefficient to target dtype".to_string(),
                    )
                })?;
                data.push(converted);
            }
        }

        let storage = DenseStorage::from_vec(data, &[window_length])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()))?;
        Ok(Tensor::from_storage(storage, B::default()))
    }

    fn bartlett_window(window_length: usize, periodic: bool) -> Result<Self> {
        let n = if periodic {
            window_length + 1
        } else {
            window_length
        };

        let mut data = Vec::with_capacity(window_length);

        if window_length == 1 {
            data.push(T::one());
        } else {
            let divisor = ((n - 1) as f64) / 2.0;
            for i in 0..window_length {
                let val = 1.0 - ((i as f64) - divisor).abs() / divisor;
                let converted = T::from_f64(val).ok_or_else(|| {
                    coeus_error::TensorError::OperationFailed(
                        "Failed to convert bartlett window coefficient to target dtype".to_string(),
                    )
                })?;
                data.push(converted);
            }
        }

        let storage = DenseStorage::from_vec(data, &[window_length])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()))?;
        Ok(Tensor::from_storage(storage, B::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::cpu::CpuBackend;
    use dtype::float::Float32;

    #[test]
    fn hann_window_matches_formula_nonperiodic() {
        type W = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
        let w = match W::hann_window(4, false) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let got: Vec<f32> = w.as_slice().iter().map(|v| v.get()).collect();
        let expected = [0.0f32, 0.75f32, 0.75f32, 0.0f32];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn hamming_window_endpoints_match_periodic_false() {
        type W = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
        let w = match W::hamming_window(8, false) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let got = w.as_slice();
        assert!((got[0].get() - 0.08).abs() < 1e-6);
        assert!((got[7].get() - 0.08).abs() < 1e-6);
    }
}
