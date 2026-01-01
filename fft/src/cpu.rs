//! CPU-based FFT implementation using rustfft

use coeus_error::Result;
pub use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::sync::Arc;
use storage::{DenseStorage, Storage};

/// CPU FFT processor using rustfft
pub struct CpuFft {
    size: usize,
    forward: Arc<dyn rustfft::Fft<f32>>,
    inverse: Arc<dyn rustfft::Fft<f32>>,
}

impl CpuFft {
    /// Create a new CPU FFT processor
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        Self {
            size,
            forward,
            inverse,
        }
    }

    /// Perform forward FFT from f32 to Complex32
    pub fn forward(
        &self,
        input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let data = input.as_slice();
        if data.len() < self.size {
            return Err(coeus_error::TensorError::ShapeMismatch(format!(
                "FFT input too short: expected at least {}, got {}",
                self.size,
                data.len()
            ))
            .into());
        }

        let mut complex_data: Vec<Complex32> = data[..self.size]
            .iter()
            .map(|&x| Complex32::new(x.get(), 0.0))
            .collect();

        self.forward.process(&mut complex_data);

        // Convert rustfft::Complex32 to dtype::complex::Complex32
        // Both are num_complex::Complex<f32>
        DenseStorage::from_vec(complex_data, &[self.size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }

    /// Perform inverse FFT from Complex32 to f32
    pub fn inverse(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        let data = input.as_slice();
        if data.len() < self.size {
            return Err(coeus_error::TensorError::ShapeMismatch(format!(
                "FFT input too short: expected at least {}, got {}",
                self.size,
                data.len()
            ))
            .into());
        }

        let mut complex_data = data[..self.size].to_vec();
        self.inverse.process(&mut complex_data);

        // Scaling is handled by rustfft if requested, but generally we scale by 1/N
        let scale = 1.0 / self.size as f32;
        let float_data: Vec<dtype::float::Float32> = complex_data
            .into_iter()
            .map(|c| dtype::float::Float32::new(c.re * scale))
            .collect();

        DenseStorage::from_vec(float_data, &[self.size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    fn approx_eq(a: &[Float32], b: &[Float32], atol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x.get() - y.get()).abs();
            assert!(d <= atol, "diff {d} > {atol}");
        }
    }

    #[test]
    fn cpu_fft_inverse_roundtrip() {
        let n = 256usize;
        let input_vec: Vec<Float32> = (0..n)
            .map(|i| {
                let t = i as f32 / 32.0;
                Float32::new((2.0 * std::f32::consts::PI * t).sin())
            })
            .collect();
        let input = match DenseStorage::from_vec(input_vec, &[n]) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };

        let fft = CpuFft::new(n);
        let spec = match fft.forward(&input) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        let recon = match fft.inverse(&spec) {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        };
        approx_eq(recon.as_slice(), input.as_slice(), 1e-4);
    }
}
