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

    /// Perform forward FFT (Complex -> Complex)
    pub fn fft(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
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

        let mut complex_data = data[..self.size].to_vec();
        self.forward.process(&mut complex_data);

        DenseStorage::from_vec(complex_data, &[self.size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }

    /// Perform inverse FFT (Complex -> Complex)
    pub fn ifft(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let data = input.as_slice();
        if data.len() < self.size {
            return Err(coeus_error::TensorError::ShapeMismatch(format!(
                "IFFT input too short: expected at least {}, got {}",
                self.size,
                data.len()
            ))
            .into());
        }

        let mut complex_data = data[..self.size].to_vec();
        self.inverse.process(&mut complex_data);

        // Standard 1/N scaling for IFFT
        let scale = 1.0 / self.size as f32;
        for c in &mut complex_data {
            c.re *= scale;
            c.im *= scale;
        }

        DenseStorage::from_vec(complex_data, &[self.size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }

    /// Perform Real-to-Complex FFT (Float -> Complex, one-sided)
    pub fn rfft(
        &self,
        input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        let data = input.as_slice();
        if data.len() < self.size {
            return Err(coeus_error::TensorError::ShapeMismatch(format!(
                "RFFT input too short: expected at least {}, got {}",
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

        // One-sided spectrum: size/2 + 1
        let out_size = self.size / 2 + 1;
        let one_sided = complex_data[..out_size].to_vec();

        DenseStorage::from_vec(one_sided, &[out_size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }

    /// Perform Complex-to-Real Inverse FFT (Complex -> Float, one-sided)
    pub fn irfft(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        let data = input.as_slice();
        let expected_input_size = self.size / 2 + 1;
        if data.len() < expected_input_size {
            return Err(coeus_error::TensorError::ShapeMismatch(format!(
                "IRFFT input too short: expected at least {}, got {}",
                expected_input_size,
                data.len()
            ))
            .into());
        }

        // Reconstruct full spectrum from positive frequencies
        let mut full_spectrum = vec![Complex32::default(); self.size];
        full_spectrum[0] = data[0];
        for i in 1..expected_input_size - 1 {
            full_spectrum[i] = data[i];
            full_spectrum[self.size - i] = data[i].conj();
        }

        // Handle Nyquist if even
        if self.size % 2 == 0 {
            full_spectrum[self.size / 2] = data[expected_input_size - 1];
        }

        self.inverse.process(&mut full_spectrum);

        // Extract real part and scale by 1/N
        let scale = 1.0 / self.size as f32;
        let float_data: Vec<dtype::float::Float32> = full_spectrum
            .into_iter()
            .map(|c| dtype::float::Float32::new(c.re * scale))
            .collect();

        DenseStorage::from_vec(float_data, &[self.size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }

    /// Legacy forward compatibility
    pub fn forward(
        &self,
        input: &DenseStorage<dtype::float::Float32>,
    ) -> Result<DenseStorage<dtype::complex::Complex32>> {
        // Old 'forward' returned full spectrum. This is less used in standard PyTorch but kept for compat.
        let data = input.as_slice();
        let mut complex_data: Vec<Complex32> = data[..self.size]
            .iter()
            .map(|&x| Complex32::new(x.get(), 0.0))
            .collect();
        self.forward.process(&mut complex_data);
        DenseStorage::from_vec(complex_data, &[self.size])
            .map_err(|e| coeus_error::StorageError::from(e.to_string()).into())
    }

    /// Legacy inverse compatibility
    pub fn inverse(
        &self,
        input: &DenseStorage<dtype::complex::Complex32>,
    ) -> Result<DenseStorage<dtype::float::Float32>> {
        let data = input.as_slice();
        let mut complex_data = data[..self.size].to_vec();
        self.inverse.process(&mut complex_data);
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
    fn cpu_fft_complex_roundtrip() {
        let n = 16usize;
        let input_vec: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(i as f32, (i as f32).sin()))
            .collect();
        let input = DenseStorage::from_vec(input_vec.clone(), &[n]).unwrap();

        let fft = CpuFft::new(n);
        let spec = fft.fft(&input).unwrap();
        let recon = fft.ifft(&spec).unwrap();

        for (a, b) in input_vec.iter().zip(recon.as_slice().iter()) {
            assert!((a.re - b.re).abs() < 1e-4);
            assert!((a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn cpu_rfft_roundtrip() {
        let n = 16usize;
        let input_vec: Vec<Float32> = (0..n)
            .map(|i| Float32::new((i as f32 / 4.0).cos()))
            .collect();
        let input = DenseStorage::from_vec(input_vec.clone(), &[n]).unwrap();

        let fft = CpuFft::new(n);
        let spec = fft.rfft(&input).unwrap();
        assert_eq!(spec.len(), n / 2 + 1);

        let recon = fft.irfft(&spec).unwrap();
        approx_eq(recon.as_slice(), &input_vec, 1e-4);
    }
}
