//! FFT operations for PyCoeus audio processing

use pyo3::prelude::*;
use pyo3::pyclass;
use rustfft::{Fft, FftPlanner};
use rustfft::num_complex::Complex32;

/// FFT operation for audio processing
#[pyclass(name = "FFT", module = "_coeus")]
pub struct FFT {
    /// Internal FFT processor
    planner: FftPlanner<f32>,
}

#[pymethods]
impl FFT {
    #[new]
    fn new(_size: usize) -> PyResult<Self> {
        let planner = FftPlanner::new();
        Ok(FFT { planner })
    }

    /// Get a description of the FFT
    #[must_use]
    fn description(&self) -> String {
        "RustFFT-based FFT processor".to_string()
    }

    /// Perform basic forward FFT on Python list
    /// This provides immediate functionality for testing and validation
    fn forward(&mut self, py: Python, input: Vec<f32>) -> PyResult<Vec<(f32, f32)>> {
        let result = py.allow_threads(|| {
            let mut complex_input: Vec<Complex32> = input.into_iter().map(|x| Complex32::new(x, 0.0)).collect();
            let fft = self.planner.plan_fft_forward(complex_input.len());
            fft.process(&mut complex_input);
            complex_input
        });

        // Convert to Python tuple of (real, imag) pairs
        Ok(result.into_iter().map(|c| (c.re, c.im)).collect())
    }

    /// Perform basic inverse FFT on complex coefficients
    /// Input should be list of (real, imag) tuples
    fn inverse(&mut self, py: Python, input: Vec<(f32, f32)>) -> PyResult<Vec<f32>> {
        // Convert tuples to Complex32
        use rustfft::num_complex::Complex32;
        let complex_input: Vec<Complex32> = input
            .into_iter()
            .map(|(re, im)| Complex32::new(re, im))
            .collect();

        let result = py.allow_threads(|| {
            let mut complex_buffer = complex_input;
            let fft = self.planner.plan_fft_inverse(complex_buffer.len());
            fft.process(&mut complex_buffer);
            complex_buffer.into_iter().map(|c| c.re).collect::<Vec<f32>>()
        });

        Ok(result)
    }

    /// Get FFT info
    fn __repr__(&self) -> String {
        "FFT(using RustFFT planner)".to_string()
    }
}

/// IFFT operation for audio processing
#[pyclass(name = "IFFT", module = "_coeus")]
pub struct IFFT {
    /// Internal FFT processor
    planner: FftPlanner<f32>,
}

#[pymethods]
impl IFFT {
    #[new]
    fn new(_size: usize) -> PyResult<Self> {
        let planner = FftPlanner::new();
        Ok(IFFT { planner })
    }

    /// Perform inverse FFT
    fn __call__(&mut self, py: Python, input: Vec<f32>) -> PyResult<String> {
        Ok("IFFT inverse transform (awaiting full tensor integration)".to_string())
    }

    /// Get IFFT info
    fn __repr__(&self) -> String {
        "IFFT(using RustFFT planner)".to_string()
    }
}
