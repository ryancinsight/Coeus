//! FFT operations for PyCoeus audio processing

use pyo3::prelude::*;
use pyo3::pyclass;
use coeus_audio::Fft;

/// FFT operation for audio processing
#[pyclass(name = "FFT", module = "_coeus")]
#[derive(Clone)]
pub struct FFT {
    /// Internal FFT processor
    fft: Fft,
}

#[pymethods]
impl FFT {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let fft = Fft::new(size).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        Ok(FFT { fft })
    }

    /// Get the FFT size
    #[must_use]
    fn size(&self) -> usize {
        self.fft.size()
    }

    /// Perform basic forward FFT on Python list
    /// This provides immediate functionality for testing and validation
    fn forward(&mut self, py: Python, input: Vec<f32>) -> PyResult<Vec<(f32, f32)>> {
        let result = py.allow_threads(|| {
            self.fft.forward_real_simple(&input)
        }).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

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
            self.fft.inverse_complex_simple(&complex_input)
        }).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        Ok(result)
    }

    /// Get FFT info
    fn __repr__(&self) -> String {
        format!("FFT(size={})", self.fft.size())
    }
}

/// IFFT operation for audio processing
#[pyclass(name = "IFFT", module = "_coeus")]
#[derive(Clone)]
pub struct IFFT {
    /// Internal FFT processor
    fft: Fft,
}

#[pymethods]
impl IFFT {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let fft = Fft::new(size).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        Ok(IFFT { fft })
    }

    /// Perform inverse FFT
    fn __call__(&mut self, py: Python, input: Vec<f32>) -> PyResult<String> {
        Ok(format!("IFFT inverse transform for size {} (awaiting full tensor integration)", self.fft.size()))
    }

    /// Get IFFT info
    fn __repr__(&self) -> String {
        format!("IFFT(size={})", self.fft.size())
    }
}
