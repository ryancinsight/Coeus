//! FFT operations for PyCoeus audio processing

use coeus_fft::cpu::CpuFft;
use dtype::complex::Complex32;
use dtype::float::Float32;
use pyo3::prelude::*;
use pyo3::pyclass;
use storage::DenseStorage;
use storage::Storage;

/// FFT operation for audio processing
#[pyclass(name = "FFT", module = "_coeus")]
pub struct FFT {
    processor: CpuFft,
}

#[pymethods]
impl FFT {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let processor = CpuFft::new(size);
        Ok(FFT { processor })
    }

    /// Get a description of the FFT
    #[must_use]
    fn description(&self) -> String {
        "Coeus-FFT based processor".to_string()
    }

    /// Perform forward FFT on Python list
    fn forward(&mut self, py: Python, input: Vec<f32>) -> PyResult<Vec<(f32, f32)>> {
        let float_data: Vec<Float32> = input.into_iter().map(Float32::new).collect();
        let size = float_data.len();
        let storage = DenseStorage::from_vec(float_data, &[size]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
        })?;

        // Temporarily use CpuFft until we have a unified dispatcher
        let results = py
            .detach(|| self.processor.forward(&storage))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FFT error: {:?}", e))
            })?;

        Ok(results.as_slice().iter().map(|c| (c.re, c.im)).collect())
    }

    /// Perform inverse FFT on complex coefficients
    fn inverse(&mut self, py: Python, input: Vec<(f32, f32)>) -> PyResult<Vec<f32>> {
        let complex_data: Vec<Complex32> = input
            .into_iter()
            .map(|(re, im)| Complex32::new(re, im))
            .collect();
        let size = complex_data.len();
        let storage = DenseStorage::from_vec(complex_data, &[size]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
        })?;

        let results = py
            .detach(|| self.processor.inverse(&storage))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FFT error: {:?}", e))
            })?;

        Ok(results.as_slice().iter().map(|f| f.get()).collect())
    }

    /// Get FFT info
    fn __repr__(&self) -> String {
        "FFT(using Coeus-FFT)".to_string()
    }
}

/// IFFT operation for audio processing
#[pyclass(name = "IFFT", module = "_coeus")]
pub struct IFFT {
    processor: CpuFft,
}

#[pymethods]
impl IFFT {
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let processor = CpuFft::new(size);
        Ok(IFFT { processor })
    }

    /// Perform inverse FFT via call
    fn __call__(&mut self, py: Python, input: Vec<(f32, f32)>) -> PyResult<Vec<f32>> {
        self.inverse(py, input)
    }

    /// Perform inverse FFT
    fn inverse(&mut self, py: Python, input: Vec<(f32, f32)>) -> PyResult<Vec<f32>> {
        let complex_data: Vec<Complex32> = input
            .into_iter()
            .map(|(re, im)| Complex32::new(re, im))
            .collect();
        let size = complex_data.len();
        let storage = DenseStorage::from_vec(complex_data, &[size]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
        })?;

        let results = py
            .detach(|| self.processor.inverse(&storage))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FFT error: {:?}", e))
            })?;

        Ok(results.as_slice().iter().map(|f| f.get()).collect())
    }

    /// Get IFFT info
    fn __repr__(&self) -> String {
        "IFFT(using Coeus-FFT)".to_string()
    }
}
