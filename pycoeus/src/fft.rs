use crate::tensor::PyTensor;
use coeus_fft::{fft, fft2, ifft, ifft2, irfft, rfft, Norm};
use pyo3::prelude::*;

/// Fast Fourier Transform
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct FFT;

#[pymethods]
impl FFT {
    #[new]
    pub fn new() -> Self {
        FFT
    }

    /// Compute 1D FFT of input tensor
    #[pyo3(signature = (input, n=None, dim=None, norm=None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        n: Option<usize>,
        dim: Option<i32>,
        norm: Option<String>,
    ) -> PyResult<PyTensor> {
        let norm_mode = match norm.as_deref() {
            Some("ortho") => Norm::Ortho,
            Some("forward") => Norm::Forward,
            Some("backward") => Norm::Backward,
            _ => Norm::None,
        };

        let result = fft(&input.tensor, n, dim, Some(norm_mode)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("FFT failed: {:?}", e))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Compute 2D FFT of input tensor
    #[pyo3(signature = (input, s=None, dim=None, norm=None))]
    pub fn fft2(
        &self,
        input: &PyTensor,
        s: Option<Vec<usize>>,
        dim: Option<Vec<i32>>,
        norm: Option<String>,
    ) -> PyResult<PyTensor> {
        let norm_mode = match norm.as_deref() {
            Some("ortho") => Norm::Ortho,
            Some("forward") => Norm::Forward,
            Some("backward") => Norm::Backward,
            _ => Norm::None,
        };

        let dim_array = dim.as_deref();
        let s_array = s.as_deref();

        let result = fft2(&input.tensor, s_array, dim_array, Some(norm_mode)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("FFT2 failed: {:?}", e))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Compute real FFT of input tensor
    #[pyo3(signature = (input, n=None, dim=None, norm=None))]
    pub fn rfft(
        &self,
        input: &PyTensor,
        n: Option<usize>,
        dim: Option<i32>,
        norm: Option<String>,
    ) -> PyResult<PyTensor> {
        let norm_mode = match norm.as_deref() {
            Some("ortho") => Norm::Ortho,
            Some("forward") => Norm::Forward,
            Some("backward") => Norm::Backward,
            _ => Norm::None,
        };

        let result = rfft(&input.tensor, n, dim, Some(norm_mode)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("RFFT failed: {:?}", e))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }
}

impl Default for FFT {
    fn default() -> Self {
        Self::new()
    }
}

/// Inverse Fast Fourier Transform
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct IFFT;

#[pymethods]
impl IFFT {
    #[new]
    pub fn new() -> Self {
        IFFT
    }

    /// Compute inverse 1D FFT of input tensor
    #[pyo3(signature = (input, n=None, dim=None, norm=None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        n: Option<usize>,
        dim: Option<i32>,
        norm: Option<String>,
    ) -> PyResult<PyTensor> {
        let norm_mode = match norm.as_deref() {
            Some("ortho") => Norm::Ortho,
            Some("forward") => Norm::Forward,
            Some("backward") => Norm::Backward,
            _ => Norm::None,
        };

        let result = ifft(&input.tensor, n, dim, Some(norm_mode)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("IFFT failed: {:?}", e))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Compute inverse 2D FFT of input tensor
    #[pyo3(signature = (input, s=None, dim=None, norm=None))]
    pub fn ifft2(
        &self,
        input: &PyTensor,
        s: Option<Vec<usize>>,
        dim: Option<Vec<i32>>,
        norm: Option<String>,
    ) -> PyResult<PyTensor> {
        let norm_mode = match norm.as_deref() {
            Some("ortho") => Norm::Ortho,
            Some("forward") => Norm::Forward,
            Some("backward") => Norm::Backward,
            _ => Norm::None,
        };

        let dim_array = dim.as_deref();
        let s_array = s.as_deref();

        let result = ifft2(&input.tensor, s_array, dim_array, Some(norm_mode)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("IFFT2 failed: {:?}", e))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }

    /// Compute inverse real FFT of input tensor
    #[pyo3(signature = (input, n=None, dim=None, norm=None))]
    pub fn irfft(
        &self,
        input: &PyTensor,
        n: Option<usize>,
        dim: Option<i32>,
        norm: Option<String>,
    ) -> PyResult<PyTensor> {
        let norm_mode = match norm.as_deref() {
            Some("ortho") => Norm::Ortho,
            Some("forward") => Norm::Forward,
            Some("backward") => Norm::Backward,
            _ => Norm::None,
        };

        let result = irfft(&input.tensor, n, dim, Some(norm_mode)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("IRFFT failed: {:?}", e))
        })?;

        Ok(PyTensor {
            tensor: result,
            requires_grad: input.requires_grad,
            device: input.device.clone(),
        })
    }
}

impl Default for IFFT {
    fn default() -> Self {
        Self::new()
    }
}
