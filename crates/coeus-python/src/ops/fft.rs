use crate::tensor::PyTensor;
use coeus_core::Complex;
use pyo3::prelude::*;

/// Python-exposed complex tensor returned by FFT operations.
#[pyclass(name = "ComplexTensor")]
#[derive(Clone)]
pub struct PyComplexTensor {
    pub(crate) inner: coeus_autograd::Var<Complex<f64>>,
}

#[pymethods]
impl PyComplexTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.tensor.shape().to_vec()
    }

    #[getter]
    fn data(&self) -> Vec<(f64, f64)> {
        self.inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    #[getter]
    fn real(&self) -> Vec<f64> {
        self.inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|c| c.re)
            .collect()
    }

    #[getter]
    fn imag(&self) -> Vec<f64> {
        self.inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|c| c.im)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("ComplexTensor(shape={:?})", self.shape())
    }
}

/// Apollo-backed 1-D FFT.
#[pyfunction]
#[pyo3(name = "fft")]
pub fn fft_1d(input: &PyTensor, py: Python<'_>) -> PyComplexTensor {
    let inner = py.allow_threads(|| coeus_fft::fft_1d_var(&input.inner));
    PyComplexTensor { inner }
}

/// Apollo-backed 1-D inverse FFT.
#[pyfunction]
#[pyo3(name = "ifft")]
pub fn ifft_1d(input: &PyComplexTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_fft::ifft_1d_var(&input.inner));
    PyTensor::from_var(inner)
}

/// Sum of squared FFT magnitudes with gradient to the real input.
#[pyfunction]
#[pyo3(name = "fft_energy")]
pub fn fft_energy(input: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_fft::fft_energy(&input.inner));
    PyTensor::from_var(inner)
}
