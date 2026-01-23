//! FFT operations for PyCoeus

use crate::tensor::PyTensor;
use backend::CpuBackend;
use coeus_fft::cpu::CpuFft;
use dtype::complex::Complex32;
use dtype::float::Float32;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use storage::DenseStorage;
use storage::Storage;
use tensor::Tensor;

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<FFT>()?;
    m.add_class::<IFFT>()?;
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(ifft, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    m.add_function(wrap_pyfunction!(irfft, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (input, n=None))]
pub fn fft(input: &PyTensor, n: Option<usize>) -> PyResult<(PyTensor, PyTensor)> {
    let size = n.unwrap_or(input.inner.len());
    let fft = CpuFft::new(size);

    // Convert Float32 tensor to Complex32 storage
    let data = input.inner.as_slice();
    let complex_data: Vec<Complex32> = data
        .iter()
        .take(size)
        .map(|&f: &Float32| Complex32::new(f.get(), 0.0))
        .collect();
    let storage = DenseStorage::from_vec(complex_data, &[size]).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
    })?;

    let result = fft.fft(&storage).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FFT failed: {:?}", e))
    })?;

    decompose_complex(&result)
}

#[pyfunction]
#[pyo3(signature = (input_real, input_imag, n=None))]
pub fn ifft(
    input_real: &PyTensor,
    input_imag: &PyTensor,
    n: Option<usize>,
) -> PyResult<(PyTensor, PyTensor)> {
    let size = n.unwrap_or(input_real.inner.len());
    let fft = CpuFft::new(size);

    let re_data = input_real.inner.as_slice();
    let im_data = input_imag.inner.as_slice();
    let complex_data: Vec<Complex32> = re_data
        .iter()
        .zip(im_data.iter())
        .take(size)
        .map(|(&r, &i): (&Float32, &Float32)| Complex32::new(r.get(), i.get()))
        .collect();

    let storage = DenseStorage::from_vec(complex_data, &[size]).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
    })?;

    let result = fft.ifft(&storage).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("IFFT failed: {:?}", e))
    })?;

    decompose_complex(&result)
}

#[pyfunction]
#[pyo3(signature = (input, n=None))]
pub fn rfft(input: &PyTensor, n: Option<usize>) -> PyResult<(PyTensor, PyTensor)> {
    let size = n.unwrap_or(input.inner.len());
    let fft = CpuFft::new(size);

    let storage = input.inner.storage_ref().clone();
    let result = fft.rfft(&storage).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("RFFT failed: {:?}", e))
    })?;

    decompose_complex(&result)
}

#[pyfunction]
#[pyo3(signature = (input_real, input_imag, n=None))]
pub fn irfft(input_real: &PyTensor, input_imag: &PyTensor, n: Option<usize>) -> PyResult<PyTensor> {
    let re_data = input_real.inner.as_slice();
    let im_data = input_imag.inner.as_slice();

    let input_len = re_data.len();
    let size = n.unwrap_or((input_len - 1) * 2);
    let fft = CpuFft::new(size);

    let complex_data: Vec<Complex32> = re_data
        .iter()
        .zip(im_data.iter())
        .map(|(&r, &i): (&Float32, &Float32)| Complex32::new(r.get(), i.get()))
        .collect();

    let storage = DenseStorage::from_vec(complex_data, &[input_len]).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
    })?;

    let result = fft.irfft(&storage).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("IRFFT failed: {:?}", e))
    })?;

    let out = Tensor::from_storage(result, CpuBackend::new());
    Ok(PyTensor {
        inner: crate::tensor::TensorWrapper::CpuDenseF32(out),
    })
}

fn decompose_complex(result: &DenseStorage<Complex32>) -> PyResult<(PyTensor, PyTensor)> {
    let shape = result.shape().dims().to_vec();
    let data = result.as_slice();
    let mut real_vec = Vec::with_capacity(data.len());
    let mut imag_vec = Vec::with_capacity(data.len());

    for c in data {
        real_vec.push(Float32::new(c.re));
        imag_vec.push(Float32::new(c.im));
    }

    let real_storage = DenseStorage::from_vec(real_vec, &shape).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
    })?;
    let imag_storage = DenseStorage::from_vec(imag_vec, &shape).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
    })?;

    let real_tensor = Tensor::from_storage(real_storage, CpuBackend::<Float32>::new());
    let imag_tensor = Tensor::from_storage(imag_storage, CpuBackend::<Float32>::new());

    Ok((
        PyTensor {
            inner: crate::tensor::TensorWrapper::CpuDenseF32(real_tensor),
        },
        PyTensor {
            inner: crate::tensor::TensorWrapper::CpuDenseF32(imag_tensor),
        },
    ))
}

#[pyclass(name = "FFT", module = "_coeus")]
pub struct FFT {
    size: usize,
}

#[pymethods]
impl FFT {
    #[new]
    fn new(size: usize) -> Self {
        FFT { size }
    }

    fn forward(&self, py: Python, input: Vec<f32>) -> PyResult<Vec<(f32, f32)>> {
        let fft = CpuFft::new(self.size);
        let float_data: Vec<Float32> = input.into_iter().map(Float32::new).collect();
        let storage = DenseStorage::from_vec(float_data, &[self.size]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
        })?;

        let results = py.detach(|| fft.forward(&storage)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("FFT error: {:?}", e))
        })?;

        Ok(results.as_slice().iter().map(|c| (c.re, c.im)).collect())
    }
}

#[pyclass(name = "IFFT", module = "_coeus")]
pub struct IFFT {
    size: usize,
}

#[pymethods]
impl IFFT {
    #[new]
    fn new(size: usize) -> Self {
        IFFT { size }
    }

    fn inverse(&self, py: Python, input: Vec<(f32, f32)>) -> PyResult<Vec<f32>> {
        let fft = CpuFft::new(self.size);
        let complex_data: Vec<Complex32> = input
            .into_iter()
            .map(|(re, im)| Complex32::new(re, im))
            .collect();
        let storage = DenseStorage::from_vec(complex_data, &[self.size]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Storage error: {:?}", e))
        })?;

        let results = py.detach(|| fft.inverse(&storage)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("IFFT error: {:?}", e))
        })?;

        Ok(results
            .as_slice()
            .iter()
            .map(|f: &Float32| f.get())
            .collect())
    }
}
