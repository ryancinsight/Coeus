use crate::tensor::PyTensor;
use backend::CpuBackend;
use coeus_signal::stft::STFT;
use coeus_signal::windows::WindowFunc;
use dtype::float::Float32;
use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult};
use storage::DenseStorage;
use tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (window_length, periodic=true))]
pub fn hann_window(window_length: usize, periodic: bool) -> PyResult<PyTensor> {
    let result = <Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as WindowFunc<
        CpuBackend<Float32>,
        Float32,
    >>::hann_window(window_length, periodic)
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "signal.hann_window failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (window_length, periodic=true))]
pub fn hamming_window(window_length: usize, periodic: bool) -> PyResult<PyTensor> {
    let result = <Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as WindowFunc<
        CpuBackend<Float32>,
        Float32,
    >>::hamming_window(window_length, periodic)
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "signal.hamming_window failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, n_fft, hop_length=None, win_length=None, window=None, center=true))]
pub fn stft(
    input: &PyTensor,
    n_fft: usize,
    hop_length: Option<usize>,
    win_length: Option<usize>,
    window: Option<&PyTensor>,
    center: bool,
) -> PyResult<(PyTensor, PyTensor)> {
    let result = <Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as STFT>::stft(
        &input.inner,
        n_fft,
        hop_length,
        win_length,
        window.map(|w| &w.inner),
        center,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("signal.stft failed: {:?}", e))
    })?;

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
        PyTensor { inner: real_tensor },
        PyTensor { inner: imag_tensor },
    ))
}
