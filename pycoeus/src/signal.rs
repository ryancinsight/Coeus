//! Python bindings for signal processing functions.
//!
//! This module exposes signal processing functions to Python via PyO3,
//! using TensorWrapper dispatch pattern for backend/dtype flexibility.

use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_signal::stft::STFT;
use coeus_signal::windows::WindowFunc;
use dtype::float::Float32;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hann_window, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_window, m)?)?;
    m.add_function(wrap_pyfunction!(stft, m)?)?;
    Ok(())
}
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
        crate::error::convert_error(format!(
            "signal.hann_window failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor {
        inner: TensorWrapper::CpuDenseF32(result),
    })
}

#[pyfunction]
#[pyo3(signature = (window_length, periodic=true))]
pub fn hamming_window(window_length: usize, periodic: bool) -> PyResult<PyTensor> {
    let result = <Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as WindowFunc<
        CpuBackend<Float32>,
        Float32,
    >>::hamming_window(window_length, periodic)
    .map_err(|e| {
        crate::error::convert_error(format!(
            "signal.hamming_window failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor {
        inner: TensorWrapper::CpuDenseF32(result),
    })
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
    // Extract F32 tensors from TensorWrapper
    let input_tensor = match &input.inner {
        TensorWrapper::CpuDenseF32(t) => t,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "STFT currently only supports float32 tensors",
            ))
        }
    };

    let window_tensor = match window {
        Some(w) => match &w.inner {
            TensorWrapper::CpuDenseF32(t) => Some(t),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Window must be a float32 tensor",
                ))
            }
        },
        None => None,
    };

    let result = <Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as STFT>::stft(
        input_tensor,
        n_fft,
        hop_length,
        win_length,
        window_tensor,
        center,
    )
    .map_err(|e| {
        crate::error::convert_error(format!("signal.stft failed: {:?}", e))
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
        crate::error::convert_error(format!("storage: Storage error: {:?}", e))
    })?;
    let imag_storage = DenseStorage::from_vec(imag_vec, &shape).map_err(|e| {
        crate::error::convert_error(format!("storage: Storage error: {:?}", e))
    })?;

    let real_tensor = Tensor::from_storage(real_storage, CpuBackend::<Float32>::new());
    let imag_tensor = Tensor::from_storage(imag_storage, CpuBackend::<Float32>::new());

    Ok((
        PyTensor {
            inner: TensorWrapper::CpuDenseF32(real_tensor),
        },
        PyTensor {
            inner: TensorWrapper::CpuDenseF32(imag_tensor),
        },
    ))
}
