pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod conv_transpose1d;
pub mod conv_transpose2d;
pub mod conv_transpose3d;
pub mod lazy;

pub use conv1d::*;
pub use conv2d::*;
pub use conv3d::*;
pub use conv_transpose1d::*;
pub use conv_transpose2d::*;
pub use conv_transpose3d::*;
pub use lazy::*;

use backend::CpuBackend;
use coeus_nn::modules::convolution::{Conv1D, Conv2D, Conv3D};
use coeus_nn::modules::convolution::{ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};

#[derive(Clone)]
pub enum Conv1DWrapper {
    CpuF32(Conv1D<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Conv1D<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Conv1D<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum Conv2DWrapper {
    CpuF32(Conv2D<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Conv2D<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Conv2D<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum Conv3DWrapper {
    CpuF32(Conv3D<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Conv3D<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Conv3D<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum ConvTranspose1DWrapper {
    CpuF32(ConvTranspose1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(ConvTranspose1d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(ConvTranspose1d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum ConvTranspose2DWrapper {
    CpuF32(ConvTranspose2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(ConvTranspose2d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(ConvTranspose2d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum ConvTranspose3DWrapper {
    CpuF32(ConvTranspose3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(ConvTranspose3d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(ConvTranspose3d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Convolution error: {}", e))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConv1d>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyConv3d>()?;
    m.add_class::<PyConvTranspose1d>()?;
    m.add_class::<PyConvTranspose2d>()?;
    m.add_class::<PyConvTranspose3d>()?;
    m.add_class::<PyLazyConv1d>()?;
    m.add_class::<PyLazyConv2d>()?;
    m.add_class::<PyLazyConv3d>()?;

    let dict = m.dict();
    dict.set_item("Conv1d", m.getattr("Conv1d")?)?;
    dict.set_item("Conv2d", m.getattr("Conv2d")?)?;
    dict.set_item("Conv3d", m.getattr("Conv3d")?)?;
    dict.set_item("ConvTranspose1d", m.getattr("ConvTranspose1d")?)?;
    dict.set_item("ConvTranspose2d", m.getattr("ConvTranspose2d")?)?;
    dict.set_item("ConvTranspose3d", m.getattr("ConvTranspose3d")?)?;
    dict.set_item("LazyConv1d", m.getattr("LazyConv1d")?)?;
    dict.set_item("LazyConv2d", m.getattr("LazyConv2d")?)?;
    dict.set_item("LazyConv3d", m.getattr("LazyConv3d")?)?;

    Ok(())
}
