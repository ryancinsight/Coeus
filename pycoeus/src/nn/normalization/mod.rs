pub mod batchnorm;
pub mod groupnorm;
pub mod instancenorm;
pub mod layernorm;
pub mod lazy;
pub mod rmsnorm;

pub use batchnorm::*;
pub use groupnorm::*;
pub use instancenorm::*;
pub use layernorm::*;
pub use lazy::*;
pub use rmsnorm::*;

use backend::CpuBackend;
use coeus_nn::modules::normalization::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, InstanceNorm, LayerNorm, RMSNorm,
};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;
use dtype::float::{Float32, Float64};

#[derive(Clone)]
pub enum BatchNorm1DWrapper {
    CpuF32(BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(BatchNorm1d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(BatchNorm1d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum BatchNorm2DWrapper {
    CpuF32(BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(BatchNorm2d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(BatchNorm2d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum BatchNorm3DWrapper {
    CpuF32(BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(BatchNorm3d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(BatchNorm3d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum LayerNormWrapper {
    CpuF32(LayerNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LayerNorm<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LayerNorm<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum GroupNormWrapper {
    CpuF32(GroupNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(GroupNorm<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(GroupNorm<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum InstanceNorm1DWrapper {
    CpuF32(InstanceNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(InstanceNorm<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(InstanceNorm<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum InstanceNorm2DWrapper {
    CpuF32(InstanceNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(InstanceNorm<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(InstanceNorm<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum InstanceNorm3DWrapper {
    CpuF32(InstanceNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(InstanceNorm<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(InstanceNorm<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum RMSNormWrapper {
    CpuF32(RMSNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(RMSNorm<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(RMSNorm<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Normalization error: {}", e))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyBatchNorm3d>()?;
    m.add_class::<PyLazyBatchNorm1d>()?;
    m.add_class::<PyLazyBatchNorm2d>()?;
    m.add_class::<PyLazyBatchNorm3d>()?;
    m.add_class::<PyGroupNorm>()?;
    m.add_class::<PyInstanceNorm1d>()?;
    m.add_class::<PyInstanceNorm2d>()?;
    m.add_class::<PyInstanceNorm3d>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyRMSNorm>()?;

    let dict = m.dict();
    dict.set_item("BatchNorm1d", m.getattr("BatchNorm1d")?)?;
    dict.set_item("BatchNorm2d", m.getattr("BatchNorm2d")?)?;
    dict.set_item("BatchNorm3d", m.getattr("BatchNorm3d")?)?;
    dict.set_item("LazyBatchNorm1d", m.getattr("LazyBatchNorm1d")?)?;
    dict.set_item("LazyBatchNorm2d", m.getattr("LazyBatchNorm2d")?)?;
    dict.set_item("LazyBatchNorm3d", m.getattr("LazyBatchNorm3d")?)?;
    dict.set_item("GroupNorm", m.getattr("GroupNorm")?)?;
    dict.set_item("InstanceNorm1d", m.getattr("InstanceNorm1d")?)?;
    dict.set_item("InstanceNorm2d", m.getattr("InstanceNorm2d")?)?;
    dict.set_item("InstanceNorm3d", m.getattr("InstanceNorm3d")?)?;
    dict.set_item("LayerNorm", m.getattr("LayerNorm")?)?;
    dict.set_item("RMSNorm", m.getattr("RMSNorm")?)?;

    Ok(())
}
