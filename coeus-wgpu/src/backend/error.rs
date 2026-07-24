use crate::kernels::layout::GpuLayoutError;
use coeus_core::BackendError;
use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Failure returned by WGPU elementwise dispatch.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WgpuBackendError {
    /// The operation rejected its tensor shapes or backend-independent layout.
    #[error("WGPU operation validation failed: {0}")]
    Validation(#[from] BackendError),
    /// Layout metadata cannot be represented by the WGSL ABI.
    #[error("WGPU layout validation failed: {0}")]
    Layout(#[from] LayoutError),
    /// The provider rejected a submitted kernel dispatch.
    #[error("WGPU {operation} dispatch failed: {source}")]
    Dispatch {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Provider failure.
        #[source]
        source: HephaestusError,
    },
}

/// WGPU layout validation failure retained at the backend boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// The layout rank exceeds the fixed WGSL descriptor.
    #[error("rank {rank} exceeds the WGSL limit {max}")]
    UnsupportedRank { rank: usize, max: usize },
    /// Shape and stride metadata have different ranks.
    #[error("shape rank {shape_rank} differs from stride rank {stride_rank}")]
    RankMismatch {
        /// Shape rank.
        shape_rank: usize,
        /// Stride rank.
        stride_rank: usize,
    },
    /// The base offset exceeds the WGSL `u32` ABI.
    #[error("offset {value} exceeds the WGSL u32 range")]
    OffsetOutOfRange { value: usize },
    /// A shape dimension exceeds the WGSL `u32` ABI.
    #[error("shape axis {axis} value {value} exceeds the WGSL u32 range")]
    ShapeOutOfRange { axis: usize, value: usize },
    /// A stride exceeds the WGSL `u32` ABI.
    #[error("stride axis {axis} value {value} exceeds the WGSL u32 range")]
    StrideOutOfRange { axis: usize, value: usize },
}

impl From<GpuLayoutError> for LayoutError {
    fn from(error: GpuLayoutError) -> Self {
        match error {
            GpuLayoutError::UnsupportedRank { rank, max } => Self::UnsupportedRank { rank, max },
            GpuLayoutError::RankMismatch {
                shape_rank,
                stride_rank,
            } => Self::RankMismatch {
                shape_rank,
                stride_rank,
            },
            GpuLayoutError::OffsetOutOfRange { value } => Self::OffsetOutOfRange { value },
            GpuLayoutError::ShapeOutOfRange { axis, value } => {
                Self::ShapeOutOfRange { axis, value }
            }
            GpuLayoutError::StrideOutOfRange { axis, value } => {
                Self::StrideOutOfRange { axis, value }
            }
        }
    }
}

impl WgpuBackendError {
    pub(crate) fn dispatch(operation: &'static str, source: HephaestusError) -> Self {
        Self::Dispatch { operation, source }
    }

    pub(crate) fn validate_layout(layout: &coeus_core::Layout) -> Result<(), Self> {
        crate::kernels::layout::GpuLayoutInfo::try_from_layout(layout)
            .map(|_| ())
            .map_err(|error| Self::Layout(error.into()))
    }
}
