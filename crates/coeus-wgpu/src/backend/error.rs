use crate::kernels::layout::GpuLayoutError;
use coeus_core::BackendError;
use hephaestus_core::HephaestusError;
use thiserror::Error;

/// Failure returned by WGPU operation dispatch.
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
    /// The selected operation has no implementation in this backend path.
    #[error("WGPU operation {operation} is unsupported")]
    UnsupportedOperation {
        /// Operation name at the backend boundary.
        operation: &'static str,
    },
    /// The requested dispatch count overflowed while being rounded to a workgroup.
    #[error("WGPU {operation} length {length} overflows workgroup-count arithmetic")]
    WorkgroupCountOverflow {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Logical element count that overflowed the rounding calculation.
        length: usize,
    },
    /// The rounded workgroup count cannot be represented by the WGPU ABI.
    #[error("WGPU {operation} workgroup count {count} exceeds the u32 dispatch ABI")]
    WorkgroupCountOutOfRange {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Unrepresentable workgroup count.
        count: usize,
    },
    /// A kernel parameter cannot be represented by the WGSL `u32` ABI.
    #[error("WGPU {operation} parameter {parameter} value {value} exceeds the u32 ABI")]
    AbiValueOutOfRange {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Kernel parameter that could not be represented.
        parameter: &'static str,
        /// Unrepresentable parameter value.
        value: usize,
    },
    /// A dispatch requires more of a device resource than the adapter exposes.
    #[error(
        "WGPU {operation} requires {requested} {resource}, exceeding the device limit {limit}"
    )]
    ResourceLimitExceeded {
        /// Operation family being dispatched.
        operation: &'static str,
        /// Limited device resource.
        resource: &'static str,
        /// Resource count or byte length requested by the dispatch.
        requested: u64,
        /// Maximum resource count or byte length available.
        limit: u64,
    },
}

/// WGPU layout validation failure retained at the backend boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// The layout rank exceeds the fixed WGSL descriptor.
    #[error("rank {rank} exceeds the WGSL limit {max}")]
    UnsupportedRank {
        /// Requested layout rank.
        rank: usize,
        /// Maximum rank representable by the WGSL descriptor.
        max: usize,
    },
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
    OffsetOutOfRange {
        /// Unrepresentable layout offset.
        value: usize,
    },
    /// A shape dimension exceeds the WGSL `u32` ABI.
    #[error("shape axis {axis} value {value} exceeds the WGSL u32 range")]
    ShapeOutOfRange {
        /// Axis containing the unrepresentable dimension.
        axis: usize,
        /// Unrepresentable shape dimension.
        value: usize,
    },
    /// A stride exceeds the WGSL `u32` ABI.
    #[error("stride axis {axis} value {value} exceeds the WGSL u32 range")]
    StrideOutOfRange {
        /// Axis containing the unrepresentable stride.
        axis: usize,
        /// Unrepresentable stride value.
        value: usize,
    },
    /// A stride cannot be represented by the signed provider layout ABI.
    #[error("stride axis {axis} value {value} exceeds the signed provider range")]
    SignedStrideOutOfRange {
        /// Axis containing the unrepresentable stride.
        axis: usize,
        /// Unrepresentable stride value.
        value: usize,
    },
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

pub(crate) fn checked_workgroup_count(
    operation: &'static str,
    length: usize,
) -> Result<u32, WgpuBackendError> {
    let workgroups = length
        .checked_add(255)
        .ok_or(WgpuBackendError::WorkgroupCountOverflow { operation, length })?
        / 256;
    u32::try_from(workgroups).map_err(|_| WgpuBackendError::WorkgroupCountOutOfRange {
        operation,
        count: workgroups,
    })
}

pub(crate) fn checked_numel(
    operation: &'static str,
    shape: &[usize],
) -> Result<usize, WgpuBackendError> {
    shape.iter().try_fold(1usize, |numel, &dimension| {
        numel
            .checked_mul(dimension)
            .ok_or(WgpuBackendError::Validation(BackendError::Overflow {
                operation,
                reason: "output element-count arithmetic overflow",
            }))
    })
}

pub(crate) fn checked_u32_parameter(
    operation: &'static str,
    parameter: &'static str,
    value: usize,
) -> Result<u32, WgpuBackendError> {
    u32::try_from(value).map_err(|_| WgpuBackendError::AbiValueOutOfRange {
        operation,
        parameter,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::{checked_numel, checked_u32_parameter, checked_workgroup_count, WgpuBackendError};
    use coeus_core::BackendError;

    #[test]
    fn rounds_lengths_without_narrowing_loss() {
        assert!(matches!(checked_workgroup_count("test", 0), Ok(0)));
        assert!(matches!(checked_workgroup_count("test", 256), Ok(1)));
        assert!(matches!(checked_workgroup_count("test", 257), Ok(2)));
    }

    #[test]
    fn rejects_rounding_overflow() {
        assert!(matches!(
            checked_workgroup_count("test", usize::MAX),
            Err(WgpuBackendError::WorkgroupCountOverflow {
                operation: "test",
                length,
            }) if length == usize::MAX
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_counts_outside_the_dispatch_abi() {
        let length = usize::try_from(u64::from(u32::MAX) * 256 + 1)
            .expect("u64 dispatch test value fits usize");
        let expected_count =
            usize::try_from(u64::from(u32::MAX) + 1).expect("u64 dispatch count fits usize");
        assert!(matches!(
            checked_workgroup_count("test", length),
            Err(WgpuBackendError::WorkgroupCountOutOfRange {
                operation: "test",
                count,
            }) if count == expected_count
        ));
    }

    #[test]
    fn rejects_output_element_count_overflow() {
        assert!(matches!(
            checked_numel("reduction", &[usize::MAX, 2]),
            Err(WgpuBackendError::Validation(BackendError::Overflow {
                operation: "reduction",
                reason: "output element-count arithmetic overflow",
            }))
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_kernel_parameters_outside_the_u32_abi() {
        let value = usize::try_from(u64::from(u32::MAX) + 1).expect("test value fits usize");

        assert!(matches!(
            checked_u32_parameter("reduction", "axis length", value),
            Err(WgpuBackendError::AbiValueOutOfRange {
                operation: "reduction",
                parameter: "axis length",
                value: rejected,
            }) if rejected == value
        ));
    }
}
