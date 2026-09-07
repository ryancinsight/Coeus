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

impl From<coeus_hephaestus::HephaestusBackendError> for WgpuBackendError {
    fn from(source: coeus_hephaestus::HephaestusBackendError) -> Self {
        match source {
            // The historical WGPU backend reported rank rejection for axis
            // reductions under the operation label "reduction"; the bridge
            // labels the same dispatch "reduce". Normalize so the public
            // `Validation(UnsupportedRank)` contract is unchanged for
            // consumers.
            coeus_hephaestus::HephaestusBackendError::Backend(BackendError::UnsupportedRank {
                operation: "reduce",
                rank,
                max_rank,
            }) => Self::Validation(BackendError::UnsupportedRank {
                operation: "reduction",
                rank,
                max_rank,
            }),
            coeus_hephaestus::HephaestusBackendError::Backend(error) => Self::Validation(error),
            coeus_hephaestus::HephaestusBackendError::Device { operation, source } => {
                Self::Dispatch { operation, source }
            }
            // The bridge error is non-exhaustive so future provider categories
            // degrade to a typed backend validation failure rather than panic.
            _ => Self::Validation(BackendError::Storage {
                operation: "hephaestus dispatch",
                reason: "unclassified Hephaestus backend failure".to_string(),
            }),
        }
    }
}

impl WgpuBackendError {
    pub(crate) fn dispatch(operation: &'static str, source: HephaestusError) -> Self {
        Self::Dispatch { operation, source }
    }
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

#[cfg(test)]
mod tests {
    use super::{checked_numel, WgpuBackendError};
    use coeus_core::BackendError;

    #[test]
    fn bridge_rank_rejection_keeps_the_historical_reduction_label() {
        let bridge_error =
            coeus_hephaestus::HephaestusBackendError::Backend(BackendError::UnsupportedRank {
                operation: "reduce",
                rank: 3,
                max_rank: 2,
            });

        let mapped: WgpuBackendError = bridge_error.into();

        assert!(matches!(
            mapped,
            WgpuBackendError::Validation(BackendError::UnsupportedRank {
                operation: "reduction",
                rank: 3,
                max_rank: 2,
            })
        ));
    }

    #[test]
    fn bridge_device_failures_surface_as_typed_dispatch_errors() {
        let bridge_error = coeus_hephaestus::HephaestusBackendError::device(
            "reduce",
            hephaestus_core::HephaestusError::DispatchFailed {
                message: "provider rejected the kernel".to_owned(),
            },
        );

        let mapped: WgpuBackendError = bridge_error.into();

        assert!(matches!(
            mapped,
            WgpuBackendError::Dispatch {
                operation: "reduce",
                source: hephaestus_core::HephaestusError::DispatchFailed { .. },
            }
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
}
