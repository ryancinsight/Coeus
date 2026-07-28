use crate::backend::WgpuBackendError;
use crate::kernels::layout::GpuLayoutInfo;
use coeus_core::{BackendError, Layout};

pub(super) fn try_layout(
    operation: &'static str,
    layout: &Layout,
    expected_rank: usize,
) -> Result<GpuLayoutInfo, WgpuBackendError> {
    if layout.ndim() != expected_rank {
        return Err(WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation,
                rank: layout.ndim(),
                max_rank: expected_rank,
            },
        ));
    }
    GpuLayoutInfo::try_from_layout(layout).map_err(|error| WgpuBackendError::Layout(error.into()))
}

pub(super) fn parameter(value: usize, name: &str) -> Result<u32, WgpuBackendError> {
    u32::try_from(value).map_err(|_| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation: "pool",
            reason: format!("{name} value {value} exceeds the WGSL u32 ABI"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{parameter, try_layout};
    use crate::backend::WgpuBackendError;
    use coeus_core::{BackendError, Layout};

    #[test]
    fn rejects_layouts_with_the_wrong_rank() {
        let layout = Layout::new(vec![2, 3].into());

        assert!(matches!(
            try_layout("pool2d", &layout, 4),
            Err(WgpuBackendError::Validation(
                BackendError::UnsupportedRank {
                    operation: "pool2d",
                    rank: 2,
                    max_rank: 4,
                }
            ))
        ));
    }

    #[test]
    fn accepts_parameters_representable_by_the_wgsl_abi() {
        assert!(matches!(parameter(17, "stride"), Ok(17)));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_parameters_outside_the_wgsl_abi() {
        let value = usize::try_from(u64::from(u32::MAX) + 1).expect("u64 value fits usize");

        assert!(matches!(
            parameter(value, "stride"),
            Err(WgpuBackendError::Validation(BackendError::Storage {
                operation: "pool",
                reason,
            })) if reason == "stride value 4294967296 exceeds the WGSL u32 ABI"
        ));
    }
}
