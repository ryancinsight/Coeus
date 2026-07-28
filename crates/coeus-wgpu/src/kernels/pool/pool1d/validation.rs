use crate::backend::WgpuBackendError;
use crate::kernels::layout::GpuLayoutInfo;
use coeus_core::{BackendError, Layout};

pub(super) fn try_layout(
    operation: &'static str,
    layout: &Layout,
) -> Result<GpuLayoutInfo, WgpuBackendError> {
    if layout.ndim() != 3 {
        return Err(WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation,
                rank: layout.ndim(),
                max_rank: 3,
            },
        ));
    }
    GpuLayoutInfo::try_from_layout(layout).map_err(|error| WgpuBackendError::Layout(error.into()))
}

#[cfg(test)]
mod tests {
    use super::try_layout;
    use crate::backend::WgpuBackendError;
    use coeus_core::{BackendError, Layout};

    #[test]
    fn rejects_non_three_dimensional_layouts() {
        let layout = Layout::new(vec![2, 3].into());

        assert!(matches!(
            try_layout("pool1d", &layout),
            Err(WgpuBackendError::Validation(
                BackendError::UnsupportedRank {
                    operation: "pool1d",
                    rank: 2,
                    max_rank: 3,
                }
            ))
        ));
    }
}
