use coeus_core::{BackendError, Layout};
use leto::Layout as LetoLayout;

/// Convert a Coeus layout to the rank-2 layout consumed by the current
/// Hephaestus reduction and scan kernels.
pub(crate) fn rank_two(
    operation: &'static str,
    layout: &Layout,
) -> Result<LetoLayout<2>, BackendError> {
    let [rows, columns] = layout.shape() else {
        return Err(BackendError::UnsupportedRank {
            operation,
            rank: layout.ndim(),
            max_rank: 2,
        });
    };
    let [row_stride, column_stride] = layout.strides() else {
        return Err(BackendError::LayoutRankMismatch {
            operation,
            lhs: layout.shape().len(),
            rhs: layout.strides().len(),
        });
    };
    let row_stride = isize::try_from(*row_stride).map_err(|_| BackendError::Overflow {
        operation,
        reason: "row stride exceeds isize range",
    })?;
    let column_stride = isize::try_from(*column_stride).map_err(|_| BackendError::Overflow {
        operation,
        reason: "column stride exceeds isize range",
    })?;
    Ok(LetoLayout::new(
        [*rows, *columns],
        [row_stride, column_stride],
        layout.offset(),
    ))
}
