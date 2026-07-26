use coeus_core::{BackendError, Layout};
use leto::Layout as LetoLayout;

/// Convert a dynamic Coeus layout to a left-padded fixed-rank Leto layout.
pub(crate) fn ranked<const N: usize>(
    operation: &'static str,
    layout: &Layout,
) -> Result<LetoLayout<N>, BackendError> {
    let rank = layout.ndim();
    if rank > N {
        return Err(BackendError::UnsupportedRank {
            operation,
            rank,
            max_rank: N,
        });
    }
    if layout.shape().len() != layout.strides().len() {
        return Err(BackendError::LayoutRankMismatch {
            operation,
            lhs: layout.shape().len(),
            rhs: layout.strides().len(),
        });
    }

    let padding = N - rank;
    let mut shape = [1_usize; N];
    let mut strides = [0_isize; N];
    for (index, (&extent, &stride)) in layout.shape().iter().zip(layout.strides()).enumerate() {
        shape[padding + index] = extent;
        strides[padding + index] = isize::try_from(stride).map_err(|_| BackendError::Overflow {
            operation,
            reason: "layout stride exceeds isize range",
        })?;
    }

    Ok(LetoLayout::new(shape, strides, layout.offset()))
}
