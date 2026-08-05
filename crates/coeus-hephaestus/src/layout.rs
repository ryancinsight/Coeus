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

/// Map a logical Coeus axis to the corresponding left-padded provider axis.
pub(crate) fn ranked_axis<const N: usize>(
    operation: &'static str,
    layout: &Layout,
    axis: usize,
) -> Result<usize, BackendError> {
    let rank = layout.ndim();
    if rank > N {
        return Err(BackendError::UnsupportedRank {
            operation,
            rank,
            max_rank: N,
        });
    }
    if axis >= rank {
        return Err(BackendError::AxisOutOfRange {
            operation,
            axis,
            rank,
        });
    }
    Ok(axis + (N - rank))
}

#[cfg(test)]
mod tests {
    use super::ranked_axis;
    use coeus_core::Layout;

    #[test]
    fn ranked_axis_maps_left_padded_rank() {
        let vector = Layout::new([6].into());
        let matrix = Layout::new([2, 3].into());

        assert_eq!(ranked_axis::<2>("reduce", &vector, 0), Ok(1));
        assert_eq!(ranked_axis::<2>("reduce", &matrix, 0), Ok(0));
        assert_eq!(ranked_axis::<2>("reduce", &matrix, 1), Ok(1));
    }
}
