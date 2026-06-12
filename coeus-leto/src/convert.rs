use coeus_core::Layout as CoeusLayout;
use leto::{ArrayView, ArrayViewMut, Layout, LetoError, Result};

/// Convert a coeus dynamic-rank layout to a leto `Layout<N>`.
///
/// Fails if the coeus rank is not exactly `N` (the caller selects `N` via the
/// [`crate::dispatch`] `match`), or if a `usize` stride does not fit in the
/// signed stride leto uses. Zero-copy: only shape/stride metadata is converted.
pub fn to_leto_layout<const N: usize>(layout: &CoeusLayout) -> Result<Layout<N>> {
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() > N {
        return Err(LetoError::StorageError {
            reason: format!("coeus rank {} exceeds leto const rank {N}", shape.len()),
        });
    }

    let mut shape_arr = [1usize; N];
    let mut stride_arr = [0isize; N];
    let pad_len = N - shape.len();
    for i in 0..shape.len() {
        shape_arr[pad_len + i] = shape[i];
        stride_arr[pad_len + i] = if shape[i] == 1 {
            0
        } else {
            isize::try_from(strides[i]).map_err(|_| LetoError::Overflow {
                reason: "coeus stride exceeds isize range",
            })?
        };
    }

    Ok(Layout::new(shape_arr, stride_arr, layout.offset()))
}

/// Build a read-only leto view of rank `N` over a coeus storage slice.
///
/// The layout is validated against the slice length, so an out-of-bounds coeus
/// layout is rejected rather than producing an unsound view.
pub fn to_leto_view<'a, T, const N: usize>(
    layout: &CoeusLayout,
    data: &'a [T],
) -> Result<ArrayView<'a, T, N>> {
    let leto_layout = to_leto_layout::<N>(layout)?;
    ArrayView::try_new(leto_layout, data)
}

/// Build a mutable leto view of rank `N` over a coeus storage slice.
pub fn to_leto_view_mut<'a, T, const N: usize>(
    layout: &CoeusLayout,
    data: &'a mut [T],
) -> Result<ArrayViewMut<'a, T, N>> {
    let leto_layout = to_leto_layout::<N>(layout)?;
    ArrayViewMut::try_new(leto_layout, data)
}
