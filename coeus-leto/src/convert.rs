use coeus_core::Layout as CoeusLayout;
use leto::{ArrayView, ArrayViewMut, Layout, LetoError, Result};

/// Convert a coeus dynamic-rank layout to a leto `Layout<N>`.
///
/// Fails if the coeus rank exceeds `N` (the caller selects `N` via the
/// [`crate::dispatch`] `match`), or if a `usize` stride does not fit in the
/// signed stride leto uses. A rank smaller than `N` is left-padded with size-1
/// dimensions. Zero-copy: only shape/stride metadata is converted.
///
/// # Examples
///
/// Convert a contiguous rank-2 coeus layout to a leto `Layout<2>`. Only
/// metadata is transferred, so the conversion is zero-cost:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::to_leto_layout;
///
/// let layout = Layout::new([2, 3].into());
/// let leto_layout = to_leto_layout::<2>(&layout).unwrap();
/// assert_eq!(leto_layout.shape, [2, 3]);
/// assert_eq!(leto_layout.strides, [3, 1]);
/// assert_eq!(leto_layout.offset, 0);
/// ```
///
/// A coeus layout whose rank exceeds `N` is rejected:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::to_leto_layout;
///
/// let layout = Layout::new([2, 3].into());
/// assert!(to_leto_layout::<1>(&layout).is_err());
/// ```
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
///
/// # Examples
///
/// Build a rank-2 view over a coeus storage slice and read a logical element
/// through the leto view:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::to_leto_view;
///
/// let layout = Layout::new([2, 3].into());
/// let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let view = to_leto_view::<f64, 2>(&layout, &data).unwrap();
/// assert_eq!(view.shape(), [2, 3]);
/// assert_eq!(*view.get([1, 2]).unwrap(), 6.0);
/// ```
///
/// A layout whose footprint exceeds the slice length is rejected:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::to_leto_view;
///
/// let layout = Layout::new([2, 3].into());
/// let short = [0.0_f64; 5]; // 5 < 6 elements
/// assert!(to_leto_view::<f64, 2>(&layout, &short).is_err());
/// ```
pub fn to_leto_view<'a, T, const N: usize>(
    layout: &CoeusLayout,
    data: &'a [T],
) -> Result<ArrayView<'a, T, N>> {
    let leto_layout = to_leto_layout::<N>(layout)?;
    ArrayView::try_new(leto_layout, data)
}

/// Build a mutable leto view of rank `N` over a coeus storage slice.
///
/// The layout is validated against the slice length, so an out-of-bounds coeus
/// layout is rejected rather than producing an unsound view.
///
/// # Examples
///
/// Build a mutable rank-1 view over a coeus storage slice and write through
/// it, observing the change in the underlying buffer:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::to_leto_view_mut;
///
/// let layout = Layout::new([3].into());
/// let mut data = [0.0_f64; 3];
/// let mut view = to_leto_view_mut::<f64, 1>(&layout, &mut data).unwrap();
/// *view.get_mut([1]).unwrap() = 9.0;
/// drop(view);
/// assert_eq!(data, [0.0, 9.0, 0.0]);
/// ```
pub fn to_leto_view_mut<'a, T, const N: usize>(
    layout: &CoeusLayout,
    data: &'a mut [T],
) -> Result<ArrayViewMut<'a, T, N>> {
    let leto_layout = to_leto_layout::<N>(layout)?;
    ArrayViewMut::try_new(leto_layout, data)
}
