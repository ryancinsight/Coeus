use crate::convert::{to_leto_layout, to_leto_view};
use coeus_core::Layout as CoeusLayout;
use leto::{LetoError, Result, Storage};

use super::{shape_n, MAX_DISPATCH_RANK};

fn from_leto_layout<const N: usize>(layout: leto::Layout<N>) -> Result<CoeusLayout> {
    let strides = layout
        .strides
        .iter()
        .map(|&stride| {
            usize::try_from(stride).map_err(|_| LetoError::StorageError {
                reason: format!("coeus layout cannot represent negative stride {stride}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CoeusLayout::from_shape_strides(
        layout.shape.to_vec().into(),
        strides.as_slice().into(),
        layout.offset,
    ))
}

fn contiguous_values_n<T: Clone, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
) -> Result<Vec<T>> {
    let input = to_leto_view::<T, N>(a_layout, a)?;
    Ok(input.to_contiguous().storage().as_slice().to_vec())
}

/// Materialize a coeus CPU tensor view into C-contiguous row-major values,
/// dispatched from dynamic rank to Leto's const-rank view materializer.
///
/// # Examples
///
/// A strided (transposed) view is materialized into row-major order:
///
/// ```
/// use coeus_core::{Layout, Shape, Strides};
/// use coeus_leto::contiguous_values;
///
/// // Storage is a [3,2] matrix laid out row-major: [[1,4],[2,5],[3,6]].
/// let storage = [1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
/// // Interpreted transposed as a [2,3] view (strides swapped).
/// let view = Layout::from_shape_strides(
///     Shape::from([2, 3]),
///     Strides::from_slice(&[1, 2]),
///     0,
/// );
/// let values = contiguous_values(&view, &storage).unwrap();
/// assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn contiguous_values<T: Clone>(a_layout: &CoeusLayout, a: &[T]) -> Result<Vec<T>> {
    match a_layout.ndim() {
        1 => contiguous_values_n::<T, 1>(a_layout, a),
        2 => contiguous_values_n::<T, 2>(a_layout, a),
        3 => contiguous_values_n::<T, 3>(a_layout, a),
        4 => contiguous_values_n::<T, 4>(a_layout, a),
        5 => contiguous_values_n::<T, 5>(a_layout, a),
        6 => contiguous_values_n::<T, 6>(a_layout, a),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn reshape_layout_n_m<const N: usize, const M: usize>(
    layout: &CoeusLayout,
    shape: &[usize],
) -> Result<CoeusLayout> {
    let reshaped = to_leto_layout::<N>(layout)?.reshape::<M>(shape_n::<M>(shape)?)?;
    from_leto_layout(reshaped)
}

fn reshape_layout_n<const N: usize>(layout: &CoeusLayout, shape: &[usize]) -> Result<CoeusLayout> {
    match shape.len() {
        1 => reshape_layout_n_m::<N, 1>(layout, shape),
        2 => reshape_layout_n_m::<N, 2>(layout, shape),
        3 => reshape_layout_n_m::<N, 3>(layout, shape),
        4 => reshape_layout_n_m::<N, 4>(layout, shape),
        5 => reshape_layout_n_m::<N, 5>(layout, shape),
        6 => reshape_layout_n_m::<N, 6>(layout, shape),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// Zero-copy reshape of coeus layout metadata, dispatched to Leto's const-rank
/// layout validator. This preserves the storage offset and returns a dynamic
/// coeus layout for the requested shape.
///
/// # Examples
///
/// Reshape a contiguous rank-1 layout into a rank-2 layout; the resulting
/// strides are recomputed for the new shape:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::reshape_layout;
///
/// let layout = Layout::new([6].into());
/// let reshaped = reshape_layout(&layout, &[2, 3]).unwrap();
/// assert_eq!(reshaped.shape(), &[2, 3]);
/// assert_eq!(reshaped.strides(), &[3, 1]);
/// assert_eq!(reshaped.offset(), 0);
/// ```
///
/// A reshape that breaks contiguity is rejected:
///
/// ```
/// use coeus_core::{Layout, Shape, Strides};
/// use coeus_leto::reshape_layout;
///
/// // A [3,2] tensor with transposed strides cannot reshape to [6].
/// let transposed = Layout::from_shape_strides(
///     Shape::from([3, 2]),
///     Strides::from_slice(&[1, 3]),
///     0,
/// );
/// assert!(reshape_layout(&transposed, &[6]).is_err());
/// ```
pub fn reshape_layout(layout: &CoeusLayout, shape: &[usize]) -> Result<CoeusLayout> {
    match layout.ndim() {
        1 => reshape_layout_n::<1>(layout, shape),
        2 => reshape_layout_n::<2>(layout, shape),
        3 => reshape_layout_n::<3>(layout, shape),
        4 => reshape_layout_n::<4>(layout, shape),
        5 => reshape_layout_n::<5>(layout, shape),
        6 => reshape_layout_n::<6>(layout, shape),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn permute_layout_n<const N: usize>(layout: &CoeusLayout, axes: &[usize]) -> Result<CoeusLayout> {
    let permuted = to_leto_layout::<N>(layout)?.transpose(shape_n::<N>(axes)?)?;
    from_leto_layout(permuted)
}

/// Zero-copy permutation of coeus layout metadata, dispatched to Leto's
/// const-rank layout validator.
///
/// # Examples
///
/// Permute a [2,3,4] layout with axes `[2,0,1]`: the new shape and strides
/// are the selected reorderings of the originals:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::permute_layout;
///
/// let layout = Layout::new([2, 3, 4].into()); // strides [12, 4, 1]
/// let permuted = permute_layout(&layout, &[2, 0, 1]).unwrap();
/// assert_eq!(permuted.shape(), &[4, 2, 3]);
/// assert_eq!(permuted.strides(), &[1, 12, 4]);
/// assert_eq!(permuted.offset(), 0);
/// ```
///
/// A permutation with a repeated axis is rejected:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::permute_layout;
///
/// let layout = Layout::new([2, 3, 4].into());
/// assert!(permute_layout(&layout, &[0, 0, 1]).is_err());
/// ```
pub fn permute_layout(layout: &CoeusLayout, axes: &[usize]) -> Result<CoeusLayout> {
    match layout.ndim() {
        1 => permute_layout_n::<1>(layout, axes),
        2 => permute_layout_n::<2>(layout, axes),
        3 => permute_layout_n::<3>(layout, axes),
        4 => permute_layout_n::<4>(layout, axes),
        5 => permute_layout_n::<5>(layout, axes),
        6 => permute_layout_n::<6>(layout, axes),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn broadcast_layout_n_m<const N: usize, const M: usize>(
    layout: &CoeusLayout,
    target_shape: &[usize],
) -> Result<CoeusLayout> {
    let broadcasted = to_leto_layout::<N>(layout)?.broadcast::<M>(shape_n::<M>(target_shape)?)?;
    from_leto_layout(broadcasted)
}

fn broadcast_layout_n<const N: usize>(
    layout: &CoeusLayout,
    target_shape: &[usize],
) -> Result<CoeusLayout> {
    match target_shape.len() {
        0 => broadcast_layout_n_m::<N, 0>(layout, target_shape),
        1 => broadcast_layout_n_m::<N, 1>(layout, target_shape),
        2 => broadcast_layout_n_m::<N, 2>(layout, target_shape),
        3 => broadcast_layout_n_m::<N, 3>(layout, target_shape),
        4 => broadcast_layout_n_m::<N, 4>(layout, target_shape),
        5 => broadcast_layout_n_m::<N, 5>(layout, target_shape),
        6 => broadcast_layout_n_m::<N, 6>(layout, target_shape),
        n => Err(LetoError::StorageError {
            reason: format!(
                "coeus-leto broadcast dispatch supports rank 0..={MAX_DISPATCH_RANK}, got {n}"
            ),
        }),
    }
}

/// Zero-copy broadcast of coeus layout metadata to `target_shape`, dispatched to
/// Leto's const-rank layout validator.
///
/// # Examples
///
/// Broadcast a `[1,3]` row to `[2,3]`: the broadcast dimension takes a zero
/// stride so the row repeats without copying data:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::broadcast_layout;
///
/// let row = Layout::new([1, 3].into()); // strides [3, 1]
/// let broadcasted = broadcast_layout(&row, &[2, 3]).unwrap();
/// assert_eq!(broadcasted.shape(), &[2, 3]);
/// assert_eq!(broadcasted.strides(), &[0, 1]);
/// assert_eq!(broadcasted.offset(), 0);
/// ```
pub fn broadcast_layout(layout: &CoeusLayout, target_shape: &[usize]) -> Result<CoeusLayout> {
    match layout.ndim() {
        0 => broadcast_layout_n::<0>(layout, target_shape),
        1 => broadcast_layout_n::<1>(layout, target_shape),
        2 => broadcast_layout_n::<2>(layout, target_shape),
        3 => broadcast_layout_n::<3>(layout, target_shape),
        4 => broadcast_layout_n::<4>(layout, target_shape),
        5 => broadcast_layout_n::<5>(layout, target_shape),
        6 => broadcast_layout_n::<6>(layout, target_shape),
        n => Err(LetoError::StorageError {
            reason: format!(
                "coeus-leto broadcast dispatch supports rank 0..={MAX_DISPATCH_RANK}, got {n}"
            ),
        }),
    }
}

/// Compute the NumPy/PyTorch-style broadcast shape for two dynamic-rank coeus
/// shapes. The result is the target shape accepted by [`broadcast_layout`] for
/// both inputs.
///
/// # Examples
///
/// A column `[2,1]` and a row `[1,3]` broadcast to `[2,3]`:
///
/// ```
/// use coeus_leto::broadcast_shape;
///
/// assert_eq!(broadcast_shape(&[2, 1], &[1, 3]).unwrap(), vec![2, 3]);
/// ```
///
/// Incompatible dimensions are rejected:
///
/// ```
/// use coeus_leto::broadcast_shape;
///
/// assert!(broadcast_shape(&[2, 2], &[3, 2]).is_err());
/// ```
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    if rank > MAX_DISPATCH_RANK {
        return Err(LetoError::StorageError {
            reason: format!(
                "coeus-leto broadcast dispatch supports rank 0..={MAX_DISPATCH_RANK}, got {rank}"
            ),
        });
    }

    let mut out = vec![0; rank];
    for index in 0..rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(index + 1)
            .map_or(1, |position| lhs[position]);
        let rhs_dim = rhs
            .len()
            .checked_sub(index + 1)
            .map_or(1, |position| rhs[position]);
        out[rank - 1 - index] = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return Err(LetoError::IncompatibleBroadcast {
                from: lhs.to_vec(),
                to: rhs.to_vec(),
            });
        };
    }

    Ok(out)
}
