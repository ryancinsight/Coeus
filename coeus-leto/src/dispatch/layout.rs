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
