use crate::convert::to_leto_view;
use coeus_core::Layout as CoeusLayout;
use leto::{LetoError, Result, Storage};

use super::MAX_DISPATCH_RANK;

fn pad_values_n<T: Clone, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    widths: &[(usize, usize)],
    fill: T,
) -> Result<Vec<T>> {
    let input = to_leto_view::<T, N>(a_layout, a)?;
    let width: leto::PadWidth<N> = widths.try_into().map_err(
        |_: std::array::TryFromSliceError| LetoError::ShapeMismatch {
            lhs: vec![N],
            rhs: vec![widths.len()],
        },
    )?;
    let padded = leto::application::pad(&input, width, fill)?;
    Ok(padded.storage().as_slice().to_vec())
}

/// Constant padding of a coeus CPU tensor, dispatched from dynamic rank to the
/// matching monomorphized leto structural kernel. The returned values are
/// C-contiguous in row-major output order.
pub fn pad_values<T: Clone>(
    a_layout: &CoeusLayout,
    a: &[T],
    widths: &[(usize, usize)],
    fill: T,
) -> Result<Vec<T>> {
    match a_layout.ndim() {
        1 => pad_values_n::<T, 1>(a_layout, a, widths, fill),
        2 => pad_values_n::<T, 2>(a_layout, a, widths, fill),
        3 => pad_values_n::<T, 3>(a_layout, a, widths, fill),
        4 => pad_values_n::<T, 4>(a_layout, a, widths, fill),
        5 => pad_values_n::<T, 5>(a_layout, a, widths, fill),
        6 => pad_values_n::<T, 6>(a_layout, a, widths, fill),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn concat_values_n<T: Clone, const N: usize>(
    layouts: &[&CoeusLayout],
    inputs: &[&[T]],
    axis: usize,
) -> Result<Vec<T>> {
    if layouts.len() != inputs.len() {
        return Err(LetoError::StorageError {
            reason: format!(
                "concat received {} layouts for {} inputs",
                layouts.len(),
                inputs.len()
            ),
        });
    }

    let mut views = Vec::with_capacity(inputs.len());
    for (&layout, &input) in layouts.iter().zip(inputs) {
        views.push(to_leto_view::<T, N>(layout, input)?);
    }

    let concatenated = leto::application::concat(&views, axis)?;
    Ok(concatenated.storage().as_slice().to_vec())
}

/// Concatenate coeus CPU tensor values along `axis`, dispatched from dynamic
/// rank to the matching monomorphized leto structural kernel. The returned
/// values are C-contiguous in row-major output order.
pub fn concat_values<T: Clone>(
    layouts: &[&CoeusLayout],
    inputs: &[&[T]],
    axis: usize,
) -> Result<Vec<T>> {
    let Some(first) = layouts.first() else {
        return Err(LetoError::StorageError {
            reason: "concat requires at least one input".to_string(),
        });
    };
    match first.ndim() {
        1 => concat_values_n::<T, 1>(layouts, inputs, axis),
        2 => concat_values_n::<T, 2>(layouts, inputs, axis),
        3 => concat_values_n::<T, 3>(layouts, inputs, axis),
        4 => concat_values_n::<T, 4>(layouts, inputs, axis),
        5 => concat_values_n::<T, 5>(layouts, inputs, axis),
        6 => concat_values_n::<T, 6>(layouts, inputs, axis),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn split_values_n<T: Clone, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<Vec<T>>> {
    let input = to_leto_view::<T, N>(a_layout, a)?;
    let views = leto::application::split(&input, axis, sizes)?;
    Ok(views
        .iter()
        .map(|view| view.to_contiguous().storage().as_slice().to_vec())
        .collect())
}

/// Split coeus CPU tensor values along `axis`, dispatched from dynamic rank to
/// the matching monomorphized leto structural kernel. Each returned chunk is
/// C-contiguous in row-major output order.
pub fn split_values<T: Clone>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<Vec<T>>> {
    match a_layout.ndim() {
        1 => split_values_n::<T, 1>(a_layout, a, axis, sizes),
        2 => split_values_n::<T, 2>(a_layout, a, axis, sizes),
        3 => split_values_n::<T, 3>(a_layout, a, axis, sizes),
        4 => split_values_n::<T, 4>(a_layout, a, axis, sizes),
        5 => split_values_n::<T, 5>(a_layout, a, axis, sizes),
        6 => split_values_n::<T, 6>(a_layout, a, axis, sizes),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn stack_values_n_m<T: Clone, const N: usize, const M: usize>(
    layouts: &[&CoeusLayout],
    inputs: &[&[T]],
    axis: usize,
) -> Result<Vec<T>>
where
    leto::RankMarker<N>: leto::InsertAxis<N, LargerShape = [usize; M]>,
{
    if layouts.len() != inputs.len() {
        return Err(LetoError::StorageError {
            reason: format!(
                "stack received {} layouts for {} inputs",
                layouts.len(),
                inputs.len()
            ),
        });
    }

    let mut views = Vec::with_capacity(inputs.len());
    for (&layout, &input) in layouts.iter().zip(inputs) {
        views.push(to_leto_view::<T, N>(layout, input)?);
    }

    let stacked = leto::application::stack::<T, N, M>(&views, axis)?;
    Ok(stacked.storage().as_slice().to_vec())
}

/// Stack equal-shaped coeus CPU tensor values along a new `axis`, dispatched
/// from dynamic rank to Leto's rank-increasing structural kernel. The returned
/// values are C-contiguous in row-major output order.
pub fn stack_values<T: Clone>(
    layouts: &[&CoeusLayout],
    inputs: &[&[T]],
    axis: usize,
) -> Result<Vec<T>> {
    let Some(first) = layouts.first() else {
        return Err(LetoError::StorageError {
            reason: "stack requires at least one input".to_string(),
        });
    };
    match first.ndim() {
        0 => stack_values_n_m::<T, 0, 1>(layouts, inputs, axis),
        1 => stack_values_n_m::<T, 1, 2>(layouts, inputs, axis),
        2 => stack_values_n_m::<T, 2, 3>(layouts, inputs, axis),
        3 => stack_values_n_m::<T, 3, 4>(layouts, inputs, axis),
        4 => stack_values_n_m::<T, 4, 5>(layouts, inputs, axis),
        5 => stack_values_n_m::<T, 5, 6>(layouts, inputs, axis),
        n => Err(LetoError::StorageError {
            reason: format!(
                "coeus-leto stack dispatch supports input rank 0..{}, got {n}",
                MAX_DISPATCH_RANK - 1
            ),
        }),
    }
}
