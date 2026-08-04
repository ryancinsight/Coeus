use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout as CoeusLayout;
use leto::{ArrayView, LetoError, Result};

use super::MAX_DISPATCH_RANK;

enum RotateHalfViews<'a, T> {
    Rank1(ArrayView<'a, T, 1>, ArrayView<'a, T, 1>),
    Rank2(ArrayView<'a, T, 2>, ArrayView<'a, T, 2>),
    Rank3(ArrayView<'a, T, 3>, ArrayView<'a, T, 3>),
    Rank4(ArrayView<'a, T, 4>, ArrayView<'a, T, 4>),
    Rank5(ArrayView<'a, T, 5>, ArrayView<'a, T, 5>),
    Rank6(ArrayView<'a, T, 6>, ArrayView<'a, T, 6>),
}

/// Validated zero-copy input views for one rotate-half dispatch.
pub struct RotateHalfPlan<'a, T> {
    views: RotateHalfViews<'a, T>,
}

/// Validate and prepare a rotate-half input before allocating its output.
///
/// # Errors
///
/// Returns a typed Leto error when the rank, final extent, layout, or storage
/// cannot represent the two input halves.
pub fn prepare_rotate_half_input<'a, T: leto_ops::RealScalar>(
    input_layout: &CoeusLayout,
    input: &'a [T],
) -> Result<RotateHalfPlan<'a, T>> {
    let rank = input_layout.ndim();
    if rank == 0 || !input_layout.shape()[rank - 1].is_multiple_of(2) {
        return Err(LetoError::StorageError {
            reason: "rotate-half requires nonzero rank and an even final extent".to_owned(),
        });
    }
    if rank > MAX_DISPATCH_RANK {
        return Err(LetoError::StorageError {
            reason: format!("rotate-half supports rank 1..={MAX_DISPATCH_RANK}, got {rank}"),
        });
    }

    let axis = rank - 1;
    let half = input_layout.shape()[axis] / 2;
    let (first, second) =
        input_layout
            .split_axis(axis, half)
            .ok_or_else(|| LetoError::StorageError {
                reason: "rotate-half input layout split overflowed".to_owned(),
            })?;
    let views = match rank {
        1 => RotateHalfViews::Rank1(
            to_leto_view::<T, 1>(&first, input)?,
            to_leto_view::<T, 1>(&second, input)?,
        ),
        2 => RotateHalfViews::Rank2(
            to_leto_view::<T, 2>(&first, input)?,
            to_leto_view::<T, 2>(&second, input)?,
        ),
        3 => RotateHalfViews::Rank3(
            to_leto_view::<T, 3>(&first, input)?,
            to_leto_view::<T, 3>(&second, input)?,
        ),
        4 => RotateHalfViews::Rank4(
            to_leto_view::<T, 4>(&first, input)?,
            to_leto_view::<T, 4>(&second, input)?,
        ),
        5 => RotateHalfViews::Rank5(
            to_leto_view::<T, 5>(&first, input)?,
            to_leto_view::<T, 5>(&second, input)?,
        ),
        6 => RotateHalfViews::Rank6(
            to_leto_view::<T, 6>(&first, input)?,
            to_leto_view::<T, 6>(&second, input)?,
        ),
        _ => unreachable!("invariant: rotate-half rank was validated as 1..=6"),
    };
    Ok(RotateHalfPlan { views })
}

fn rotate_half_n<T: leto_ops::RealScalar, const N: usize>(
    first: ArrayView<'_, T, N>,
    second: ArrayView<'_, T, N>,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    let axis = N - 1;
    let half = first.shape()[axis];
    let (output_first, output_second) =
        output_layout
            .split_axis(axis, half)
            .ok_or_else(|| LetoError::StorageError {
                reason: "rotate-half output layout split overflowed".to_owned(),
            })?;

    {
        let mut destination = to_leto_view_mut::<T, N>(&output_first, output)?;
        leto_ops::unary_map_into(leto_ops::NegOp, &second, &mut destination)?;
    }
    let mut destination = to_leto_view_mut::<T, N>(&output_second, output)?;
    leto_ops::map_into(&first, &mut destination, |value| value)
}

/// Write `[-x₂, x₁]` into caller-owned CPU storage through Leto.
///
/// # Errors
///
/// Returns a typed Leto error for rank, layout, storage, or shape violations.
pub fn rotate_half_into<T: leto_ops::RealScalar>(
    plan: RotateHalfPlan<'_, T>,
    output_layout: &CoeusLayout,
    output: &mut [T],
) -> Result<()> {
    match plan.views {
        RotateHalfViews::Rank1(first, second) => {
            rotate_half_n(first, second, output_layout, output)
        }
        RotateHalfViews::Rank2(first, second) => {
            rotate_half_n(first, second, output_layout, output)
        }
        RotateHalfViews::Rank3(first, second) => {
            rotate_half_n(first, second, output_layout, output)
        }
        RotateHalfViews::Rank4(first, second) => {
            rotate_half_n(first, second, output_layout, output)
        }
        RotateHalfViews::Rank5(first, second) => {
            rotate_half_n(first, second, output_layout, output)
        }
        RotateHalfViews::Rank6(first, second) => {
            rotate_half_n(first, second, output_layout, output)
        }
    }
}
