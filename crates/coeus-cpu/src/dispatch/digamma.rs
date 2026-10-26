//! Layout adapter for Leto's real digamma operation.
use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout;
use leto::{LetoError, Result};
use leto_ops::{DigammaOp, RealScalar};

#[inline]
fn map<T: RealScalar, const N: usize>(
    input_layout: &Layout,
    input: &[T],
    output_layout: &Layout,
    output: &mut [T],
) -> Result<()> {
    let input = to_leto_view::<T, N>(input_layout, input)?;
    let mut output = to_leto_view_mut::<T, N>(output_layout, output)?;
    leto_ops::unary_map_into(DigammaOp, &input, &mut output)
}

/// Apply Leto's digamma operation without copying or changing the input layout.
///
/// # Errors
/// Returns provider layout/storage errors, or rejects ranks above the adapter's
/// supported bound. Scalar (rank-zero) and empty tensors are supported.
#[inline]
pub fn elementwise_digamma_into<T: RealScalar>(
    input_layout: &Layout,
    input: &[T],
    output_layout: &Layout,
    output: &mut [T],
) -> Result<()> {
    match output_layout.ndim() {
        0 => map::<T, 0>(input_layout, input, output_layout, output),
        1 => map::<T, 1>(input_layout, input, output_layout, output),
        2 => map::<T, 2>(input_layout, input, output_layout, output),
        3 => map::<T, 3>(input_layout, input, output_layout, output),
        4 => map::<T, 4>(input_layout, input, output_layout, output),
        5 => map::<T, 5>(input_layout, input, output_layout, output),
        6 => map::<T, 6>(input_layout, input, output_layout, output),
        rank => Err(LetoError::StorageError {
            reason: format!("coeus-cpu digamma supports rank 0..=6, got {rank}"),
        }),
    }
}
