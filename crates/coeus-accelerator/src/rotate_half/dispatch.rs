use super::RotateHalfProvider;
use crate::{layout::ranked, HephaestusProvider};
use coeus_core::{Layout, Scalar};
use hephaestus_core::{
    ComputeDevice, ElementwiseOps, HephaestusError, IdentityOp, NegOp, StridedView, UnaryExpr,
};

type Buffer<P, T> = <<P as HephaestusProvider>::Device as ComputeDevice>::Buffer<T>;

fn half_layouts(layout: &Layout) -> hephaestus_core::Result<(Layout, Layout)> {
    let rank = layout.ndim();
    if rank == 0 {
        return Err(HephaestusError::InvalidConfiguration {
            message: "rotate-half requires nonzero rank".to_owned(),
        });
    }
    let axis = rank - 1;
    let extent = layout.shape()[axis];
    if !extent.is_multiple_of(2) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!("rotate-half requires an even final extent, got {extent}"),
        });
    }
    layout
        .split_axis(axis, extent / 2)
        .ok_or_else(|| HephaestusError::InvalidConfiguration {
            message: "rotate-half layout split overflowed".to_owned(),
        })
}

fn ranked_layout<const N: usize>(layout: &Layout) -> hephaestus_core::Result<leto::Layout<N>> {
    ranked::<N>("rotate_half", layout).map_err(|source| HephaestusError::InvalidConfiguration {
        message: source.to_string(),
    })
}

fn rotate_half_rank<P, T, const N: usize>(
    input: &Buffer<P, T>,
    layout: &Layout,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: RotateHalfProvider<T>,
    T: Scalar,
    IdentityOp: UnaryExpr<<P::Operations as ElementwiseOps<P::Device, T>>::Dialect>,
    NegOp: UnaryExpr<<P::Operations as ElementwiseOps<P::Device, T>>::Dialect>,
{
    let output_layout = Layout::new(layout.shape_cloned());
    let (input_first, input_second) = half_layouts(layout)?;
    let (output_first, output_second) = half_layouts(&output_layout)?;
    let input_first = ranked_layout::<N>(&input_first)?;
    let input_second = ranked_layout::<N>(&input_second)?;
    let output_first = ranked_layout::<N>(&output_first)?;
    let output_second = ranked_layout::<N>(&output_second)?;
    let device = P::try_device()?;
    let output = device.alloc_uninitialized::<T>(layout.numel())?;
    let operations = P::Operations::default();

    operations.unary_into::<NegOp, N>(
        device,
        StridedView::new(input, &input_second),
        StridedView::new(&output, &output_first),
    )?;
    operations.unary_into::<IdentityOp, N>(
        device,
        StridedView::new(input, &input_first),
        StridedView::new(&output, &output_second),
    )?;
    Ok(output)
}

/// Allocate and initialize provider storage for `[-x₂, x₁]`.
///
/// # Errors
///
/// Returns a typed provider failure for invalid rank, odd final extent,
/// allocation, layout conversion, or elementwise dispatch failure.
pub fn rotate_half<P, T>(
    input: &Buffer<P, T>,
    layout: &Layout,
) -> hephaestus_core::Result<Buffer<P, T>>
where
    P: RotateHalfProvider<T>,
    T: Scalar,
    IdentityOp: UnaryExpr<<P::Operations as ElementwiseOps<P::Device, T>>::Dialect>,
    NegOp: UnaryExpr<<P::Operations as ElementwiseOps<P::Device, T>>::Dialect>,
{
    match layout.ndim() {
        1 => rotate_half_rank::<P, T, 1>(input, layout),
        2 => rotate_half_rank::<P, T, 2>(input, layout),
        3 => rotate_half_rank::<P, T, 3>(input, layout),
        4 => rotate_half_rank::<P, T, 4>(input, layout),
        rank => Err(HephaestusError::InvalidConfiguration {
            message: format!("accelerator rotate-half supports rank 1..=4, got {rank}"),
        }),
    }
}
