//! CPU finite-difference kernels, delegating to the Leto provider.
//!
//! Nothing is reimplemented here. Leto owns the stencils — the fixed central
//! and Yee schemes and the arbitrary-even-order staggered pair — and this
//! module is the adaptation between a Coeus backend's buffers and layouts and
//! the provider's array views.
//!
//! The adaptation is a borrow, not a copy: Leto's destination is a mutable
//! view, so a device buffer's host-addressable slice is written in place. An
//! owned-array parameter would have meant one allocation and one copy per
//! sweep, which inside an FDTD timestep is the cost the whole seam exists to
//! avoid.

use coeus_core::{BackendError, Layout};
use leto::{ArrayView3, ArrayViewMut3, Layout as LetoLayout};
use leto_ops::{Axis, FiniteDifference3D, FiniteDifference3DScheme, StaggeredLeapfrog3D};

use super::error::map_leto_error;

/// Scalars a CPU finite-difference kernel accepts: Coeus's element vocabulary
/// intersected with the real-field arithmetic the stencils execute in.
pub trait FdScalar: coeus_core::Scalar + eunomia::RealField + eunomia::FloatElement + Copy {}

impl<T> FdScalar for T where
    T: coeus_core::Scalar + eunomia::RealField + eunomia::FloatElement + Copy
{
}

/// A rank-3 contiguous row-major layout, or the error saying why it is not one.
///
/// The stencils sweep by index arithmetic over a dense field, so a strided or
/// broadcast view is not a shape they can serve. Rejecting it here keeps the
/// failure at the boundary with a name, rather than as a wrong answer inside a
/// kernel.
fn dense_layout(
    operation: &'static str,
    layout: &Layout,
) -> Result<(LetoLayout<3>, [usize; 3]), BackendError> {
    let shape = layout.shape();
    let [nx, ny, nz] = match *shape {
        [nx, ny, nz] => [nx, ny, nz],
        _ => {
            return Err(BackendError::Storage {
                operation,
                reason: format!("finite differences need a rank-3 field, got shape {shape:?}"),
            })
        }
    };
    if !layout.is_contiguous() || layout.offset() != 0 {
        return Err(BackendError::Storage {
            operation,
            reason: format!(
                "finite differences need a contiguous field at offset zero, got strides {:?} \
                 at offset {}",
                layout.strides(),
                layout.offset()
            ),
        });
    }
    let strides = [
        isize::try_from(ny * nz).map_err(|_| BackendError::Overflow {
            operation,
            reason: "row-major stride exceeds isize",
        })?,
        isize::try_from(nz).map_err(|_| BackendError::Overflow {
            operation,
            reason: "row-major stride exceeds isize",
        })?,
        1,
    ];
    let leto_layout = LetoLayout::<3>::try_new([nx, ny, nz], strides, 0)
        .map_err(|error| map_leto_error(operation, error))?;
    Ok((leto_layout, [nx, ny, nz]))
}

/// Borrow both sides as Leto views over the caller's own storage.
fn views<'a, T: FdScalar>(
    operation: &'static str,
    input: &'a [T],
    input_layout: &Layout,
    output: &'a mut [T],
    output_layout: &Layout,
) -> Result<(ArrayView3<'a, T>, ArrayViewMut3<'a, T>), BackendError> {
    let (in_layout, _) = dense_layout(operation, input_layout)?;
    let (out_layout, _) = dense_layout(operation, output_layout)?;
    let field =
        ArrayView3::try_new(in_layout, input).map_err(|error| map_leto_error(operation, error))?;
    let dst = ArrayViewMut3::try_new(out_layout, output)
        .map_err(|error| map_leto_error(operation, error))?;
    Ok((field, dst))
}

/// Prepare the staggered gradient/divergence pair once, so no sweep pays for
/// the coefficient derivation.
pub(super) fn prepare_staggered_pair<T: FdScalar>(
    order: usize,
    spacing: [T; 3],
) -> Result<StaggeredLeapfrog3D<T>, BackendError> {
    StaggeredLeapfrog3D::new(order, spacing[0], spacing[1], spacing[2])
        .map_err(|error| map_leto_error("prepare_staggered_pair", error))
}

pub(super) fn staggered_gradient<T: FdScalar>(
    pair: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    input: &[T],
    input_layout: &Layout,
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    const OP: &str = "staggered_gradient";
    let (field, mut dst) = views(OP, input, input_layout, output, output_layout)?;
    pair.gradient_into(axis, field, &mut dst)
        .map_err(|error| map_leto_error(OP, error))
}

pub(super) fn staggered_divergence<T: FdScalar>(
    pair: &StaggeredLeapfrog3D<T>,
    axis: Axis,
    input: &[T],
    input_layout: &Layout,
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    const OP: &str = "staggered_divergence";
    let (field, mut dst) = views(OP, input, input_layout, output, output_layout)?;
    pair.divergence_into(axis, field, &mut dst)
        .map_err(|error| map_leto_error(OP, error))
}

pub(super) fn finite_difference<T: FdScalar>(
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    spacing: [T; 3],
    input: &[T],
    input_layout: &Layout,
    output: &mut [T],
    output_layout: &Layout,
) -> Result<(), BackendError> {
    const OP: &str = "finite_difference";
    let operator = FiniteDifference3D::new(scheme, spacing[0], spacing[1], spacing[2])
        .map_err(|error| map_leto_error(OP, error))?;
    let (field, mut dst) = views(OP, input, input_layout, output, output_layout)?;
    match axis {
        Axis::X => operator.apply_x_into(field, &mut dst),
        Axis::Y => operator.apply_y_into(field, &mut dst),
        Axis::Z => operator.apply_z_into(field, &mut dst),
    }
    .map_err(|error| map_leto_error(OP, error))
}

#[cfg(test)]
mod tests;
