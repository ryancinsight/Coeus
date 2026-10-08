use super::provider::FixedFdBackend;
use crate::layout::check_contiguous_exact;
use coeus_core::{Layout, StorageMut};
use coeus_ops::{Axis, FiniteDifference3DScheme};
use core::borrow::Borrow;
use hephaestus_core::{
    FixedFd3DOps, FixedFd3DParams, FixedFd3DScheme as ProviderScheme, StaggeredAxis,
};

fn provider_axis(axis: Axis) -> StaggeredAxis {
    match axis {
        Axis::X => StaggeredAxis::X,
        Axis::Y => StaggeredAxis::Y,
        Axis::Z => StaggeredAxis::Z,
    }
}

fn provider_scheme(scheme: FiniteDifference3DScheme) -> ProviderScheme {
    match scheme {
        FiniteDifference3DScheme::CentralSecondOrder => ProviderScheme::CentralSecondOrder,
        FiniteDifference3DScheme::CentralFourthOrder => ProviderScheme::CentralFourthOrder,
        FiniteDifference3DScheme::CentralSixthOrder => ProviderScheme::CentralSixthOrder,
        FiniteDifference3DScheme::StaggeredForward => ProviderScheme::StaggeredForward,
        FiniteDifference3DScheme::StaggeredBackward => ProviderScheme::StaggeredBackward,
    }
}

/// Reject a layout the stencils cannot serve by name at the boundary — the
/// alternative is a kernel sweeping a shape it was not given.
fn check_layout<B>(
    operation: &'static str,
    operand: &'static str,
    layout: &Layout,
) -> Result<(), B::Error>
where
    B: FixedFdBackend,
{
    check_contiguous_exact::<3, _>(operation, "fixed-fd", operand, layout, |reason| {
        B::fixed_fd_configuration_error(operation, reason)
    })
}

fn axis_lane(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

fn block<B>(
    operation: &'static str,
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    spacing: [f32; 3],
    grid: &[usize],
) -> Result<FixedFd3DParams, B::Error>
where
    B: FixedFdBackend,
{
    let mut dims = [0_u32; 3];
    for (slot, &extent) in dims.iter_mut().zip(grid) {
        *slot = u32::try_from(extent).map_err(|error| {
            B::fixed_fd_configuration_error(
                operation,
                format!("fixed-fd grid extent {extent} does not fit u32: {error}"),
            )
        })?;
    }
    FixedFd3DParams::new(
        dims[0],
        dims[1],
        dims[2],
        provider_axis(axis),
        provider_scheme(scheme),
        spacing,
    )
    .map_err(|source| B::fixed_fd_dispatch_error(operation, source))
}

/// Build the provider parameter block for one sweep.
///
/// The output shape must equal the input shape, except a forward sweep drops
/// one plane on the differentiated axis.
fn parameters<B>(
    operation: &'static str,
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    spacing: [f32; 3],
    layouts: (&Layout, &Layout),
) -> Result<FixedFd3DParams, B::Error>
where
    B: FixedFdBackend,
{
    check_layout::<B>(operation, "input", layouts.0)?;
    check_layout::<B>(operation, "output", layouts.1)?;
    let mut expected = layouts.0.shape().to_vec();
    if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
        let lane = axis_lane(axis);
        expected[lane] = expected[lane].saturating_sub(1);
    }
    if layouts.1.shape() != expected.as_slice() {
        return Err(B::fixed_fd_configuration_error(
            operation,
            format!(
                "fixed-fd {scheme:?} output shape {:?} must be {expected:?} for input shape {:?}",
                layouts.1.shape(),
                layouts.0.shape(),
            ),
        ));
    }
    block::<B>(operation, scheme, axis, spacing, layouts.0.shape())
}

/// Build the provider parameter block for one transpose sweep.
///
/// The gradient always has the full input grid; the upstream has the forward
/// sweep's output shape, shrunk on the axis for a forward sweep.
fn adjoint_parameters<B>(
    operation: &'static str,
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    spacing: [f32; 3],
    upstream_layout: &Layout,
    grad_layout: &Layout,
) -> Result<FixedFd3DParams, B::Error>
where
    B: FixedFdBackend,
{
    check_layout::<B>(operation, "upstream", upstream_layout)?;
    check_layout::<B>(operation, "grad", grad_layout)?;
    let mut expected = grad_layout.shape().to_vec();
    if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
        let lane = axis_lane(axis);
        expected[lane] = expected[lane].saturating_sub(1);
    }
    if upstream_layout.shape() != expected.as_slice() {
        return Err(B::fixed_fd_configuration_error(
            operation,
            format!(
                "fixed-fd {scheme:?} upstream shape {:?} must be {expected:?} for grad shape {:?}",
                upstream_layout.shape(),
                grad_layout.shape(),
            ),
        ));
    }
    block::<B>(operation, scheme, axis, spacing, grad_layout.shape())
}

/// Sweep one fixed-scheme derivative while preserving destination clones.
///
/// # Errors
///
/// Returns the backend's typed error unless both layouts are contiguous,
/// zero-offset rank-three fields with the scheme's output shape, or when the
/// provider rejects the stencil parameters or dispatch.
pub fn sweep<B>(
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    spacing: [f32; 3],
    input: (&B::DeviceBuffer<f32>, &Layout),
    output: (&mut B::DeviceBuffer<f32>, &Layout),
) -> Result<(), B::Error>
where
    B: FixedFdBackend,
    B::Kernel: Borrow<<B::Operations as FixedFd3DOps<B::Device>>::FixedFd3D>,
{
    const OPERATION: &str = "finite_difference";
    let params = parameters::<B>(OPERATION, scheme, axis, spacing, (input.1, output.1))?;
    output.0.make_unique();
    let kernel = B::fixed_fd_kernel()?;
    B::Operations::default()
        .fixed_fd_into(
            B::fixed_fd_device(),
            kernel.borrow(),
            B::fixed_fd_buffer(input.0),
            B::fixed_fd_buffer(output.0),
            &params,
        )
        .map_err(|source| B::fixed_fd_dispatch_error(OPERATION, source))
}

/// Sweep one fixed-scheme transpose while preserving destination clones.
///
/// The upstream has the forward sweep's output shape; the gradient the full
/// input grid.
///
/// # Errors
///
/// Returns the backend's typed error unless both layouts are contiguous,
/// zero-offset rank-three fields with the transpose's shapes, or when the
/// provider rejects the stencil parameters or dispatch.
pub fn adjoint<B>(
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    spacing: [f32; 3],
    upstream: (&B::DeviceBuffer<f32>, &Layout),
    grad: (&mut B::DeviceBuffer<f32>, &Layout),
) -> Result<(), B::Error>
where
    B: FixedFdBackend,
    B::Kernel: Borrow<<B::Operations as FixedFd3DOps<B::Device>>::FixedFd3D>,
{
    const OPERATION: &str = "finite_difference_adjoint";
    let params = adjoint_parameters::<B>(OPERATION, scheme, axis, spacing, upstream.1, grad.1)?;
    grad.0.make_unique();
    let kernel = B::fixed_fd_kernel()?;
    B::Operations::default()
        .fixed_fd_adjoint_into(
            B::fixed_fd_device(),
            kernel.borrow(),
            B::fixed_fd_buffer(upstream.0),
            B::fixed_fd_buffer(grad.0),
            &params,
        )
        .map_err(|source| B::fixed_fd_dispatch_error(OPERATION, source))
}
