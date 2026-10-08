use super::provider::FixedFdBackend;
use crate::layout::ranked_exact;
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

/// Build the provider parameter block for one sweep.
///
/// Rejects a layout the stencils cannot serve by name at the boundary — the
/// alternative is a kernel sweeping a shape it was not given. The output
/// shape must equal the input shape, except a forward sweep drops one plane
/// on the differentiated axis.
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
    for (operand, layout) in [("input", layouts.0), ("output", layouts.1)] {
        ranked_exact::<3>(operation, layout)
            .map_err(|error| B::fixed_fd_configuration_error(operation, error.to_string()))?;
        // The provider parameter block carries dimensions only, so it cannot
        // represent an operand's strides or base offset.
        if !layout.is_contiguous() || layout.offset() != 0 {
            return Err(B::fixed_fd_configuration_error(
                operation,
                format!(
                    "fixed-fd {operand} layout must be contiguous with zero offset, got strides {:?} and offset {}",
                    layout.strides(),
                    layout.offset(),
                ),
            ));
        }
    }
    let mut expected = layouts.0.shape().to_vec();
    if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
        let lane = match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        };
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
    let shape = layouts.0.shape();
    let mut dims = [0_u32; 3];
    for (slot, &extent) in dims.iter_mut().zip(shape) {
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
