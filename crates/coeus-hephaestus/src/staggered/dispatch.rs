use super::provider::StaggeredBackend;
use crate::layout::ranked_exact;
use coeus_core::Layout;
use coeus_ops::FiniteDifferenceAxis as Axis;
use hephaestus_core::{Staggered3DOps, Staggered3DParams, StaggeredAxis};
use leto_ops::{staggered_first_derivative_coefficients, TapCoefficients};

/// A prepared staggered pair: the compiled device kernels plus the derived taps
/// and spacings every dispatch needs.
///
/// The taps are derived once here rather than per sweep, because an order-`2N`
/// staggered stencil's coefficients come from solving a Taylor system and a
/// linear solve inside an FDTD timestep is exactly what preparation exists to
/// remove. The grid is *not* part of this: the provider's parameter block binds
/// dimensions and axis together with the taps, and both arrive only at dispatch
/// — from the operand layout and the caller's axis.
pub struct PreparedStaggeredPair<B>
where
    B: StaggeredBackend,
{
    kernel: <B::Operations as Staggered3DOps<B::Device>>::Staggered3D,
    taps: TapCoefficients<f32>,
    spacing: [f32; 3],
}

impl<B> PreparedStaggeredPair<B>
where
    B: StaggeredBackend,
{
    /// Derive the taps and compile the device kernels once.
    ///
    /// # Errors
    ///
    /// Returns the backend's typed error for an order the provider derivation
    /// does not cover, and the provider's kernel-compilation failure.
    pub fn new(order: usize, spacing: [f32; 3]) -> Result<Self, B::Error> {
        const OPERATION: &str = "prepare_staggered_pair";
        if order == 0 || !order.is_multiple_of(2) {
            return Err(B::staggered_configuration_error(
                OPERATION,
                format!("staggered order must be even and non-zero, got {order}"),
            ));
        }
        let taps = staggered_first_derivative_coefficients::<f32>(order / 2)
            .map_err(|error| B::staggered_configuration_error(OPERATION, error.to_string()))?;
        let kernel = B::Operations::default()
            .prepare_staggered_3d(B::staggered_device())
            .map_err(|source| B::staggered_dispatch_error(OPERATION, source))?;
        Ok(Self {
            kernel,
            taps,
            spacing,
        })
    }
}

fn provider_axis(axis: Axis) -> StaggeredAxis {
    match axis {
        Axis::X => StaggeredAxis::X,
        Axis::Y => StaggeredAxis::Y,
        Axis::Z => StaggeredAxis::Z,
    }
}

/// Build the provider parameter block for one dispatch.
///
/// Rejects a layout the stencils cannot serve by name at the boundary — the
/// alternative is a kernel sweeping a shape it was not given.
fn parameters<B>(
    operation: &'static str,
    pair: &PreparedStaggeredPair<B>,
    axis: Axis,
    layout: &Layout,
) -> Result<Staggered3DParams, B::Error>
where
    B: StaggeredBackend,
{
    let leto_layout = ranked_exact::<3>(operation, layout)
        .map_err(|error| B::staggered_configuration_error(operation, error.to_string()))?;
    let shape = leto_layout.shape();
    let mut dims = [0_u32; 3];
    for (slot, extent) in dims.iter_mut().zip(shape) {
        *slot = u32::try_from(extent).map_err(|error| {
            B::staggered_configuration_error(
                operation,
                format!("staggered grid extent {extent} does not fit u32: {error}"),
            )
        })?;
    }
    Staggered3DParams::new(
        dims[0],
        dims[1],
        dims[2],
        provider_axis(axis),
        pair.taps.taps(),
        pair.spacing,
    )
    .map_err(|source| B::staggered_dispatch_error(operation, source))
}

/// Dispatch the staggered gradient through the selected provider.
///
/// # Errors
///
/// Returns the backend's typed error for a layout the stencils cannot serve,
/// and the provider's dispatch failure.
pub fn gradient<B>(
    pair: &PreparedStaggeredPair<B>,
    axis: Axis,
    input: (&B::DeviceBuffer<f32>, &Layout),
    output: (&B::DeviceBuffer<f32>, &Layout),
) -> Result<(), B::Error>
where
    B: StaggeredBackend,
{
    const OPERATION: &str = "staggered_gradient";
    let params = parameters::<B>(OPERATION, pair, axis, output.1)?;
    parameters::<B>(OPERATION, pair, axis, input.1)?;
    B::Operations::default()
        .staggered_gradient_into(
            B::staggered_device(),
            &pair.kernel,
            B::staggered_buffer(input.0),
            B::staggered_buffer(output.0),
            &params,
        )
        .map_err(|source| B::staggered_dispatch_error(OPERATION, source))
}

/// Dispatch the staggered divergence through the selected provider.
///
/// # Errors
///
/// See [`gradient`].
pub fn divergence<B>(
    pair: &PreparedStaggeredPair<B>,
    axis: Axis,
    input: (&B::DeviceBuffer<f32>, &Layout),
    output: (&B::DeviceBuffer<f32>, &Layout),
) -> Result<(), B::Error>
where
    B: StaggeredBackend,
{
    const OPERATION: &str = "staggered_divergence";
    let params = parameters::<B>(OPERATION, pair, axis, output.1)?;
    parameters::<B>(OPERATION, pair, axis, input.1)?;
    B::Operations::default()
        .staggered_divergence_into(
            B::staggered_device(),
            &pair.kernel,
            B::staggered_buffer(input.0),
            B::staggered_buffer(output.0),
            &params,
        )
        .map_err(|source| B::staggered_dispatch_error(OPERATION, source))
}
