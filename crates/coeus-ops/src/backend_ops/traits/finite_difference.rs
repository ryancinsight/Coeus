//! Finite-difference sub-trait.
//!
//! [`FiniteDifference3DOps`] provides fixed central and Yee first-derivative
//! stencils. [`StaggeredPairOps`] provides the arbitrary-even-order staggered
//! gradient/divergence pair used by a conservative leapfrog.
//!
//! # Why this is not part of [`BackendOps`]
//!
//! [`BackendOps`] is the set every backend must satisfy. A backend that has no
//! stencil kernels yet would have to supply stubs to join it, and a stub that
//! returns zeros or an error is a mock wearing a trait impl. Consumers bind
//! `B: FiniteDifference3DOps<T>` or `B: StaggeredPairOps<T>` directly, so a
//! backend advertises each capability when it implements that family.
//!
//! # Preparation
//!
//! The coefficients of an order-`2N` staggered stencil come from solving a
//! Taylor system; deriving them per sweep would put a linear solve inside an
//! FDTD timestep. [`StaggeredPairOps::prepare_staggered_pair`] does that
//! once and returns whatever the backend needs to cache — taps on the CPU, a
//! compiled kernel and an uploaded coefficient buffer on a device.
//!
//! [`BackendOps`]: super::super::BackendOps

use coeus_core::{ComputeBackend, Layout, Scalar};

pub use leto_ops::{Axis, FiniteDifference3DScheme};

/// The staggered gradient/divergence pair at arbitrary even order.
///
/// Split from [`FiniteDifference3DOps`] because the two capabilities do not
/// arrive together: an accelerator can serve this pair — the operator an FDTD
/// leapfrog actually runs — long before it has the fixed central and Yee
/// schemes. Bundling them would oblige such a backend to supply a body for
/// what it cannot do, and a body that returns an error is a mock wearing a
/// trait impl.
///
/// Dispatch operands have matching row-major `[nx, ny, nz]` layouts with zero
/// offsets. Gradient and divergence write into caller-supplied destinations;
/// preparation derives coefficients and acquires provider resources.
pub trait StaggeredPairOps<T: Scalar>: ComputeBackend {
    /// Backend-side form of a prepared staggered gradient/divergence pair.
    type StaggeredPair;

    /// Derive an order-`order` staggered pair for a grid of the given
    /// per-axis spacings.
    ///
    /// The pair's gradient and divergence are negative adjoints, `D = -Gᵀ`,
    /// which is the condition for a conserved discrete energy in a leapfrog.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error for an odd or zero order, an order
    /// beyond the coefficient derivation's verified range, or a non-positive
    /// spacing.
    fn prepare_staggered_pair(
        &self,
        order: usize,
        spacing: [T; 3],
    ) -> Result<Self::StaggeredPair, Self::Error>;

    /// Gradient along `axis`: cell-centred `input` to face-centred `output`,
    /// with face `i+½` stored at index `i`.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when a layout is not a contiguous
    /// rank-3 field, or when the two layouts disagree.
    fn staggered_gradient(
        &self,
        pair: &Self::StaggeredPair,
        axis: Axis,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Divergence along `axis`: face-centred `input` back to cell-centred
    /// `output`. This is `-Gᵀ` of [`Self::staggered_gradient`].
    ///
    /// # Errors
    ///
    /// See [`Self::staggered_gradient`].
    fn staggered_divergence(
        &self,
        pair: &Self::StaggeredPair,
        axis: Axis,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;
}

/// Fixed-scheme three-dimensional first derivatives.
///
/// The central and Yee families whose coefficients are fixed by the scheme
/// rather than derived per order. A backend implements this when it has those
/// kernels; the staggered pair is [`StaggeredPairOps`], which arrives first on
/// an accelerator.
pub trait FiniteDifference3DOps<T: Scalar>: ComputeBackend {
    /// Fixed-scheme first derivative along `axis`.
    ///
    /// The central schemes keep the field's shape; the staggered schemes
    /// follow the shape contract documented on [`FiniteDifference3DScheme`].
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error for a non-positive spacing, a
    /// non-contiguous or non-rank-3 layout, an axis too short for the scheme,
    /// or an output shape the scheme does not produce.
    fn finite_difference(
        &self,
        scheme: FiniteDifference3DScheme,
        axis: Axis,
        spacing: [T; 3],
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
