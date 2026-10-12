//! Provider-owned fused-expression operation markers.

use coeus_core::Scalar;
use hephaestus_core::{FusedElementwiseOps, FusedReductionOps, KernelDialect};

use crate::HephaestusProvider;

/// Provider-owned fused-expression operation marker.
///
/// The [`Dialect`](Self::Dialect) pins the single kernel dialect both fused
/// operation families must author; the equality bindings on
/// [`Operations`](Self::Operations) verify at each vendor impl that its
/// elementwise and reduction bundles agree, so one expression adapter serves
/// both dispatch paths.
pub trait FusionProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Kernel dialect authored by both fused operation families.
    type Dialect: KernelDialect;
    /// Monomorphized fused operation bundle selected by this provider.
    type Operations: FusedElementwiseOps<Self::Device, T, Dialect = Self::Dialect>
        + FusedReductionOps<Self::Device, T, Dialect = Self::Dialect>
        + Default;
}
