//! Coeus expression and layout adaptation for provider-owned WGPU fusion.
//!
//! Bundle and entry wiring over the shared generic dispatch
//! ([`coeus_accelerator::fusion`], ADR 0082); all dispatch logic lives there
//! and monomorphizes once for [`WgpuBackend`].

use coeus_accelerator::FusionBackend;
use coeus_core::{BackendError, Layout};
use coeus_ops::fuse::ExprNode;
use hephaestus_core::{FusedReduction, Wgsl};

use crate::backend::{WgpuBackend, WgpuScalar};
use crate::WgpuBackendError;

// SAFETY: the seam contract is upheld by the existing
// `HephaestusSeamBackend` implementation, and `WgpuFusionOps` initializes
// every logical output element on success.
unsafe impl<T: WgpuScalar> FusionBackend<T> for WgpuBackend {
    type Dialect = Wgsl;
    type FusionOps = hephaestus_wgpu::WgpuFusionOps;

    fn validation_error(error: BackendError) -> Self::Error {
        WgpuBackendError::Validation(error)
    }
}

/// Dispatch a fused elementwise expression through Hephaestus.
pub(crate) fn dispatch_fused<T, E>(
    expression: &E,
    output: &mut coeus_accelerator::HephaestusStorage<crate::WgpuBackend, T>,
    output_layout: &Layout,
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar,
    E: ExprNode<T, WgpuBackend>,
{
    coeus_accelerator::dispatch_fused::<T, WgpuBackend, E>(expression, output, output_layout)
}

/// Dispatch a fused expression reduction through Hephaestus.
pub(crate) fn dispatch_fused_reduce<T, E>(
    expression: &E,
    reduction: FusedReduction,
    axis: usize,
    output: &mut coeus_accelerator::HephaestusStorage<crate::WgpuBackend, T>,
    output_layout: &Layout,
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar,
    E: ExprNode<T, WgpuBackend>,
{
    coeus_accelerator::dispatch_fused_reduce::<T, WgpuBackend, E>(
        expression,
        reduction,
        axis,
        output,
        output_layout,
    )
}
