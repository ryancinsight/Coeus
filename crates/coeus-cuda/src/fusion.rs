//! Coeus expression and layout adaptation for provider-owned CUDA fusion.
//!
//! Bundle and entry wiring over the shared generic dispatch
//! ([`coeus_hephaestus::fusion`], ADR 0082); all dispatch logic lives there
//! and monomorphizes once for [`CudaBackend`].

use coeus_core::{BackendError, Layout};
use coeus_hephaestus::FusionBackend;
use coeus_ops::fuse::ExprNode;
use hephaestus_core::{CudaC, FusedReduction};

use crate::backend::{CudaBackend, CudaScalar};
use crate::CudaBackendError;

// SAFETY: the seam contract is upheld by the existing
// `HephaestusSeamBackend` implementation, and `CudaFusionOps` initializes
// every logical output element on success.
unsafe impl<T: CudaScalar> FusionBackend<T> for CudaBackend {
    type Dialect = CudaC;
    type FusionOps = hephaestus_cuda::CudaFusionOps;

    fn validation_error(error: BackendError) -> Self::Error {
        CudaBackendError::validation(error)
    }
}

/// Dispatch a fused elementwise expression through Hephaestus CUDA.
pub(crate) fn dispatch_fused<T, E>(
    expression: &E,
    output: &mut coeus_hephaestus::HephaestusStorage<crate::CudaBackend, T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: CudaScalar,
    E: ExprNode<T, CudaBackend>,
{
    coeus_hephaestus::dispatch_fused::<T, CudaBackend, E>(expression, output, output_layout)
}

/// Dispatch a fused expression reduction through Hephaestus CUDA.
pub(crate) fn dispatch_fused_reduce<T, E>(
    expression: &E,
    reduction: FusedReduction,
    axis: usize,
    output: &mut coeus_hephaestus::HephaestusStorage<crate::CudaBackend, T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: CudaScalar,
    E: ExprNode<T, CudaBackend>,
{
    coeus_hephaestus::dispatch_fused_reduce::<T, CudaBackend, E>(
        expression,
        reduction,
        axis,
        output,
        output_layout,
    )
}
