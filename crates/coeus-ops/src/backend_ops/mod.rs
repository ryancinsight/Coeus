// ── Backend-parameterized execution operations ──
// Unifies CPU and GPU dispatch via monomorphized associated traits.
#![allow(clippy::too_many_arguments)]

mod cpu_impl;
pub(crate) mod defaults;
/// Operation enum types (BinaryOp, ReductionOp, UnaryOp) re-exported from coeus_core.
pub mod ops;
/// `BackendOps` super-trait definition and blanket impl.
pub mod trait_def;
/// Interface-segregated sub-traits and the `BackendOps` super-trait.
pub mod traits;

pub use cpu_impl::CpuBackend;
pub use ops::{BinaryOp, ReductionOp, UnaryOp};
pub use trait_def::BackendOps;
pub use traits::{
    AttentionOps, AttentionScalar, ConvOps, ConvolutionBackward, ConvolutionForward,
    ElementwiseOps, MatmulOps, OptimizerOps, OptimizerStateRef, OptimizerStepRule,
    OptimizerStepValidation, PoolOps, ReductionOps, UnfoldFoldOps,
};
