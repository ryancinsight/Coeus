// ── Backend-parameterized execution operations ──
// Unifies CPU and GPU dispatch via monomorphized associated traits.
//
// The kernel trait methods (unfold/fold, pooling, optimizer, attention,
// finite-difference) take the full geometric argument list — kernel, stride,
// padding, dilation per spatial axis — and are implemented by every backend.
// Grouping them into parameter structs is a cross-crate API change; until then
// the whole subtree shares one suppression, replacing the four redundant
// per-module copies that previously restated it.
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
/// Host-memory cross-product fold shared by every backend without an
/// on-device seam (`CrossOps` implementors outside this crate use it
/// directly; see ADR 0077).
pub use defaults::cross::cross_fold;
pub use ops::{BinaryOp, ReductionOp, UnaryOp};
pub use trait_def::BackendOps;
pub use traits::Axis;
pub use traits::{
    AttentionOps, AttentionScalar, ConvOps, ConvolutionBackward, ConvolutionForward,
    CrossEntropyOps, CrossOps, CtcBatch, CtcOps, ElementwiseOps, FiniteDifference3DOps,
    FiniteDifference3DScheme, MatmulOps, OptimizerOps, OptimizerStateRef, OptimizerStepRule,
    OptimizerStepValidation, PoolOps, RandomInitOps, ReductionOps, RotateHalfOps, ScalarPowerOps,
    StaggeredPairOps, UnfoldFoldOps,
};
