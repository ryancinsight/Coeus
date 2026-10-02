// ── Backend-parameterized execution operations ──
// Unifies CPU and GPU dispatch via monomorphized associated traits.
//
// The `too_many_arguments` expectation lives on the two subtrees that trip it
// — `traits` (the kernel contracts) and `cpu_impl` (their CPU realizations) —
// rather than on this shared parent.

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
