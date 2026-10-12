// ── Dtype traits ──
// Sealed trait hierarchy for numeric scalar types used in tensors.
//
// Design notes:
// - `Scalar` is the base: Copy + eunomia::Pod + Send + Sync + 'static
// - `Float` extends Scalar with transcendental and rounding ops
// - `Int` extends Scalar with bitwise and modular ops
// - All traits are sealed (private Sealed supertrait) for monomorphization
// - eunomia::Pod guarantees safe host/device byte layouts

mod cpu_dispatch;
mod float;
mod float_ops;
mod int;
mod op_tags;
mod scalar;

// ── Sealed pattern ──
pub(crate) mod private {
    pub trait Sealed {}
}

pub use cpu_dispatch::CpuUnaryDispatch;
pub use float::Float;
pub use float_ops::FloatOps;
pub use int::Int;
pub use op_tags::{BinaryOp, CpuUnaryOp, ReductionOp};
pub use scalar::Scalar;
