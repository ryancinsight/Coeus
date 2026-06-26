// ── Backend-parameterized execution operations ──
// Unifies CPU and GPU dispatch via monomorphized associated traits.
#![allow(clippy::too_many_arguments)]

mod cpu_impl;
pub(crate) mod defaults;
/// Operation enum types (BinaryOp, ReductionOp, UnaryOp) re-exported from coeus_core.
pub mod ops;
/// BackendOps trait definition.
pub mod trait_def;

pub use cpu_impl::CpuBackend;
pub use ops::{BinaryOp, ReductionOp, UnaryOp};
pub use trait_def::BackendOps;
