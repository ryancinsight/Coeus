// ── Backend-parameterized execution operations ──
// Unifies CPU and GPU dispatch via monomorphized associated traits.
#![allow(clippy::too_many_arguments)]

mod cpu_impl;
pub mod layout_helpers;
pub mod ops;
pub mod trait_def;

pub use cpu_impl::CpuBackend;
pub(crate) use layout_helpers::{
    compute_broadcast_offsets, compute_reduction_base_offset, compute_unary_offset,
};
pub use ops::{BinaryOp, ReductionOp, UnaryOp};
pub use trait_def::BackendOps;
