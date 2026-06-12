//! Fundamental dtype, layout, storage, and backend abstractions for the Coeus tensor stack.
//!
//! # Trait hierarchy
//! - [`Scalar`] — sealed base for all numeric element types (`f32`, `f64`, `i32`, …).
//! - [`Float`] / [`FloatOps`] — floating-point refinement with transcendental ops.
//! - [`CpuUnaryDispatch`] — CPU-side dispatch table for scalar unary kernels.
//! - [`Backend`] / [`ComputeBackend`] — device abstraction with associated `DeviceBuffer<T>`.
//! - [`Layout`] / [`ConstLayout`] — shape, stride, and contiguity descriptors.
//! - [`Storage`] / [`CpuStorage`] — raw buffer ownership and CPU addressability contracts.

// ── Coeus Core: Fundamental abstractions ──
// Provides dtype system, layout descriptors, storage primitives,
// and backend execution abstractions.

pub mod backend;
pub mod dtype;
pub mod layout;
pub mod ptr;
pub mod storage;

// Re-export the most commonly used items
pub use backend::{Backend, ComputeBackend, MoiraiBackend, SequentialBackend};
pub use dtype::{
    BinaryOp, Complex, CpuUnaryDispatch, CpuUnaryOp, Float, FloatOps, Int, ReductionOp, Scalar,
};
pub use layout::{ConstLayout, ConstShape, Layout, Shape, Strides};
pub use ptr::{SendPtr, SendPtrMut};
pub use storage::{
    CowStorage, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage, Storage, StorageMut,
};
