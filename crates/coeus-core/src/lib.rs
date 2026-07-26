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
#![deny(missing_docs)]

/// Backend execution abstractions (`Backend`, `ComputeBackend`).
pub mod backend;
/// Dtype system (`Scalar`, `Float`, `Complex`, operation enums).
pub mod dtype;
/// Layout descriptors (`Shape`, `Layout`, `ConstShape`, `ConstLayout`, strides).
pub mod layout;
/// Raw pointer utilities for thread-safe shared storage.
pub mod ptr;
/// Storage abstractions (`Storage`, `CpuStorage`, `CowStorage`).
pub mod storage;

// Re-export the most commonly used items
pub use backend::{Backend, BackendError, ComputeBackend, MoiraiBackend, SequentialBackend};
pub use dtype::{
    BinaryOp, Complex, CpuUnaryDispatch, CpuUnaryOp, Float, FloatOps, Int, ReductionOp, Scalar,
};
pub use layout::{
    is_contiguous, row_major_strides, ConstLayout, ConstShape, Layout, Shape, Strides,
};
pub use ptr::{SendPtr, SendPtrMut};
pub use storage::{
    CowStorage, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage, Storage, StorageMut,
};
