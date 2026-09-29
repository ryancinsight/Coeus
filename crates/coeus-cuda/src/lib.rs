//! # coeus-cuda
//!
//! NVIDIA CUDA implementation of the Coeus [`ComputeBackend`](coeus_core::ComputeBackend)
//! / [`BackendOps`](coeus_ops::BackendOps) surface. The crate is a pure backend:
//! it adds no domain logic, only on-device realizations of the kernel contract
//! the CPU [`SequentialBackend`](coeus_core::SequentialBackend) defines.
//!
//! ## Feature gating
//!
//! The real device path is behind the `cuda` feature (NVRTC + the CUDA driver
//! through `hephaestus-cuda`). Without it, [`CudaBackend`] exposes metadata and
//! storage types so the workspace builds on machines without a CUDA toolkit,
//! but it implements no mathematical backend traits.
//!
//! ## Dispatch architecture
//!
//! Attention, convolution, and stateful optimizer updates bind directly to
//! provider-owned Hephaestus operation markers over borrowed CUDA buffers.
//! All mathematical operations route to monomorphized provider-owned
//! Hephaestus kernels and return typed backend errors when the selected
//! provider rejects validation, compilation, or dispatch. No operation changes
//! execution backend after CUDA has been selected.
//!
//! Provider capability boundaries are explicit in their operation contracts
//! and are covered by differential parity tests in `tests/cuda/`. Native and
//! fused CUDA entry points return provider failures to the caller rather than
//! changing execution backends.
#![deny(missing_docs)]

mod error;
pub use error::CudaBackendError;

#[cfg(feature = "cuda")]
mod backend;
#[cfg(not(feature = "cuda"))]
#[path = "backend_stub.rs"]
mod backend;

#[cfg(all(test, feature = "cuda"))]
mod storage;

#[cfg(feature = "cuda")]
mod fusion;

mod fuse_api;

pub use backend::{CudaBackend, CudaScalar};
pub use fuse_api::{evaluate_fused, evaluate_fused_reduce};
