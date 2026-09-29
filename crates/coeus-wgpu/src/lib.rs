//! # coeus-wgpu
//!
//! Cross-platform WebGPU implementation of the Coeus
//! [`ComputeBackend`] /
//! [`BackendOps`](coeus_ops::BackendOps) surface, built on `hephaestus-wgpu`.
#![deny(missing_docs)]
//! Like the other backends it carries no domain logic — only on-device
//! realizations of the kernel contract the CPU
//! [`SequentialBackend`](coeus_core::SequentialBackend) defines.
//!
//! ## Dispatch architecture
//!
//! Backend operations delegate to provider-owned Hephaestus kernels. Coeus
//! supplies tensor layouts and operation contracts; Hephaestus owns WGSL
//! source generation, metadata, pipeline caching, and command submission.
//! The element type is resolved through [`WgpuScalar`] (`f32`/`i32`/`u32`);
//! float-only operations such as attention remain constrained to `f32`.
//!
//! Attention masks remain provider buffers with explicit borrowed layouts.
//! Mutable outputs retain their owners through shared copy-on-write dispatch.

mod api;
mod backend;
mod fusion;
#[cfg(test)]
mod storage;

pub use api::{add, evaluate_fused, evaluate_fused_reduce, matmul};
pub use backend::{WgpuBackend, WgpuBackendError, WgpuScalar};
