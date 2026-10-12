//! # coeus-cuda
//!
//! NVIDIA CUDA implementation of the Coeus [`ComputeBackend`](coeus_core::ComputeBackend)
//! / [`BackendOps`](coeus_ops::BackendOps) surface through the generic
//! [`HephaestusBackend<CudaProvider>`](coeus_hephaestus::HephaestusBackend),
//! built on `hephaestus-cuda`. The crate is a pure backend: it adds no
//! domain logic, only the provider binding (device acquisition plus
//! operation-bundle selection) over the kernel contracts the shared
//! Coeus-Hephaestus bridge defines.
//!
//! ## Feature gating
//!
//! The library binds unconditionally over `hephaestus_cuda::CudaDevice`.
//! Without the `cuda` feature that type is hephaestus-cuda's stub, so the
//! crate still compiles but device acquisition fails honestly with an
//! unavailable-device error. The `cuda` feature enables the real NVRTC +
//! driver backend (plus `themis` transfer hints) and gates the on-device
//! integration tests, storage tests, and benches.
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
//!
//! ## Fused expressions
//!
//! Fused elementwise and reduction evaluation lives in
//! [`coeus_hephaestus::evaluate_fused`] /
//! [`coeus_hephaestus::evaluate_fused_reduce`], monomorphized over
//! [`CudaProvider`]. Accelerator expressions cannot enter the CPU evaluator:
//!
//! ```compile_fail,E0277
//! use coeus_cuda::CudaProvider;
//! use coeus_hephaestus::HephaestusBackend;
//! use coeus_ops::fuse::{evaluate_fused_cpu, TensorExprExt};
//! use coeus_tensor::Tensor;
//!
//! fn reject_cpu_evaluator(
//!     tensor: &Tensor<f32, HephaestusBackend<CudaProvider>>,
//!     backend: &HephaestusBackend<CudaProvider>,
//! ) {
//!     let expression = tensor.expr();
//!     let _ = evaluate_fused_cpu(&expression, backend);
//! }
//! ```
//!
//! ```compile_fail,E0277
//! use coeus_cuda::CudaProvider;
//! use coeus_hephaestus::HephaestusBackend;
//! use coeus_ops::fuse::{evaluate_fused_reduce_cpu, TensorExprExt};
//! use coeus_ops::ReductionOp;
//! use coeus_tensor::Tensor;
//!
//! fn reject_cpu_evaluator(
//!     tensor: &Tensor<f32, HephaestusBackend<CudaProvider>>,
//!     backend: &HephaestusBackend<CudaProvider>,
//! ) {
//!     let expression = tensor.expr();
//!     let _ = evaluate_fused_reduce_cpu(&expression, ReductionOp::Sum, 0, backend);
//! }
//! ```
#![deny(missing_docs)]

mod backend;

#[cfg(all(test, feature = "cuda"))]
mod storage;

pub use backend::{CudaProvider, CudaScalar, HephaestusBackend};
