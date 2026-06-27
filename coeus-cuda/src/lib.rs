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
//! via `hephaestus-cuda`). Without it, [`CudaBackend`] resolves to a stub so the
//! workspace builds on machines without a CUDA toolkit; only `--features cuda`
//! exercises the GPU.
//!
//! ## Dispatch architecture
//!
//! Each `BackendOps<T>` method routes to a `cuda_*` method that:
//! 1. checks a live CUDA context exists and `T == f32` (the kernels are
//!    monomorphized for `f32`; the [`TypeId`](std::any::TypeId) guard plus the
//!    zero-copy reinterpret in `backend::ops::cast` keep the generic surface
//!    honest without a fake-generic widen/narrow);
//! 2. launches the on-device kernel (hand-written PTX in the `kernels` module for
//!    conv/attention, NVRTC CUDA C for fused/elementwise/optimizer paths);
//! 3. falls back to the CPU reference (`fallback::*`, a host round-trip) only
//!    when the on-device path is unavailable or for an explicitly documented
//!    capability boundary (e.g. a strided key-padding mask in attention).
//!
//! The fallback is a capability boundary, never a silent defect mask: the
//! observable result matches the CPU reference either way, verified by the
//! differential parity tests in `tests/cuda/`.
#![deny(missing_docs)]

#[cfg(feature = "cuda")]
mod backend;
#[cfg(not(feature = "cuda"))]
#[path = "backend_stub.rs"]
mod backend;

#[cfg(feature = "cuda")]
/// CUDA driver context management for the real device backend.
pub mod driver;
#[cfg(not(feature = "cuda"))]
#[path = "driver_stub.rs"]
/// Stub CUDA driver surface used when the `cuda` feature is disabled.
pub mod driver;

mod fallback;
#[cfg(feature = "cuda")]
/// CUDA kernel modules and launch helpers for on-device computation.
pub mod kernels;

#[cfg(feature = "cuda")]
mod storage;
#[cfg(not(feature = "cuda"))]
#[path = "storage_stub.rs"]
mod storage;

pub use backend::{CudaBackend, CudaScalar};
pub use driver::{get_cuda_context, CudaDriver};
pub use storage::CudaStorage;

#[cfg(feature = "cuda")]
use coeus_core::Layout;
use coeus_tensor::Tensor;

/// Evaluate a fused element-wise expression on the CUDA device.
///
/// Compiles and dispatches a dynamic kernel on the GPU, falling back to CPU if unavailable.
pub fn evaluate_fused<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
) -> Tensor<T, CudaBackend> {
    #[cfg(not(feature = "cuda"))]
    {
        coeus_ops::fuse::evaluate_fused_cpu(expr, &CudaBackend::new())
    }

    #[cfg(feature = "cuda")]
    {
        let out_shape = expr
            .shape()
            .expect("Fused expression must have at least one tensor input to determine shape");
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        if kernels::dispatch_fused(expr, out.storage_mut(), &out_layout) {
            out
        } else {
            coeus_ops::fuse::evaluate_fused_cpu(expr, &CudaBackend::new())
        }
    }
}

/// Evaluate a fused reduction along an axis on the CUDA device.
pub fn evaluate_fused_reduce<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Tensor<T, CudaBackend> {
    #[cfg(not(feature = "cuda"))]
    {
        coeus_ops::fuse::evaluate_fused_reduce_cpu(expr, op, axis, &CudaBackend::new())
    }

    #[cfg(feature = "cuda")]
    {
        let expr_shape = expr
            .shape()
            .expect("Fused expression must have at least one tensor input to determine shape");
        assert!(
            axis < expr_shape.len(),
            "Axis out of bounds in evaluate_fused_reduce"
        );

        let mut out_shape = expr_shape;
        out_shape[axis] = 1;
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        if kernels::dispatch_fused_reduce(expr, op, axis, out.storage_mut(), &out_layout) {
            out
        } else {
            coeus_ops::fuse::evaluate_fused_reduce_cpu(expr, op, axis, &CudaBackend::new())
        }
    }
}
