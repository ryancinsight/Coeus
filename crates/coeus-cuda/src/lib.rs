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
//! via `hephaestus-cuda`). Without it, [`CudaBackend`] exposes metadata and
//! storage types so the workspace builds on machines without a CUDA toolkit,
//! but it implements no mathematical backend traits.
//!
//! ## Dispatch architecture
//!
//! Attention and convolution bind directly to provider-owned Hephaestus
//! operation markers over borrowed CUDA buffers. Other `BackendOps<T>` methods
//! route to monomorphized on-device kernels and return typed backend errors when
//! the selected provider rejects validation, compilation, or dispatch. No
//! operation changes execution backend after CUDA has been selected.
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

#[cfg(feature = "cuda")]
/// CUDA driver context management for the real device backend.
pub mod driver;
#[cfg(not(feature = "cuda"))]
#[path = "driver_stub.rs"]
/// Stub CUDA driver surface used when the `cuda` feature is disabled.
pub mod driver;

#[cfg(feature = "cuda")]
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
/// The CUDA feature requires a live provider, a valid expression layout, and
/// a successful NVRTC compilation and kernel launch. Provider failure is
/// returned to the caller; this API never changes execution backends after a
/// CUDA dispatch has been selected.
///
/// # Errors
///
/// Returns [`CudaBackendError`] when the expression, CUDA provider, generated
/// kernel, or launch ABI rejects the operation.
pub fn evaluate_fused<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
) -> Result<Tensor<T, CudaBackend>, CudaBackendError> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = expr;
        Err(CudaBackendError::kernel(
            "fused elementwise",
            "the CUDA provider feature is disabled",
        ))
    }

    #[cfg(feature = "cuda")]
    {
        let out_shape = expr.shape().ok_or_else(|| {
            CudaBackendError::validation(coeus_core::BackendError::Storage {
                operation: "fused elementwise",
                reason: "expression has no tensor input from which to derive its shape".to_string(),
            })
        })?;
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        kernels::dispatch_fused(expr, out.storage_mut(), &out_layout)?;
        Ok(out)
    }
}

/// Evaluate a fused reduction along an axis on the CUDA device.
///
/// # Errors
///
/// Returns [`CudaBackendError`] when the expression, axis, CUDA provider,
/// generated kernel, or launch ABI rejects the operation.
pub fn evaluate_fused_reduce<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Result<Tensor<T, CudaBackend>, CudaBackendError> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (expr, op, axis);
        Err(CudaBackendError::kernel(
            "fused reduction",
            "the CUDA provider feature is disabled",
        ))
    }

    #[cfg(feature = "cuda")]
    {
        let expr_shape = expr.shape().ok_or_else(|| {
            CudaBackendError::validation(coeus_core::BackendError::Storage {
                operation: "fused reduction",
                reason: "expression has no tensor input from which to derive its shape".to_string(),
            })
        })?;
        if axis >= expr_shape.len() {
            return Err(CudaBackendError::validation(
                coeus_core::BackendError::AxisOutOfRange {
                    operation: "fused reduction",
                    axis,
                    rank: expr_shape.len(),
                },
            ));
        }

        let mut out_shape = expr_shape;
        let out_rank = out_shape.len();
        let output_axis = out_shape.get_mut(axis).ok_or_else(|| {
            CudaBackendError::validation(coeus_core::BackendError::AxisOutOfRange {
                operation: "fused reduction",
                axis,
                rank: out_rank,
            })
        })?;
        *output_axis = 1;
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        kernels::dispatch_fused_reduce(expr, op, axis, out.storage_mut(), &out_layout)?;
        Ok(out)
    }
}
