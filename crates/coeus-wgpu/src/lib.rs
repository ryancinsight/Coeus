//! # coeus-wgpu
//!
//! Cross-platform WebGPU integration of the Coeus
//! [`ComputeBackend`] /
//! [`BackendOps`](coeus_ops::BackendOps) surface through the generic
//! [`HephaestusBackend<WgpuProvider>`](coeus_hephaestus::HephaestusBackend),
//! built on `hephaestus-wgpu`.
#![deny(missing_docs)]
//! Like the other backends it carries no domain logic — only the provider
//! binding (device acquisition plus operation-bundle selection) over the
//! kernel contracts the shared Coeus-Hephaestus bridge defines.
//!
//! ## Dispatch architecture
//!
//! Backend operations delegate to provider-owned Hephaestus kernels. Coeus
//! supplies tensor layouts and operation contracts; Hephaestus owns WGSL
//! source generation, metadata, pipeline caching, and command submission.
//! The element type resolves through the per-operator provider bounds
//! (`f32`/`i32`/`u32` for elementwise and reductions); float-only operations
//! such as attention remain constrained to `f32`.
//!
//! Attention masks remain provider buffers with explicit borrowed layouts.
//! Mutable outputs retain their owners through shared copy-on-write dispatch.
//!
//! ## Fused expressions
//!
//! Fused elementwise and reduction evaluation lives in
//! [`coeus_hephaestus::evaluate_fused`] /
//! [`coeus_hephaestus::evaluate_fused_reduce`], monomorphized over
//! [`WgpuProvider`]. Accelerator expressions cannot enter the CPU evaluator:
//!
//! ```compile_fail,E0277
//! use coeus_hephaestus::HephaestusBackend;
//! use coeus_ops::fuse::{evaluate_fused_cpu, TensorExprExt};
//! use coeus_tensor::Tensor;
//! use coeus_wgpu::WgpuProvider;
//!
//! fn reject_cpu_evaluator(
//!     tensor: &Tensor<f32, HephaestusBackend<WgpuProvider>>,
//!     backend: &HephaestusBackend<WgpuProvider>,
//! ) {
//!     let expression = tensor.expr();
//!     let _ = evaluate_fused_cpu(&expression, backend);
//! }
//! ```
//!
//! ```compile_fail,E0277
//! use coeus_hephaestus::HephaestusBackend;
//! use coeus_ops::fuse::{evaluate_fused_reduce_cpu, TensorExprExt};
//! use coeus_ops::ReductionOp;
//! use coeus_tensor::Tensor;
//! use coeus_wgpu::WgpuProvider;
//!
//! fn reject_cpu_evaluator(
//!     tensor: &Tensor<f32, HephaestusBackend<WgpuProvider>>,
//!     backend: &HephaestusBackend<WgpuProvider>,
//! ) {
//!     let expression = tensor.expr();
//!     let _ = evaluate_fused_reduce_cpu(&expression, ReductionOp::Sum, 0, backend);
//! }
//! ```

mod backend;
#[cfg(test)]
mod storage;

pub use backend::{HephaestusBackend, WgpuProvider};

use coeus_core::{BackendError, Layout, Scalar};
use coeus_hephaestus::HephaestusBackendError;
use coeus_tensor::Tensor;

/// Element-wise addition of two WebGPU tensors.
///
/// # Errors
///
/// Returns [`HephaestusBackendError`] when the input shapes differ.
pub fn add<T: Scalar>(
    a: &Tensor<T, HephaestusBackend<WgpuProvider>>,
    b: &Tensor<T, HephaestusBackend<WgpuProvider>>,
) -> Result<Tensor<T, HephaestusBackend<WgpuProvider>>, HephaestusBackendError>
where
    HephaestusBackend<WgpuProvider>: coeus_ops::ElementwiseOps<T>,
{
    if a.shape() != b.shape() {
        return Err(BackendError::ShapeMismatch {
            operation: "add",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }
        .into());
    }
    coeus_ops::elementwise_binary(
        a,
        b,
        &HephaestusBackend::<WgpuProvider>::new(),
        coeus_ops::BinaryOp::Add,
    )
}

/// Matrix multiplication of two WebGPU tensors: c = a x b.
///
/// # Errors
///
/// Returns [`HephaestusBackendError`] when the input ranks, inner dimensions,
/// output element count, or WGSL layout metadata violate the backend contract.
pub fn matmul<
    T: Scalar
        + leto_ops::Scalar
        + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>
        + hephaestus_wgpu::MatmulZero,
>(
    a: &Tensor<T, HephaestusBackend<WgpuProvider>>,
    b: &Tensor<T, HephaestusBackend<WgpuProvider>>,
) -> Result<Tensor<T, HephaestusBackend<WgpuProvider>>, HephaestusBackendError> {
    let a_shape = a.shape();
    let [m, k] = a_shape else {
        return Err(BackendError::UnsupportedRank {
            operation: "matmul",
            rank: a_shape.len(),
            max_rank: 2,
        }
        .into());
    };
    let b_shape = b.shape();
    let [k2, n] = b_shape else {
        return Err(BackendError::UnsupportedRank {
            operation: "matmul",
            rank: b_shape.len(),
            max_rank: 2,
        }
        .into());
    };
    if k != k2 {
        return Err(BackendError::ShapeMismatch {
            operation: "matmul",
            lhs: a_shape.to_vec(),
            rhs: b_shape.to_vec(),
        }
        .into());
    }

    let element_count = m.checked_mul(*n).ok_or(BackendError::Overflow {
        operation: "matmul",
        reason: "output element count overflow",
    })?;
    let mut c_storage = coeus_hephaestus::HephaestusStorage::<WgpuProvider, _>::new(element_count);
    let c_layout = Layout::new([*m, *n].into());

    coeus_ops::MatmulOps::matmul(
        &HephaestusBackend::<WgpuProvider>::new(),
        a.storage(),
        a.layout(),
        b.storage(),
        b.layout(),
        &mut c_storage,
        &c_layout,
    )?;

    Ok(Tensor::from_raw_parts(c_storage, c_layout))
}
