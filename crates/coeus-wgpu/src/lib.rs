//! # coeus-wgpu
//!
//! Cross-platform WebGPU implementation of the Coeus
//! [`ComputeBackend`](coeus_core::ComputeBackend) /
//! [`BackendOps`](coeus_ops::BackendOps) surface, built on `hephaestus-wgpu`.
#![deny(missing_docs)]
//! Like the other backends it carries no domain logic — only on-device
//! realizations of the kernel contract the CPU
//! [`SequentialBackend`](coeus_core::SequentialBackend) defines.
//!
//! ## Dispatch architecture
//!
//! Each `BackendOps<T>` method dispatches to a WGSL compute shader in the
//! `kernels` module. Shaders are generated as `T::WGSL_TYPE`-templated source
//! strings, compiled once and cached by the pipeline cache, then bound against
//! the raw `wgpu::Buffer` behind each [`WgpuStorage`]. The element type is
//! resolved through [`WgpuScalar`] (`f32`/`i32`/`u32`); float-only kernels such
//! as attention are written for `f32`, the only `Float + WgpuScalar` type.
//!
//! ## Capability boundaries
//!
//! Every operation exposed by this backend stays on the WGPU dispatch path.
//! Unsupported layouts return a typed capability error; they are not moved to
//! the CPU evaluator. Differential tests use the explicit CPU reference as an
//! independent oracle, never as an implicit execution fallback.

mod backend;
mod kernels;
mod storage;

pub use backend::{LayoutError, WgpuBackend, WgpuBackendError, WgpuScalar};
pub use storage::WgpuStorage;

use coeus_core::{BackendError, Layout};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;

fn validate_fused_inputs<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
) -> Result<(), WgpuBackendError> {
    let mut input_ptrs = Vec::new();
    expr.collect_inputs(&mut input_ptrs);
    if input_ptrs.iter().any(|ptr| ptr.is_null()) {
        return Err(BackendError::Storage {
            operation: "wgpu fused evaluation",
            reason: "expression contains a null tensor input pointer".to_owned(),
        }
        .into());
    }
    for ptr in input_ptrs {
        // SAFETY: `ExprNode::collect_inputs` returns pointers to tensors held by
        // the expression. Null pointers are rejected before dereferencing and
        // the expression borrow keeps the captured tensors alive for dispatch.
        let input = unsafe { &*ptr };
        backend::WgpuBackendError::validate_layout(input.layout())?;
    }
    Ok(())
}

/// Element-wise addition of two WebGPU tensors.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when the input shapes differ.
pub fn add<T: WgpuScalar>(
    a: &Tensor<T, WgpuBackend>,
    b: &Tensor<T, WgpuBackend>,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError> {
    if a.shape() != b.shape() {
        return Err(BackendError::ShapeMismatch {
            operation: "add",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }
        .into());
    }
    let len = a.numel();

    let c_storage = WgpuStorage::try_new(len)?;

    kernels::dispatch_contiguous_binary::<T>(
        coeus_ops::BinaryOp::Add,
        a.storage().buffer.raw(),
        b.storage().buffer.raw(),
        c_storage.buffer.raw(),
        len,
    )?;

    Ok(Tensor::from_raw_parts(
        c_storage,
        Layout::new(a.shape_cloned()),
    ))
}

/// Matrix multiplication of two WebGPU tensors: c = a x b.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when the input ranks, inner dimensions, output
/// element count, or WGSL layout metadata violate the backend contract.
pub fn matmul<T: WgpuScalar>(
    a: &Tensor<T, WgpuBackend>,
    b: &Tensor<T, WgpuBackend>,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError> {
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
    let c_storage = WgpuStorage::try_new(element_count)?;
    let c_layout = Layout::new([*m, *n].into());

    kernels::dispatch_matmul::<T>(
        a.storage().buffer.raw(),
        a.layout(),
        b.storage().buffer.raw(),
        b.layout(),
        c_storage.buffer.raw(),
        &c_layout,
    )
    .map_err(|error| WgpuBackendError::Layout(error.into()))?;

    Ok(Tensor::from_raw_parts(c_storage, c_layout))
}

/// Evaluate a fused element-wise expression on the WebGPU device.
///
/// # Errors
///
/// Returns a typed validation or allocation error. Device dispatch is not
/// replaced with a host evaluator when the selected backend is WGPU.
pub fn evaluate_fused<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError> {
    let out_shape = expr.shape().ok_or_else(|| BackendError::Storage {
        operation: "wgpu fused evaluation",
        reason: "expression has no tensor input to determine shape".to_owned(),
    })?;
    let out_layout = Layout::new(out_shape.clone());
    validate_fused_inputs(expr)?;
    backend::WgpuBackendError::validate_layout(&out_layout)?;
    backend::checked_workgroup_count("fused evaluation", out_layout.numel())?;
    let mut out_storage = WgpuStorage::try_new(out_layout.numel())?;

    kernels::dispatch_fused(expr, &mut out_storage, &out_layout);

    Ok(Tensor::from_raw_parts(out_storage, out_layout))
}

/// Evaluate a fused reduction along an axis on the WebGPU device.
///
/// # Errors
///
/// Returns a typed validation or allocation error. Device dispatch is not
/// replaced with a host evaluator when the selected backend is WGPU.
pub fn evaluate_fused_reduce<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError> {
    let expr_shape = expr.shape().ok_or_else(|| BackendError::Storage {
        operation: "wgpu fused reduction",
        reason: "expression has no tensor input to determine shape".to_owned(),
    })?;
    if axis >= expr_shape.len() {
        return Err(BackendError::AxisOutOfRange {
            operation: "wgpu fused reduction",
            axis,
            rank: expr_shape.len(),
        }
        .into());
    }
    validate_fused_inputs(expr)?;

    let mut out_shape = expr_shape;
    out_shape[axis] = 1;
    let out_layout = Layout::new(out_shape.clone());
    backend::WgpuBackendError::validate_layout(&out_layout)?;
    backend::checked_workgroup_count("fused reduction", out_layout.numel())?;
    let mut out_storage = WgpuStorage::try_new(out_layout.numel())?;

    kernels::dispatch_fused_reduce(expr, op, axis, &mut out_storage, &out_layout);

    Ok(Tensor::from_raw_parts(out_storage, out_layout))
}
