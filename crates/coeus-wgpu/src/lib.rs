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
//! ## CPU-reference boundaries
//!
//! A few paths fall back to the CPU reference via host transfer — currently a
//! strided key-padding mask in attention. This is an explicit capability
//! boundary, not a silent defect mask: the observable result matches the CPU
//! reference, verified by the differential parity tests in `tests/wgpu/`, and
//! the on-device speedup over that reference is tracked in `benches/`.

mod backend;
mod fusion;
mod kernels;
mod storage;

pub use backend::{LayoutError, WgpuBackend, WgpuBackendError, WgpuScalar};
pub use storage::WgpuStorage;

use coeus_core::{BackendError, ComputeBackend, Layout};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;

/// Element-wise addition of two WebGPU tensors.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when the input shapes differ.
pub fn add<T: WgpuScalar>(
    a: &Tensor<T, WgpuBackend>,
    b: &Tensor<T, WgpuBackend>,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError>
where
    WgpuBackend: coeus_ops::ElementwiseOps<T>,
{
    if a.shape() != b.shape() {
        return Err(BackendError::ShapeMismatch {
            operation: "add",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }
        .into());
    }
    coeus_ops::elementwise_binary(a, b, &WgpuBackend, coeus_ops::BinaryOp::Add)
}

/// Matrix multiplication of two WebGPU tensors: c = a x b.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when the input ranks, inner dimensions, output
/// element count, or WGSL layout metadata violate the backend contract.
pub fn matmul<
    T: WgpuScalar
        + leto_ops::Scalar
        + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>
        + hephaestus_wgpu::MatmulZero,
>(
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
    let mut c_storage = WgpuStorage::new(element_count);
    let c_layout = Layout::new([*m, *n].into());

    coeus_ops::MatmulOps::matmul(
        &WgpuBackend,
        a.storage(),
        a.layout(),
        b.storage(),
        b.layout(),
        &mut c_storage,
        &c_layout,
    )?;

    Ok(Tensor::from_raw_parts(c_storage, c_layout))
}

/// Evaluate a fused element-wise expression on the WebGPU device.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when the expression has no tensor input or its
/// child shapes cannot be broadcast.
///
/// Accelerator expressions cannot enter the CPU evaluator:
///
/// ```compile_fail,E0277
/// use coeus_ops::fuse::{evaluate_fused_cpu, TensorExprExt};
/// use coeus_tensor::Tensor;
/// use coeus_wgpu::WgpuBackend;
///
/// fn reject_cpu_evaluator(tensor: &Tensor<f32, WgpuBackend>, backend: &WgpuBackend) {
///     let expression = tensor.expr();
///     let _ = evaluate_fused_cpu(&expression, backend);
/// }
/// ```
pub fn evaluate_fused<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError> {
    let out_shape = expr.shape()?.ok_or_else(|| {
        WgpuBackendError::Validation(coeus_core::BackendError::Storage {
            operation: "fused expression",
            reason: "expression has no tensor input from which to derive its shape".to_string(),
        })
    })?;
    let out_layout = Layout::new(out_shape);
    let mut out_storage = WgpuStorage::new(out_layout.numel());

    fusion::dispatch_fused(expr, &mut out_storage, &out_layout)?;

    Ok(Tensor::from_raw_parts(out_storage, out_layout))
}

/// Evaluate a fused reduction along an axis on the WebGPU device.
///
/// # Errors
///
/// Returns [`WgpuBackendError`] when the expression has no tensor input, the
/// axis is invalid, an empty axis is used with mean, maximum, or minimum, or
/// the layout and dispatch cannot be represented by the active WebGPU device.
///
/// Accelerator expressions cannot enter the CPU reduction evaluator:
///
/// ```compile_fail,E0277
/// use coeus_ops::fuse::{evaluate_fused_reduce_cpu, TensorExprExt};
/// use coeus_ops::ReductionOp;
/// use coeus_tensor::Tensor;
/// use coeus_wgpu::WgpuBackend;
///
/// fn reject_cpu_evaluator(tensor: &Tensor<f32, WgpuBackend>, backend: &WgpuBackend) {
///     let expression = tensor.expr();
///     let _ = evaluate_fused_reduce_cpu(&expression, ReductionOp::Sum, 0, backend);
/// }
/// ```
pub fn evaluate_fused_reduce<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Result<Tensor<T, WgpuBackend>, WgpuBackendError> {
    const OPERATION: &str = "fused reduction";

    let mut inputs = Vec::new();
    expr.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(WgpuBackendError::Validation(
            coeus_core::BackendError::Storage {
                operation: OPERATION,
                reason: "expression contains no tensor inputs".to_string(),
            },
        ));
    }
    let expr_shape = expr.shape()?.ok_or_else(|| {
        WgpuBackendError::Validation(coeus_core::BackendError::Storage {
            operation: OPERATION,
            reason: "expression has no tensor input from which to derive its shape".to_string(),
        })
    })?;

    let out_rank = expr_shape.len();
    let axis_len = *expr_shape.get(axis).ok_or({
        WgpuBackendError::Validation(coeus_core::BackendError::AxisOutOfRange {
            operation: OPERATION,
            axis,
            rank: out_rank,
        })
    })?;
    let mut out_shape = expr_shape.clone();
    let output_axis = out_shape.get_mut(axis).ok_or({
        WgpuBackendError::Validation(coeus_core::BackendError::AxisOutOfRange {
            operation: OPERATION,
            axis,
            rank: out_rank,
        })
    })?;
    *output_axis = 1;
    let out_layout = Layout::new(out_shape.clone());
    let out_numel = backend::checked_numel(OPERATION, out_layout.shape())?;
    let mut out_storage = WgpuStorage::new(out_numel);

    if axis_len == 0 {
        let identity = match op {
            coeus_ops::ReductionOp::Sum => T::zero(),
            coeus_ops::ReductionOp::Prod => T::one(),
            coeus_ops::ReductionOp::Mean
            | coeus_ops::ReductionOp::Max
            | coeus_ops::ReductionOp::Min => {
                return Err(BackendError::EmptyReduction {
                    operation: OPERATION,
                    reduction: op,
                }
                .into());
            }
        };
        WgpuBackend::new().fill(&mut out_storage, identity);
        return Ok(Tensor::from_raw_parts(out_storage, out_layout));
    }

    fusion::dispatch_fused_reduce(expr, op, axis, &mut out_storage, &out_layout)?;

    Ok(Tensor::from_raw_parts(out_storage, out_layout))
}
