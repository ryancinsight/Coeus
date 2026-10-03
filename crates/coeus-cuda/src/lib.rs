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

pub use backend::{CudaBackend, CudaScalar};

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
///
/// Accelerator expressions cannot enter the CPU evaluator:
///
/// ```compile_fail,E0277
/// use coeus_cuda::CudaBackend;
/// use coeus_ops::fuse::{evaluate_fused_cpu, TensorExprExt};
/// use coeus_tensor::Tensor;
///
/// fn reject_cpu_evaluator(tensor: &Tensor<f32, CudaBackend>, backend: &CudaBackend) {
///     let expression = tensor.expr();
///     let _ = evaluate_fused_cpu(&expression, backend);
/// }
/// ```
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
        let out_shape = expr.shape()?.ok_or_else(|| {
            CudaBackendError::validation(coeus_core::BackendError::Storage {
                operation: "fused elementwise",
                reason: "expression has no tensor input from which to derive its shape".to_string(),
            })
        })?;
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        fusion::dispatch_fused(expr, out.storage_mut(), &out_layout)?;
        Ok(out)
    }
}

/// Evaluate a fused reduction along an axis on the CUDA device.
///
/// Mean is float-only and evaluates through [`evaluate_fused_mean`].
///
/// # Errors
///
/// Returns [`CudaBackendError`] carrying
/// [`BackendError::FloatOnlyReduction`](coeus_core::BackendError::FloatOnlyReduction)
/// for [`ReductionOp::Mean`](coeus_ops::ReductionOp::Mean), and when the
/// expression, axis, CUDA provider, generated kernel, or launch ABI rejects
/// the operation. Empty maximum and minimum reductions are undefined and
/// rejected.
///
/// Accelerator expressions cannot enter the CPU reduction evaluator:
///
/// ```compile_fail,E0277
/// use coeus_cuda::CudaBackend;
/// use coeus_ops::fuse::{evaluate_fused_reduce_cpu, TensorExprExt};
/// use coeus_ops::ReductionOp;
/// use coeus_tensor::Tensor;
///
/// fn reject_cpu_evaluator(tensor: &Tensor<f32, CudaBackend>, backend: &CudaBackend) {
///     let expression = tensor.expr();
///     let _ = evaluate_fused_reduce_cpu(&expression, ReductionOp::Sum, 0, backend);
/// }
/// ```
pub fn evaluate_fused_reduce<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Result<Tensor<T, CudaBackend>, CudaBackendError> {
    let reduction = coeus_core::ClosedReduction::from_op("fused reduction", op)
        .map_err(CudaBackendError::validation)?;
    fused_reduction(expr, reduction.into(), axis)
}

/// Evaluate a fused arithmetic mean along an axis on the CUDA device.
///
/// The `FloatElement` bound makes integer mean unrepresentable: integer
/// division would truncate the quotient.
///
/// # Errors
///
/// Returns [`CudaBackendError`] when the expression, axis, CUDA provider,
/// generated kernel, or launch ABI rejects the operation, or the axis is
/// empty.
pub fn evaluate_fused_mean<
    T: CudaScalar + coeus_core::FloatElement,
    E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy,
>(
    expr: &E,
    axis: usize,
) -> Result<Tensor<T, CudaBackend>, CudaBackendError> {
    fused_reduction(expr, coeus_ops::ReductionOp::Mean, axis)
}

/// Shared fused reduction body. `op` is a closed reduction, or the mean from
/// [`evaluate_fused_mean`] alone.
fn fused_reduction<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
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
        let expr_shape = expr.shape()?.ok_or_else(|| {
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
        let axis_len = expr_shape[axis];
        coeus_ops::fuse::validate_fused_reduction_axis(op, axis_len)
            .map_err(CudaBackendError::validation)?;

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

        fusion::dispatch_fused_reduce(
            expr,
            coeus_hephaestus::fused_selector(op),
            axis,
            out.storage_mut(),
            &out_layout,
        )?;
        Ok(out)
    }
}
