//! # coeus-wgpu
//!
//! Cross-platform WebGPU implementation of the Coeus
//! [`ComputeBackend`](coeus_core::ComputeBackend) /
//! [`BackendOps`](coeus_ops::BackendOps) surface, built on `hephaestus-wgpu`.
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
//! ## CPU-reference boundaries
//!
//! A few paths fall back to the CPU reference via host transfer — currently a
//! strided key-padding mask in attention. This is an explicit capability
//! boundary, not a silent defect mask: the observable result matches the CPU
//! reference, verified by the differential parity tests in `tests/wgpu/`, and
//! the on-device speedup over that reference is tracked in `benches/`.

mod backend;
mod kernels;
mod storage;

pub use backend::{WgpuBackend, WgpuScalar};
pub use storage::WgpuStorage;

use coeus_core::Layout;
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;

/// Element-wise addition of two WebGPU tensors.
pub fn add<T: WgpuScalar>(
    a: &Tensor<T, WgpuBackend>,
    b: &Tensor<T, WgpuBackend>,
) -> Tensor<T, WgpuBackend> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch in wgpu add");
    let len = a.numel();

    let c_storage = WgpuStorage::new(len);

    kernels::dispatch_contiguous_binary::<T>(
        coeus_ops::BinaryOp::Add,
        a.storage().buffer.raw(),
        b.storage().buffer.raw(),
        c_storage.buffer.raw(),
        len,
    );

    Tensor::from_raw_parts(c_storage, Layout::new(a.shape_cloned()))
}

/// Matrix multiplication of two WebGPU tensors: c = a x b.
pub fn matmul<T: WgpuScalar>(
    a: &Tensor<T, WgpuBackend>,
    b: &Tensor<T, WgpuBackend>,
) -> Tensor<T, WgpuBackend> {
    assert_eq!(a.ndim(), 2, "matmul requires 2D input A");
    assert_eq!(b.ndim(), 2, "matmul requires 2D input B");
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    assert_eq!(k, k2, "matmul inner dimension mismatch: {} vs {}", k, k2);

    let c_storage = WgpuStorage::new(m * n);
    let c_layout = Layout::new([m, n].into());

    kernels::dispatch_matmul::<T>(
        a.storage().buffer.raw(),
        a.layout(),
        b.storage().buffer.raw(),
        b.layout(),
        c_storage.buffer.raw(),
        &c_layout,
    );

    Tensor::from_raw_parts(c_storage, c_layout)
}

/// Evaluate a fused element-wise expression on the WebGPU device.
pub fn evaluate_fused<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
) -> Tensor<T, WgpuBackend> {
    let out_shape = expr
        .shape()
        .expect("Fused expression must have at least one tensor input to determine shape");
    let out_layout = Layout::new(out_shape.clone());
    let mut out_storage = WgpuStorage::new(out_layout.numel());

    kernels::dispatch_fused(expr, &mut out_storage, &out_layout);

    Tensor::from_raw_parts(out_storage, out_layout)
}

/// Evaluate a fused reduction along an axis on the WebGPU device.
pub fn evaluate_fused_reduce<T: WgpuScalar, E: ExprNode<T, WgpuBackend>>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Tensor<T, WgpuBackend> {
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
    let mut out_storage = WgpuStorage::new(out_layout.numel());

    kernels::dispatch_fused_reduce(expr, op, axis, &mut out_storage, &out_layout);

    Tensor::from_raw_parts(out_storage, out_layout)
}
