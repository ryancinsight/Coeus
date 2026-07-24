// ── Binary comparison ops (mask) ──

use super::kernel::elementwise_binary;
use crate::backend_ops::{BackendOps, BinaryOp};
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Element-wise equality comparison mask.
#[inline]
pub fn eq<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Eq).expect("eq: incompatible shapes")
}

/// Element-wise inequality comparison mask.
#[inline]
pub fn ne<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Ne).expect("ne: incompatible shapes")
}

/// Element-wise less-than comparison mask.
#[inline]
pub fn lt<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Lt).expect("lt: incompatible shapes")
}

/// Element-wise greater-than comparison mask.
#[inline]
pub fn gt<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Gt).expect("gt: incompatible shapes")
}

/// Element-wise less-than-or-equal comparison mask.
#[inline]
pub fn le<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Le).expect("le: incompatible shapes")
}

/// Element-wise greater-than-or-equal comparison mask.
#[inline]
pub fn ge<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    elementwise_binary(a, b, backend, BinaryOp::Ge).expect("ge: incompatible shapes")
}
