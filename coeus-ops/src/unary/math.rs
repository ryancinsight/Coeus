// ── Unary math ops ──

use coeus_core::{Scalar, Float};
use coeus_tensor::Tensor;
use crate::backend_ops::{BackendOps, UnaryOp};
use super::kernel::{elementwise_unary, elementwise_unary_assign};

/// Element-wise sine.
#[inline]
pub fn sin<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sin)
}

/// Element-wise cosine.
#[inline]
pub fn cos<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Cos)
}

/// Element-wise exponential.
#[inline]
pub fn exp<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Exp)
}

/// Element-wise natural log.
#[inline]
pub fn log<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Log)
}

/// Element-wise negation (works for any Scalar).
#[inline]
pub fn neg<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Neg)
}

/// Element-wise absolute value.
#[inline]
pub fn abs<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Abs)
}

/// Element-wise square root.
#[inline]
pub fn sqrt<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sqrt)
}

/// In-place element-wise sine.
#[inline]
pub fn sin_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sin);
}

/// In-place element-wise cosine.
#[inline]
pub fn cos_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Cos);
}

/// In-place element-wise exponential.
#[inline]
pub fn exp_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Exp);
}

/// In-place element-wise natural log.
#[inline]
pub fn log_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Log);
}

/// In-place element-wise negation.
#[inline]
pub fn neg_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Neg);
}

/// In-place element-wise absolute value.
#[inline]
pub fn abs_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Abs);
}

/// In-place element-wise square root.
#[inline]
pub fn sqrt_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sqrt);
}
