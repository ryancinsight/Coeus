// ── Unary math ops ──

use super::kernel::{elementwise_unary, elementwise_unary_assign};
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;

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

/// Element-wise reciprocal: 1/x.
#[inline]
pub fn recip<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Recip)
}

/// Element-wise signum: -1, 0, or 1.
#[inline]
pub fn sign<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sign)
}

/// Element-wise floor.
#[inline]
pub fn floor<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Floor)
}

/// Element-wise ceil.
#[inline]
pub fn ceil<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Ceil)
}

/// Element-wise round to nearest integer.
#[inline]
pub fn round<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Round)
}

/// Element-wise truncation toward zero.
#[inline]
pub fn trunc<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Trunc)
}

/// In-place element-wise reciprocal.
#[inline]
pub fn recip_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Recip);
}

/// In-place element-wise signum.
#[inline]
pub fn sign_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sign);
}

/// In-place element-wise floor.
#[inline]
pub fn floor_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Floor);
}

/// In-place element-wise ceil.
#[inline]
pub fn ceil_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Ceil);
}

/// In-place element-wise round.
#[inline]
pub fn round_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Round);
}

/// In-place element-wise truncation.
#[inline]
pub fn trunc_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Trunc);
}
