// ── Activation functions ──
// Element-wise neural network activation functions.

use coeus_core::{Scalar, Float};
use coeus_tensor::Tensor;
use crate::backend_ops::{BackendOps, UnaryOp};
use super::kernel::{elementwise_unary, elementwise_unary_assign};

/// Rectified Linear Unit: max(0, x).
#[inline]
pub fn relu<T: Scalar, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Relu)
}

/// Sigmoid: 1 / (1 + exp(-x)).
#[inline]
pub fn sigmoid<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Sigmoid)
}

/// Hyperbolic tangent.
#[inline]
pub fn tanh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Tanh)
}

/// GELU (Gaussian Error Linear Unit).
/// Approx: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Gelu)
}

/// In-place Rectified Linear Unit: max(0, x).
#[inline]
pub fn relu_assign<T: Scalar, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Relu);
}

/// In-place Sigmoid: 1 / (1 + exp(-x)).
#[inline]
pub fn sigmoid_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Sigmoid);
}

/// In-place Hyperbolic tangent.
#[inline]
pub fn tanh_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Tanh);
}

/// In-place GELU.
#[inline]
pub fn gelu_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Gelu);
}

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x).
#[inline]
pub fn silu<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Silu)
}

/// In-place SiLU.
#[inline]
pub fn silu_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Silu);
}

/// Mish (Self-Regularized Non-Monotonic Activation Function): x * tanh(softplus(x)).
#[inline]
pub fn mish<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Mish)
}

/// In-place Mish.
#[inline]
pub fn mish_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Mish);
}

/// ELU (Exponential Linear Unit): x >= 0 ? x : exp(x) - 1.
#[inline]
pub fn elu<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Elu)
}

/// In-place ELU.
#[inline]
pub fn elu_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Elu);
}

/// Softplus: log(1 + exp(x)).
#[inline]
pub fn softplus<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::Softplus)
}

/// In-place Softplus.
#[inline]
pub fn softplus_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::Softplus);
}

/// GELU tanh approximation.
#[inline]
pub fn gelu_tanh<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::GeluTanh)
}

/// In-place GELU tanh approximation.
#[inline]
pub fn gelu_tanh_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B) {
    elementwise_unary_assign(input, backend, UnaryOp::GeluTanh);
}

/// LeakyReLU: x >= 0 ? x : negative_slope * x.
#[inline]
pub fn leaky_relu<T: Float, B: BackendOps<T>>(input: &Tensor<T, B>, backend: &B, negative_slope: f64) -> Tensor<T, B> {
    elementwise_unary(input, backend, UnaryOp::LeakyRelu(f64::to_bits(negative_slope)))
}

/// In-place LeakyReLU.
#[inline]
pub fn leaky_relu_assign<T: Float, B: BackendOps<T>>(input: &mut Tensor<T, B>, backend: &B, negative_slope: f64) {
    elementwise_unary_assign(input, backend, UnaryOp::LeakyRelu(f64::to_bits(negative_slope)));
}

/// Numerically-stable log-softmax along `axis`.
///
/// `log_softmax(x)_i = x_i - max(x) - log(Σ_j exp(x_j - max(x)))`
///
/// Implemented via tensor ops: max_axis → sub → exp → sum_axis → log → sub.
/// Returns a tensor of the same shape as `input`.
#[inline]
pub fn log_softmax_axis<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    let ndim = input.ndim();
    assert!(axis < ndim, "log_softmax_axis: axis {axis} out of bounds for ndim {ndim}");
    // Shift by max for numerical stability: shifted = x - max(x, axis)
    let max_vals = super::super::reduction::max_axis(input, axis, backend);
    let shifted = super::super::binary::sub(input, &max_vals, backend);
    // exp(shifted)
    let exp_shifted = elementwise_unary(&shifted, backend, UnaryOp::Exp);
    // sum(exp(shifted), axis)
    let sum_exp = super::super::reduction::sum_axis(&exp_shifted, axis, backend);
    // log(sum_exp)
    let log_sum_exp = elementwise_unary(&sum_exp, backend, UnaryOp::Log);
    // out = shifted - log_sum_exp  (broadcasts log_sum_exp along axis)
    super::super::binary::sub(&shifted, &log_sum_exp, backend)
}
