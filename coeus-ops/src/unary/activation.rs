// ── Activation functions ──
// Element-wise neural network activation functions.

use super::kernel::{elementwise_unary, elementwise_unary_assign};
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, Scalar};
use coeus_tensor::Tensor;

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

/// Exact GELU (Gaussian Error Linear Unit).
///
/// Formula: `0.5 * x * (1 + erf(x / sqrt(2)))`.
/// The tanh approximation is exposed separately as [`gelu_tanh`].
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
pub fn leaky_relu<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
    negative_slope: f64,
) -> Tensor<T, B> {
    elementwise_unary(
        input,
        backend,
        UnaryOp::LeakyRelu(f64::to_bits(negative_slope)),
    )
}

/// In-place LeakyReLU.
#[inline]
pub fn leaky_relu_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
    negative_slope: f64,
) {
    elementwise_unary_assign(
        input,
        backend,
        UnaryOp::LeakyRelu(f64::to_bits(negative_slope)),
    );
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
    assert!(
        axis < ndim,
        "log_softmax_axis: axis {axis} out of bounds for ndim {ndim}"
    );
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

/// Gated Linear Unit (GLU): splits `input` in half along `dim`, returns
/// `first_half * sigmoid(second_half)`.
///
/// `input.shape()[dim]` must be even.
/// Equivalent to `torch.nn.functional.glu(input, dim)`.
#[inline]
pub fn glu<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    assert!(
        dim < ndim,
        "glu: dim {dim} out of bounds for {ndim}D tensor"
    );
    let dim_size = input.shape()[dim];
    assert!(
        dim_size.is_multiple_of(2),
        "glu: dim {dim} size {dim_size} must be even"
    );
    let half = dim_size / 2;
    let mut parts = super::super::shape::split(input, half, dim);
    assert_eq!(parts.len(), 2);
    let b_part = parts.pop().unwrap();
    let a_part = parts.pop().unwrap();
    let gate = elementwise_unary(&b_part, backend, UnaryOp::Sigmoid);
    super::super::binary::mul(&a_part, &gate, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn glu_splits_axis_and_gates_first_half() {
        let backend = SequentialBackend::new();
        let input = Tensor::<f64, SequentialBackend>::from_slice([2], &[2.0, 4.0]);
        let out = glu(&input, 0, &backend);
        let expected = 2.0 / (1.0 + (-4.0f64).exp());
        assert_eq!(out.shape(), &[1]);
        assert!(
            (out.as_slice()[0] - expected).abs() <= 1e-12,
            "glu output {} vs {expected}",
            out.as_slice()[0]
        );
    }
}
