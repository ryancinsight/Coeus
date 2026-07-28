// ── Activation functions ──
// Element-wise neural network activation functions.

use super::kernel::{elementwise_unary, elementwise_unary_assign};
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;

/// Rectified Linear Unit: max(0, x).
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::relu;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[-1.0, 0.0, 1.0]).expect("construct tensor");
/// let b = relu(&a, &backend).expect("evaluate operation");
/// assert_eq!(b.as_slice(), &[0.0, 0.0, 1.0]);
/// ```
#[inline]
pub fn relu<T: Scalar, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Relu)
}
/// Sigmoid: 1 / (1 + exp(-x)).
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::sigmoid;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[0.0, 1.0, -1.0]).expect("construct tensor");
/// let b = sigmoid(&a, &backend).expect("evaluate operation");
/// let s = b.as_slice();
/// assert!((s[0] - 0.5).abs() < 1e-5);
/// assert!((s[1] - 0.73105858_f32).abs() < 1e-5);
/// assert!((s[2] - 0.26894142_f32).abs() < 1e-5);
/// ```
#[inline]
pub fn sigmoid<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Sigmoid)
}

/// Hyperbolic tangent.
#[inline]
pub fn tanh<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Tanh)
}

/// Exact GELU (Gaussian Error Linear Unit).
///
/// Formula: `0.5 * x * (1 + erf(x / sqrt(2)))`.
/// The tanh approximation is exposed separately as [`gelu_tanh`].
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::gelu;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2], &[0.0, 1.0]).expect("construct tensor");
/// let b = gelu(&a, &backend).expect("evaluate operation");
/// let s = b.as_slice();
/// assert!((s[0] - 0.0).abs() < 1e-5);
/// assert!((s[1] - 0.8413447_f32).abs() < 1e-5);
/// ```
#[inline]
pub fn gelu<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Gelu)
}

/// In-place Rectified Linear Unit: max(0, x).
#[inline]
pub fn relu_assign<T: Scalar, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Relu)
}

/// In-place Sigmoid: 1 / (1 + exp(-x)).
#[inline]
pub fn sigmoid_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Sigmoid)
}

/// In-place Hyperbolic tangent.
#[inline]
pub fn tanh_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Tanh)
}

/// In-place GELU.
#[inline]
pub fn gelu_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Gelu)
}

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x).
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::silu;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2], &[0.0, 1.0]).expect("construct tensor");
/// let b = silu(&a, &backend).expect("evaluate operation");
/// let s = b.as_slice();
/// assert!((s[0] - 0.0).abs() < 1e-5);
/// assert!((s[1] - 0.73105858_f32).abs() < 1e-5);
/// ```
#[inline]
pub fn silu<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Silu)
}

/// In-place SiLU.
#[inline]
pub fn silu_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Silu)
}

/// Mish (Self-Regularized Non-Monotonic Activation Function): x * tanh(softplus(x)).
#[inline]
pub fn mish<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Mish)
}

/// In-place Mish.
#[inline]
pub fn mish_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Mish)
}

/// ELU (Exponential Linear Unit): x >= 0 ? x : exp(x) - 1.
#[inline]
pub fn elu<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Elu)
}

/// In-place ELU.
#[inline]
pub fn elu_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Elu)
}

/// Softplus: log(1 + exp(x)).
#[inline]
pub fn softplus<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::Softplus)
}

/// In-place Softplus.
#[inline]
pub fn softplus_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::Softplus)
}

/// GELU tanh approximation.
#[inline]
pub fn gelu_tanh<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    elementwise_unary(input, backend, UnaryOp::GeluTanh)
}

/// In-place GELU tanh approximation.
#[inline]
pub fn gelu_tanh_assign<T: Float, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    elementwise_unary_assign(input, backend, UnaryOp::GeluTanh)
}

/// LeakyReLU: x >= 0 ? x : negative_slope * x.
#[inline]
pub fn leaky_relu<T: Float, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
    negative_slope: f64,
) -> Result<Tensor<T, B>, B::Error> {
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
) -> Result<(), B::Error> {
    elementwise_unary_assign(
        input,
        backend,
        UnaryOp::LeakyRelu(f64::to_bits(negative_slope)),
    )
}

mod gated;
mod softmax;

pub use gated::glu;
pub use softmax::{causal_softmax, log_softmax_axis, masked_softmax};

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-12,
                "{label}[{index}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn masked_softmax_excludes_masked_lanes() {
        let backend = SequentialBackend::new();
        let input =
            Tensor::<f64, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
        let mask =
            Tensor::<f64, SequentialBackend>::from_slice([2, 3], &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]).expect("construct tensor");
        let out =
            masked_softmax(&input, &mask, 1, &backend).expect("valid masked softmax test input");
        let exp_1 = 1.0f64.exp();
        let exp_3 = 3.0f64.exp();
        let exp_5 = 5.0f64.exp();
        let exp_6 = 6.0f64.exp();
        let expected = [
            exp_1 / (exp_1 + exp_3),
            0.0,
            exp_3 / (exp_1 + exp_3),
            0.0,
            exp_5 / (exp_5 + exp_6),
            exp_6 / (exp_5 + exp_6),
        ];
        assert_eq!(out.shape(), &[2, 3]);
        assert_close(out.as_slice(), &expected, "masked_softmax");
    }

    #[test]
    fn masked_softmax_all_masked_row_is_zero() {
        let backend = SequentialBackend::new();
        let input = Tensor::<f64, SequentialBackend>::from_slice([1, 3], &[1.0, 2.0, 3.0]).expect("construct tensor");
        let mask = Tensor::<f64, SequentialBackend>::zeros([1, 3]).expect("construct tensor");
        let out =
            masked_softmax(&input, &mask, 1, &backend).expect("valid masked softmax test input");
        assert_eq!(out.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn causal_softmax_masks_future_columns() {
        let backend = SequentialBackend::new();
        let input = Tensor::<f64, SequentialBackend>::from_slice(
            [1, 3, 3],
            &[1.0, 9.0, 9.0, 1.0, 2.0, 9.0, 1.0, 2.0, 3.0],
        ).expect("construct tensor");
        let out = causal_softmax(&input, 2, &backend).expect("valid causal softmax test input");
        let exp_1 = 1.0f64.exp();
        let exp_2 = 2.0f64.exp();
        let exp_3 = 3.0f64.exp();
        let expected = [
            1.0,
            0.0,
            0.0,
            exp_1 / (exp_1 + exp_2),
            exp_2 / (exp_1 + exp_2),
            0.0,
            exp_1 / (exp_1 + exp_2 + exp_3),
            exp_2 / (exp_1 + exp_2 + exp_3),
            exp_3 / (exp_1 + exp_2 + exp_3),
        ];
        assert_close(out.as_slice(), &expected, "causal_softmax");
    }

    #[test]
    fn glu_splits_axis_and_gates_first_half() {
        let backend = SequentialBackend::new();
        let input = Tensor::<f64, SequentialBackend>::from_slice([2], &[2.0, 4.0]).expect("construct tensor");
        let out = glu(&input, 0, &backend).expect("valid GLU test input");
        let expected = 2.0 / (1.0 + (-4.0f64).exp());
        assert_eq!(out.shape(), &[1]);
        assert!(
            (out.as_slice()[0] - expected).abs() <= 1e-12,
            "glu output {} vs {expected}",
            out.as_slice()[0]
        );
    }
}
