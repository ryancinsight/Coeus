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

/// Masked Softmax: excludes masked positions while computing softmax.
///
/// `mask` is a float tensor with the same shape as `input` where `0.0` means
/// "mask this position" and `1.0` means "keep this position".
///
/// Equivalent to `torch.softmax(input.masked_fill(mask == 0, -inf), dim)` for
/// rows with at least one unmasked element. Fully masked rows return zeros.
///
/// # Panics
/// Panics if `dim` is out of range or shapes differ.
#[inline]
pub fn masked_softmax<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    mask: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input.shape(),
        mask.shape(),
        "masked_softmax: input and mask must have the same shape"
    );
    let ndim = input.ndim();
    assert!(
        dim < ndim,
        "masked_softmax: dim {dim} out of bounds for {ndim}D input"
    );
    let shape = input.shape();
    let axis = shape[dim];
    let pre_count: usize = shape[..dim].iter().product();
    let post_count: usize = shape[dim + 1..].iter().product();

    let input_contiguous = input.to_contiguous_on(backend);
    let mask_contiguous = mask.to_contiguous_on(backend);
    let input_values = input_contiguous.as_slice();
    let mask_values = mask_contiguous.as_slice();
    let mut output = vec![T::zero(); input.numel()];

    for pre in 0..pre_count {
        for post in 0..post_count {
            let base = pre * axis * post_count + post;
            let mut row_max: Option<T> = None;
            for lane in 0..axis {
                let idx = base + lane * post_count;
                if mask_values[idx] != T::zero() {
                    row_max = Some(match row_max {
                        Some(current) if current > input_values[idx] => current,
                        _ => input_values[idx],
                    });
                }
            }

            let Some(row_max) = row_max else {
                continue;
            };

            let mut row_sum = T::zero();
            for lane in 0..axis {
                let idx = base + lane * post_count;
                if mask_values[idx] != T::zero() {
                    let value = (input_values[idx] - row_max).exp();
                    output[idx] = value;
                    row_sum = row_sum + value;
                }
            }

            if row_sum != T::zero() {
                for lane in 0..axis {
                    let idx = base + lane * post_count;
                    output[idx] = output[idx] / row_sum;
                }
            }
        }
    }

    Tensor::from_slice_on(shape.to_vec(), &output, backend)
}

/// Causal (lower-triangular) Softmax along `dim`.
///
/// Positions where `j > i` (future tokens) are masked before softmax.
/// Intended for attention-weight matrices `[..., seq_q, seq_k]` where
/// `dim == ndim - 1`.
///
/// # Panics
/// Panics if `dim` is out of range or is not the final axis.
#[inline]
pub fn causal_softmax<T: Float, B: BackendOps<T> + Default>(
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
        "causal_softmax: dim {dim} out of bounds for {ndim}D input"
    );
    assert!(
        dim > 0 && dim + 1 == ndim,
        "causal_softmax: dim must be the final axis of a rank >= 2 tensor"
    );
    let shape = input.shape();
    let seq_q = shape[dim - 1];
    let seq_k = shape[dim];

    // Build lower-triangular mask (1.0 = keep, 0.0 = masked).
    let numel = input.numel();
    let outer: usize = shape[..dim - 1].iter().product::<usize>().max(1);
    let mut mask_data = vec![T::zero(); numel];
    for batch in 0..outer {
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j <= i {
                    let flat = batch * seq_q * seq_k + i * seq_k + j;
                    if flat < numel {
                        mask_data[flat] = T::one();
                    }
                }
            }
        }
    }
    let mask = Tensor::from_slice_on(shape.to_vec(), &mask_data, backend);
    masked_softmax(input, &mask, dim, backend)
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
            Tensor::<f64, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mask =
            Tensor::<f64, SequentialBackend>::from_slice([2, 3], &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let out = masked_softmax(&input, &mask, 1, &backend);
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
        let input = Tensor::<f64, SequentialBackend>::from_slice([1, 3], &[1.0, 2.0, 3.0]);
        let mask = Tensor::<f64, SequentialBackend>::zeros([1, 3]);
        let out = masked_softmax(&input, &mask, 1, &backend);
        assert_eq!(out.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn causal_softmax_masks_future_columns() {
        let backend = SequentialBackend::new();
        let input = Tensor::<f64, SequentialBackend>::from_slice(
            [1, 3, 3],
            &[1.0, 9.0, 9.0, 1.0, 2.0, 9.0, 1.0, 2.0, 3.0],
        );
        let out = causal_softmax(&input, 2, &backend);
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
