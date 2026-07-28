//! Softmax-family activation kernels.

use super::super::kernel::elementwise_unary;
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_tensor::Tensor;

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
) -> Result<Tensor<T, B>, B::Error> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "log_softmax_axis",
            axis,
            rank: ndim,
        }));
    }
    // Shift by max for numerical stability: shifted = x - max(x, axis)
    let max_vals = crate::reduction::max_axis(input, axis, backend)?;
    let shifted = crate::binary::sub(input, &max_vals, backend)?;
    // exp(shifted)
    let exp_shifted = elementwise_unary(&shifted, backend, UnaryOp::Exp)?;
    // sum(exp(shifted), axis)
    let sum_exp = crate::reduction::sum_axis(&exp_shifted, axis, backend)?;
    // log(sum_exp)
    let log_sum_exp = elementwise_unary(&sum_exp, backend, UnaryOp::Log)?;
    // out = shifted - log_sum_exp  (broadcasts log_sum_exp along axis)
    Ok(crate::binary::sub(&shifted, &log_sum_exp, backend)?)
}

/// Masked Softmax: excludes masked positions while computing softmax.
///
/// `mask` is a float tensor with the same shape as `input` where `0.0` means
/// "mask this position" and `1.0` means "keep this position".
///
/// Equivalent to `torch.softmax(input.masked_fill(mask == 0, -inf), dim)` for
/// rows with at least one unmasked element. Fully masked rows return zeros.
///
/// # Errors
/// Returns a backend error when `dim` is out of range, shapes differ, or
/// materialization fails.
#[inline]
pub fn masked_softmax<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    mask: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if input.shape() != mask.shape() {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "masked_softmax",
            lhs: input.shape().to_vec(),
            rhs: mask.shape().to_vec(),
        }));
    }
    let ndim = input.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "masked_softmax",
            axis: dim,
            rank: ndim,
        }));
    }
    let shape = input.shape();
    let axis = shape[dim];
    let pre_count: usize = shape[..dim].iter().product();
    let post_count: usize = shape[dim + 1..].iter().product();

    let input_contiguous = input.to_contiguous_on(backend)?;
    let mask_contiguous = mask.to_contiguous_on(backend)?;
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
                    row_sum += value;
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
/// # Errors
/// Returns a backend error when `dim` is out of range or is not the final
/// axis, or when materialization fails.
#[inline]
pub fn causal_softmax<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    if dim >= ndim || dim == 0 || dim + 1 != ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "causal_softmax",
            axis: dim,
            rank: ndim,
        }));
    }
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
    let mask = Tensor::from_slice_on(shape.to_vec(), &mask_data, backend)?;
    masked_softmax(input, &mask, dim, backend)
}
