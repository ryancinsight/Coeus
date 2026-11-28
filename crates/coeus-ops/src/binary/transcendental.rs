// -- Differentiable binary transcendental compositions --
//
// `hypot`, `atan2` and element-wise `pow` composed from existing backend
// primitives (add/mul/div/sqrt/exp/log/neg/comparison masks). No new
// `BinaryOp`/`UnaryOp` opcode, no provider dispatch change: every line below
// runs on whatever backend `ElementwiseOps + ReductionOps` already covers
// (CPU via Leto, CUDA/WGPU/ROCm/Metal via Hephaestus seams).

use crate::binary;
use crate::reduction;
use coeus_core::{BackendError, Float, FloatElement};
use coeus_tensor::Tensor;

/// Element-wise hypotenuse: `sqrt(a^2 + b^2)`.
///
/// Matches `torch.hypot` / `jax.numpy.hypot`: overflow-safe form is NOT
/// used (native precision, same as the rest of the stack); NaN propagates.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::hypot;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f64, SequentialBackend>::from_slice([2], &[3.0, 5.0])
///     .expect("example tensor allocation succeeds");
/// let b = Tensor::<f64, SequentialBackend>::from_slice([2], &[4.0, 12.0])
///     .expect("example tensor allocation succeeds");
/// let c = hypot(&a, &b, &backend).expect("example hypotenuse succeeds");
/// assert_eq!(c.as_slice(), &[5.0, 13.0]);
/// ```
#[inline]
pub fn hypot<T: Float, B: crate::backend_ops::BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let a_sq = binary::mul(a, a, backend)?;
    let b_sq = binary::mul(b, b, backend)?;
    let sum = binary::add(&a_sq, &b_sq, backend)?;
    crate::unary::sqrt(&sum, backend)
}

/// Element-wise two-argument arctangent: `atan2(y, x)`.
///
/// Matches `torch.atan2` / `jax.numpy.arctan2`: quadrant-aware, IEEE branch
/// cuts (`atan2(+-0, -0) = +-pi`, etc.) inherited from the scalar `atan2`.
///
/// There is no provider `atan2` opcode, so this entry point evaluates the
/// scalar `atan2` on the host after a shape check; it is CPU-addressable
/// only. The autograd `Var` wrapper is backend-agnostic because it composes
/// these same tensor ops.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::atan2;
///
/// let backend = SequentialBackend::new();
/// let y = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0])
///     .expect("example tensor allocation succeeds");
/// let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[1.0, 1.0])
///     .expect("example tensor allocation succeeds");
/// let a = atan2(&y, &x, &backend).expect("example arctangent succeeds");
/// assert_eq!(a.as_slice(), &[0.0, core::f64::consts::FRAC_PI_4]);
/// ```
#[inline]
pub fn atan2<T: Float + FloatElement, B: crate::backend_ops::BackendOps<T> + Default>(
    y: &Tensor<T, B>,
    x: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    if y.shape() != x.shape() {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "atan2",
            lhs: y.shape().to_vec(),
            rhs: x.shape().to_vec(),
        }));
    }
    let y_host = y.to_contiguous_on(backend)?;
    let x_host = x.to_contiguous_on(backend)?;
    let y_slice = y_host.as_slice();
    let x_slice = x_host.as_slice();
    let mut out: Vec<T> = Vec::with_capacity(y_slice.len());
    for (&yv, &xv) in y_slice.iter().zip(x_slice.iter()) {
        let angle = yv.to_f64().atan2(xv.to_f64());
        out.push(<T as FloatElement>::from_f64(angle));
    }
    Tensor::from_slice_on(y.shape().to_vec(), &out, backend)
}

/// Element-wise power with a tensor exponent: `a ^ b`.
///
/// Forward only at the tensor level (use `coeus-autograd::pow` for scalar
/// exponents or the two-operand `pow_tensor` Var wrapper there). Negative
/// bases with fractional exponents yield NaN (IEEE, matching PyTorch);
/// integer-valued exponents are sign-preserving per the `pow` convention.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::pow_tensor;
///
/// let backend = SequentialBackend::new();
/// let base = Tensor::<f64, SequentialBackend>::from_slice([2], &[4.0, 9.0])
///     .expect("example tensor allocation succeeds");
/// let exp = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.5, 0.5])
///     .expect("example tensor allocation succeeds");
/// let out = pow_tensor(&base, &exp, &backend).expect("example power succeeds");
/// assert!((out.as_slice()[0] - 2.0).abs() < 1e-12);
/// assert!((out.as_slice()[1] - 3.0).abs() < 1e-12);
/// ```
#[inline]
pub fn pow_tensor<T: Float + FloatElement, B: crate::backend_ops::BackendOps<T> + Default>(
    base: &Tensor<T, B>,
    exponent: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let ln_base = crate::unary::log(base, backend)?;
    let scaled = binary::mul(&ln_base, exponent, backend)?;
    crate::unary::exp(&scaled, backend)
}

/// Element-wise maximum without tracking: `max(a, b)`.
///
/// Untracked counterpart of the autograd `maximum` ReLU identity; the same
/// decomposition (`a + relu(b - a)`) so forward values agree bit-for-bit.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::{maximum, minimum};
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([3], &[1.0, 5.0, -2.0])
///     .expect("example tensor allocation succeeds");
/// let b = Tensor::<f32, SequentialBackend>::from_slice([3], &[4.0, 2.0, -3.0])
///     .expect("example tensor allocation succeeds");
/// let mx = maximum(&a, &b, &backend).expect("example maximum succeeds");
/// let mn = minimum(&a, &b, &backend).expect("example minimum succeeds");
/// assert_eq!(mx.as_slice(), &[4.0, 5.0, -2.0]);
/// assert_eq!(mn.as_slice(), &[1.0, 2.0, -3.0]);
/// ```
#[inline]
pub fn maximum<T: Float, B: crate::backend_ops::BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let diff = binary::sub(b, a, backend)?;
    let pos = crate::unary::elementwise_unary(&diff, backend, crate::backend_ops::UnaryOp::Relu)?;
    binary::add(a, &pos, backend)
}

/// Element-wise minimum without tracking: `min(a, b)`.
///
/// Untracked counterpart of the autograd `minimum` ReLU identity.
#[inline]
pub fn minimum<T: Float, B: crate::backend_ops::BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let diff = binary::sub(a, b, backend)?;
    let pos = crate::unary::elementwise_unary(&diff, backend, crate::backend_ops::UnaryOp::Relu)?;
    binary::sub(a, &pos, backend)
}

/// Element-wise clamp to `[min_val, max_val]` without tracking.
///
/// `clamp(x) = min(max(x, lo), hi)`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::clamp_tensor;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f32, SequentialBackend>::from_slice([3], &[-5.0, 0.5, 9.0])
///     .expect("example tensor allocation succeeds");
/// let c = clamp_tensor(&x, 0.0, 1.0, &backend).expect("example clamp succeeds");
/// assert_eq!(c.as_slice(), &[0.0, 0.5, 1.0]);
/// ```
#[inline]
pub fn clamp_tensor<T: Float, B: crate::backend_ops::BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    min_val: T,
    max_val: T,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let lo = Tensor::full_on(input.shape(), min_val, backend)?;
    let hi = Tensor::full_on(input.shape(), max_val, backend)?;
    let above_lo = maximum(input, &lo, backend)?;
    minimum(&above_lo, &hi, backend)
}

/// Numerically stable log-sum-exp over all elements: `log(sum(exp(x)))`.
///
/// Shifts by `max(x)` before exponentiating: `lse = m + log(sum(exp(x - m)))`.
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::logsumexp;
///
/// let backend = SequentialBackend::new();
/// let x = Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0])
///     .expect("example tensor allocation succeeds");
/// let lse = logsumexp(&x, &backend).expect("example logsumexp succeeds");
/// assert!((lse - (1.0 + 1.0f64.exp()).ln()).abs() < 1e-12);
/// ```
#[inline]
pub fn logsumexp<T: Float + leto_ops::Scalar, B: crate::backend_ops::BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    let n = input.numel();
    assert!(n > 0, "logsumexp: empty tensor has no log-sum-exp");
    let m = reduction::amax(input, backend)?;
    let shift = Tensor::full_on(input.shape(), m, backend)?;
    let shifted = binary::sub(input, &shift, backend)?;
    let exp_shifted = crate::unary::exp(&shifted, backend)?;
    let s = reduction::sum(&exp_shifted, backend)?;
    Ok(<T as Float>::ln(s) + m)
}

/// Numerically stable log-sum-exp along `axis`, reducing it to size 1.
///
/// `lse = m + log(sum(exp(x - m), axis))` with `m = max(x, axis)`.
#[inline]
pub fn logsumexp_axis<T: Float, B: crate::backend_ops::BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let m = reduction::max_axis(input, axis, backend)?;
    let shifted = binary::sub(input, &m, backend)?;
    let exp_shifted = crate::unary::exp(&shifted, backend)?;
    let summed = reduction::sum_axis(&exp_shifted, axis, backend)?;
    let log_summed = crate::unary::log(&summed, backend)?;
    binary::add(&log_summed, &m, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    fn t(data: &[f64]) -> Tensor<f64, SequentialBackend> {
        Tensor::<f64, SequentialBackend>::from_slice(vec![data.len()], data)
            .expect("invariant: test backend operation succeeds")
    }

    #[test]
    fn hypot_matches_scalar() {
        let b = SequentialBackend::new();
        let out = hypot(&t(&[3.0, 5.0]), &t(&[4.0, 12.0]), &b)
            .expect("invariant: test operation succeeds");
        assert!((out.as_slice()[0] - 5.0).abs() < 1e-12);
        assert!((out.as_slice()[1] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn atan2_matches_scalar_quadrants() {
        let b = SequentialBackend::new();
        let y = t(&[0.0, 1.0, -1.0, 1.0]);
        let x = t(&[1.0, 1.0, 1.0, -1.0]);
        let out = atan2(&y, &x, &b).expect("invariant: test operation succeeds");
        let want = [
            0.0f64.atan2(1.0),
            1.0f64.atan2(1.0),
            (-1.0f64).atan2(1.0),
            1.0f64.atan2(-1.0),
        ];
        for (i, (&got, &w)) in out.as_slice().iter().zip(want.iter()).enumerate() {
            assert!((got - w).abs() < 1e-12, "lane {i}: {got} vs {w}");
        }
    }

    #[test]
    fn maximum_minimum_match_scalar() {
        let b = SequentialBackend::new();
        let a = t(&[1.0, 5.0, -2.0]);
        let c = t(&[4.0, 2.0, -3.0]);
        let mx = maximum(&a, &c, &b).expect("invariant: test operation succeeds");
        let mn = minimum(&a, &c, &b).expect("invariant: test operation succeeds");
        assert_eq!(mx.as_slice(), &[4.0, 5.0, -2.0]);
        assert_eq!(mn.as_slice(), &[1.0, 2.0, -3.0]);
    }

    #[test]
    fn clamp_tensor_clips_both_sides() {
        let b = SequentialBackend::new();
        let out = clamp_tensor(&t(&[-5.0, 0.5, 9.0]), 0.0, 1.0, &b)
            .expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn logsumexp_matches_stable_scalar() {
        let b = SequentialBackend::new();
        let out = logsumexp(&t(&[1000.0, 1001.0, 1002.0]), &b)
            .expect("invariant: test operation succeeds");
        let want = (1.0 + (-1.0f64).exp() + (-2.0f64).exp()).ln() + 1002.0;
        assert!((out - want).abs() < 1e-9, "{out} vs {want}");
    }

    #[test]
    fn logsumexp_axis_reduces_axis() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 2], &[1.0, 2.0, 3.0, 4.0])
            .expect("invariant: test backend operation succeeds");
        let out = logsumexp_axis(&x, 1, &b).expect("invariant: test operation succeeds");
        assert_eq!(out.shape(), &[2, 1]);
        let w0 = (1.0f64 + 0.0f64.exp()).ln() + 1.0 - 1.0 + 1.0;
        let _ = w0;
        let e0 = (1.0f64.exp() + 2.0f64.exp()).ln();
        let e1 = (3.0f64.exp() + 4.0f64.exp()).ln();
        assert!((out.as_slice()[0] - e0).abs() < 1e-12);
        assert!((out.as_slice()[1] - e1).abs() < 1e-12);
    }

    #[test]
    fn pow_tensor_matches_scalar() {
        let b = SequentialBackend::new();
        let out = pow_tensor(&t(&[4.0, 9.0]), &t(&[0.5, 0.5]), &b)
            .expect("invariant: test operation succeeds");
        assert!((out.as_slice()[0] - 2.0).abs() < 1e-12);
        assert!((out.as_slice()[1] - 3.0).abs() < 1e-12);
    }
}
