// ── Variance and standard deviation reductions ──
//
// Composed from `BackendOps` primitives (`sub`, `mul`, `reduce(Mean)`,
// `reduce(Sum)`, native `T::sqrt`) per SSOT — no new backend dispatch.
//
// Two-pass `mean · E[(x − μ)²]` form matches PyTorch's
// `torch.var_mean`/`torch.std_mean` (Bessel-corrected when `unbiased=true`,
// `correction=1`). Squared deviations route through `BinaryOp::Mul` (not
// `BinaryOp::Pow`, which remains deferred per MS-62).
//
// Pair functions (`var_mean`, `std_mean`, `var_mean_axis`, `std_mean_axis`)
// are the SSOT: the singleton variants (`var`, `std_dev`, `var_axis`,
// `std_dev_axis`) delegate to them so the mean computation is never
// duplicated.

use crate::backend_ops::BackendOps;
use crate::binary;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Number of elements along `axis` — the denominator for `var_axis` variants.
#[inline]
fn axis_count(shape: &[usize], axis: usize) -> usize {
    shape[axis]
}

/// Variance over all elements with optional Bessel correction.
///
/// `unbiased = true` divides by `(N − 1)` (PyTorch/JAX default);
/// `unbiased = false` divides by `N` (population variance).
#[inline]
pub fn var<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    unbiased: bool,
    backend: &B,
) -> T {
    var_mean(a, unbiased, backend).0
}

/// Variance over all elements together with the scalar mean.
///
/// Returns `(variance, mean)`. The variance uses the same Bessel-corrected
/// denominator as [`var`] (`unbiased=true` → divide by `N - 1`;
/// `unbiased=false` → divide by `N`), matching PyTorch's
/// `torch.var_mean(input, correction=...)` for flattened input.
#[inline]
pub fn var_mean<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    unbiased: bool,
    backend: &B,
) -> (T, T) {
    let n = a.numel();
    assert!(n > 0, "var_mean: empty tensor has no variance");

    let mu = super::mean(a, backend);
    let flattened = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([n])
    } else {
        a.to_contiguous_on(backend).reshape([n])
    };
    let mu_full = Tensor::full_on([n], mu, backend);
    let dev = binary::sub(&flattened, &mu_full, backend);
    let sq = binary::mul(&dev, &dev, backend);
    let s = super::sum(&sq, backend);
    let denom = if unbiased && n > 1 { n - 1 } else { n };
    (s / T::from_usize(denom), mu)
}

/// Variance along a specific axis, reducing it to size 1.
#[inline]
pub fn var_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    unbiased: bool,
    backend: &B,
) -> Tensor<T, B> {
    var_mean_axis(a, axis, unbiased, backend).0
}

/// Standard deviation over all elements with optional Bessel correction.
#[inline]
pub fn std_dev<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    unbiased: bool,
    backend: &B,
) -> T {
    std_mean(a, unbiased, backend).0
}

/// Standard deviation over all elements together with the scalar mean.
///
/// Returns `(std_dev, mean)`. Composed on [`var_mean`] and native `T::sqrt`;
/// matches `torch.std_mean(input, correction=...)` for flattened input.
#[inline]
pub fn std_mean<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    unbiased: bool,
    backend: &B,
) -> (T, T) {
    let (v, mu) = var_mean(a, unbiased, backend);
    (<T as Float>::sqrt(v), mu)
}

/// Standard deviation along a specific axis, reducing it to size 1.
#[inline]
pub fn std_dev_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    unbiased: bool,
    backend: &B,
) -> Tensor<T, B> {
    std_mean_axis(a, axis, unbiased, backend).0
}

/// Variance along `axis` together with the per-slice mean, computed in a single
/// two-pass host fold.
///
/// Returns `(variance, mean)`. The mean is identical to
/// [`mean_axis(a, axis, …)`](super::mean_axis) and the variance uses the same
/// Bessel-corrected denominator as [`var_axis`] (`unbiased=true` → divide by
/// `n − 1`, matching `torch.var_mean(input, dim, correction=1)`;
/// `unbiased=false` → divide by `n`, matching `correction=0`).
#[inline]
pub fn var_mean_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    unbiased: bool,
    backend: &B,
) -> (Tensor<T, B>, Tensor<T, B>) {
    assert!(axis < a.ndim(), "var_mean_axis: axis {axis} out of bounds");
    let n = axis_count(a.shape(), axis);
    assert!(n > 0, "var_mean_axis: axis {axis} has zero elements");

    let mu = super::mean_axis(a, axis, backend);
    let dev = binary::sub(a, &mu, backend);
    let sq = binary::mul(&dev, &dev, backend);
    let s = super::sum_axis(&sq, axis, backend);
    let denom = if unbiased && n > 1 { n - 1 } else { n };
    let denom_full = Tensor::full_on(s.shape_cloned(), T::from_usize(denom), backend);
    let v = binary::div(&s, &denom_full, backend);
    (v, mu)
}

/// Standard deviation along `axis` together with the per-slice mean.
///
/// Returns `(std_dev, mean)`. Composed on [`var_mean_axis`] and native
/// `T::sqrt`; matches `torch.std_mean(input, dim, correction=...)`.
#[inline]
pub fn std_mean_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    unbiased: bool,
    backend: &B,
) -> (Tensor<T, B>, Tensor<T, B>) {
    let (v, mu) = var_mean_axis(a, axis, unbiased, backend);
    let std = crate::unary::sqrt(&v, backend);
    (std, mu)
}
