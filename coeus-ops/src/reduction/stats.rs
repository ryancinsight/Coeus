// ── Statistical reductions ──
//
// Variance, standard deviation, and L2 norm composed from the existing
// `BackendOps` primitives (`sub`, `mul`, `reduce(Mean)`, `reduce(Sum)`,
// `sqrt`) per SSOT — no new backend dispatch is introduced.
//
// The two-pass `mean · E[(x − μ)²]` form keeps the algorithm numerically
// identical to PyTorch's `torch.var`/`torch.std` (Bessel-corrected when
// `unbiased = true`, matching `correction = 1`), and routes the
// squared-deviation square through a single backend multiply rather than a
// new `Pow` opcode — `BinaryOp` does not include `Pow` and adding it would
// mirror the indexing concerns described in `docs/backlog.md` MS-62.
//
// `norm` follows PyTorch's default `p = 2` (L2), matching
// `torch.linalg.vector_norm(x, ord=2)` over a flattened view. The
// `norm_p` variant supports arbitrary finite positive `p` values via a
// host-side fold with `T::powf`, avoiding any new `BinaryOp::Pow` opcode
// (the `Pow` decision is owned by `docs/backlog.md` MS-62 quantum and
// remains intentionally deferred to keep the backend dispatch surface
// minimal).
//
// The per-axis variants rely on the existing `BinaryOp` broadcast path
// (`coeus_leto::elementwise_binary_into` automatically broadcasts a
// `[d0, 1, …, 1]` mean against the full input).

use crate::backend_ops::BackendOps;
use crate::binary;
use crate::reduction::{mean, mean_axis, sum, sum_axis};
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Number of elements along `axis` — the denominator for `var_axis`
/// variants.
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
    let n = a.numel();
    assert!(n > 0, "var: empty tensor has no variance");

    let mu = mean(a, backend);
    let flattened = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([n])
    } else {
        a.to_contiguous_on(backend).reshape([n])
    };
    let mu_full = Tensor::full_on([n], mu, backend);
    let dev = binary::sub(&flattened, &mu_full, backend);
    let sq = binary::mul(&dev, &dev, backend);
    let s = sum(&sq, backend);
    let denom = if unbiased && n > 1 { n - 1 } else { n };
    s / T::from_usize(denom)
}

/// Variance along a specific axis, reducing it to size 1.
#[inline]
pub fn var_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    unbiased: bool,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "var_axis: axis {axis} out of bounds");
    let n = axis_count(a.shape(), axis);
    assert!(n > 0, "var_axis: axis {axis} has zero elements");

    let mu = mean_axis(a, axis, backend); // shape: axis-dim reduced to 1
    let dev = binary::sub(a, &mu, backend); // broadcasts mu along axis
    let sq = binary::mul(&dev, &dev, backend);
    let s = sum_axis(&sq, axis, backend);
    let denom = if unbiased && n > 1 { n - 1 } else { n };
    let denom_full = Tensor::full_on(s.shape_cloned(), T::from_usize(denom), backend);
    binary::div(&s, &denom_full, backend)
}

/// Standard deviation over all elements with optional Bessel correction.
#[inline]
pub fn std_dev<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    unbiased: bool,
    backend: &B,
) -> T {
    var(a, unbiased, backend).sqrt()
}

/// Standard deviation along a specific axis, reducing it to size 1.
#[inline]
pub fn std_dev_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    unbiased: bool,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "std_dev_axis: axis {axis} out of bounds");
    let n = axis_count(a.shape(), axis);
    assert!(n > 0, "std_dev_axis: axis {axis} has zero elements");

    let mu = mean_axis(a, axis, backend);
    let dev = binary::sub(a, &mu, backend);
    let sq = binary::mul(&dev, &dev, backend);
    let s = sum_axis(&sq, axis, backend);
    let denom = if unbiased && n > 1 { n - 1 } else { n };
    let denom_full = Tensor::full_on(s.shape_cloned(), T::from_usize(denom), backend);
    let v = binary::div(&s, &denom_full, backend);
    crate::unary::sqrt(&v, backend)
}

/// Euclidean (L2) norm over all elements: `sqrt(sum(x²))`.
///
/// Special case of [`norm_p`] with `p = 2`. Retained as the performance-
/// critical short-circuit when only L2 is needed — the ord-p variant
/// uses a host-side fold with `T::powf` and a final per-element `^(1/p)`.
#[inline]
pub fn norm<T: Float, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    let n = a.numel();
    if n == 0 {
        return T::from_usize(0);
    }
    let flattened = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([n])
    } else {
        a.to_contiguous_on(backend).reshape([n])
    };
    let sq = binary::mul(&flattened, &flattened, backend);
    sum(&sq, backend).sqrt()
}

/// `L_p` norm over all elements: `(Σ|xᵢ|^p)^(1/p)` for finite `p > 0`.
///
/// Matches `torch.linalg.vector_norm(x, ord=p)` over a flattened view for
/// any `p` in `(0, ∞)`. Implemented as a single host-side fold with the
/// native `T::powf` accumulation so the input can stay on any backend
/// (`B::DeviceBuffer<T>` only requires `copy_to_host`-read access via
/// the existing `BackendOps` surface) and no new `BinaryOp::Pow` opcode
/// is needed. Per-element `T::powf` runs at hardware-mapped precision of
/// `T`, matching the `Scalar`/native-precision execution rule.
///
/// # Panics
/// Panics if `p <= 0`, `p` is not finite, or the input is empty.
#[inline]
pub fn norm_p<T: Float, B: BackendOps<T> + Default>(a: &Tensor<T, B>, p: T, backend: &B) -> T {
    let n = a.numel();
    assert!(n > 0, "norm_p: empty tensor has no norm");
    assert!(
        p > T::zero() && p.is_finite(),
        "norm_p: ord must be a finite positive number, got {p:?}"
    );
    let flattened = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([n])
    } else {
        a.to_contiguous_on(backend).reshape([n])
    };
    let mut host = vec![T::zero(); n];
    backend.copy_to_host(flattened.storage(), &mut host);
    let mut acc = T::zero();
    for &v in &host {
        acc = acc + v.abs().powf(p);
    }
    acc.powf(T::one() / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    fn v3() -> Tensor<f64, SequentialBackend> {
        Tensor::from_slice(vec![5], &[1.0f64, -2.0, 3.0, -4.0, 5.0])
    }

    // Helper: integer-typed accumulators avoid 1/ε rounding for the L2 form.
    fn ref_p(x: &[f64], p: f64) -> f64 {
        let s: f64 = x.iter().map(|&v| v.abs().powf(p)).sum();
        s.powf(1.0 / p)
    }

    #[test]
    fn norm_p_p2_matches_classical_l2() {
        let b = SequentialBackend::new();
        let x = v3();
        let got = norm_p(&x, 2.0_f64, &b);
        // Sum of squares: 1+4+9+16+25 = 55; L2 = sqrt(55) ≈ 7.4162
        let want = (55.0_f64).sqrt();
        assert!(
            (got - want).abs() < 1e-12,
            "norm_p(p=2) = {got}, want {want}"
        );
    }

    #[test]
    fn norm_p_p1_matches_manhattan_distance() {
        let b = SequentialBackend::new();
        let x = v3();
        let got = norm_p(&x, 1.0_f64, &b);
        // Sum of |x| = 1+2+3+4+5 = 15
        let want = 15.0_f64;
        assert!(
            (got - want).abs() < 1e-12,
            "norm_p(p=1) = {got}, want {want}"
        );
    }

    #[test]
    fn norm_p_p3_matches_cubic_reference() {
        let b = SequentialBackend::new();
        let x = v3();
        let got = norm_p(&x, 3.0_f64, &b);
        let want = ref_p(&[1.0, -2.0, 3.0, -4.0, 5.0], 3.0);
        assert!(
            (got - want).abs() < 1e-10,
            "norm_p(p=3) = {got}, want {want}"
        );
    }

    #[test]
    fn norm_p_is_identical_to_norm_at_p2() {
        let b = SequentialBackend::new();
        let x = v3();
        let n = norm(&x, &b);
        let n_p = norm_p(&x, 2.0_f64, &b);
        // Reduce-order independence: both paths go through the same backend,
        // so the result bits are equal.
        assert_eq!(n.to_bits(), n_p.to_bits());
    }

    #[test]
    #[should_panic(expected = "empty tensor has no norm")]
    fn norm_p_empty_panics() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(vec![0], &[0.0f64; 0]);
        let _ = norm_p(&x, 2.0_f64, &b);
    }

    #[test]
    #[should_panic(expected = "ord must be a finite positive number")]
    fn norm_p_negative_ord_panics() {
        let b = SequentialBackend::new();
        let x = v3();
        let _ = norm_p(&x, -1.0_f64, &b);
    }

    #[test]
    #[should_panic(expected = "ord must be a finite positive number")]
    fn norm_p_zero_ord_panics() {
        let b = SequentialBackend::new();
        let x = v3();
        let _ = norm_p(&x, 0.0_f64, &b);
    }

    #[test]
    #[should_panic(expected = "ord must be a finite positive number")]
    fn norm_p_infinite_ord_panics() {
        let b = SequentialBackend::new();
        let x = v3();
        let _ = norm_p(&x, f64::INFINITY, &b);
    }
}
