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
// minimal). The per-axis `norm_p_axis` mirrors the same per-slice host
// fold over the `axis`-indexed slice lattice, matching
// `torch.linalg.vector_norm(x, ord=p, dim=...)` up to torch's
// output shape convention (collapsed axis → size 1; keepdim is the
// caller's responsibility, identical to `var_axis` / `std_dev_axis`).
//
// `frobenius_norm` / `frobenius_norm_batched` compose on the same
// `BinaryOp::Mul` + `ReductionOp::Sum` + native `T::sqrt` chain:
// `norm(x) = sqrt(sum(x·x))` is the canonical 2-D matrix_norm
// (matches `torch.linalg.matrix_norm(A, ord='fro')`), and the per-batch
// variant reduces over the last two dimensions for any tensor of rank
// ≥ 2 — raw composition, no new `BinaryOp::Pow` opcode (the `Pow`
// deferral remains owned by MS-62).
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
    var_mean(a, unbiased, backend).0
}

/// Variance over all elements together with the scalar mean.
///
/// Returns `(variance, mean)`. The variance uses the same Bessel-corrected
/// denominator as [`var`] (`unbiased=true` -> divide by `N - 1`;
/// `unbiased=false` -> divide by `N`), matching PyTorch's
/// `torch.var_mean(input, correction=...)` for flattened input.
#[inline]
pub fn var_mean<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    unbiased: bool,
    backend: &B,
) -> (T, T) {
    let n = a.numel();
    assert!(n > 0, "var_mean: empty tensor has no variance");

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
    (v.sqrt(), mu)
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
/// two-pass host fold to avoid recomputing the mean.
///
/// Returns `(variance, mean)`. The mean is identical to [`mean_axis(a, axis, …)`]
/// and the variance uses the same Bessel-corrected denominator as [`var_axis`]
/// (`unbiased=true` → divide by `n − 1`, matching PyTorch's
/// `torch.var_mean(input, dim, correction=1)`; `unbiased=false` → divide by `n`,
/// matching `torch.var_mean(input, dim, correction=0)`).
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

    // Two-pass form: compute μ along the axis once, reuse it for both the
    // returned mean tensor and the deviation/variance path. This is bitwise-
    // identical to calling `var_axis` and `mean_axis` independently but skips the
    // second `mean_axis` call (an extra O(N) reduction over `axis`).
    let mu = mean_axis(a, axis, backend);
    let dev = binary::sub(a, &mu, backend);
    let sq = binary::mul(&dev, &dev, backend);
    let s = sum_axis(&sq, axis, backend);
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

/// Per-axis `L_p` norm: tensor reduced along `axis` to size 1, with each
/// slice evaluated as `(Σ|xᵢ|^p)^(1/p)` for finite `p > 0`.
///
/// Matches `torch.linalg.vector_norm(x, ord=p, dim=axis)` over a flattened
/// view of every `axis`-slice. Implemented as a host-side fold with
/// `T::powf` accumulation so the input can stay on any backend (the
/// storage only requires `copy_to_host`-read access via `BackendOps`),
/// and no new `BinaryOp::Pow` opcode is added.
///
/// Output shape is `input.shape` with `axis` reduced to size 1 — the same
/// `ReductionOp::Sum`/`Mean` reduce shape used by [`mean_axis`] /
/// [`sum_axis`] so callers compose a follow-up keepdim/squeeze in their
/// own code (matching the pattern of `var_axis` / `std_dev_axis`).
///
/// # Panics
/// Panics if `axis` is out of range, the axis has zero elements, `p <= 0`,
/// or `p` is not finite.
#[inline]
pub fn norm_p_axis<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    p: T,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "norm_p_axis: axis {axis} out of bounds");
    let n_axis = a.shape()[axis];
    assert!(n_axis > 0, "norm_p_axis: axis {axis} has zero elements");
    assert!(
        p > T::zero() && p.is_finite(),
        "norm_p_axis: ord must be a finite positive number, got {p:?}"
    );

    // Materialize contiguous storage so we can use the host fold with
    // contiguous index math rather than chasing a strided layout. The
    // contiguous-fast-path avoids an allocation for the common case
    // (matches `norm` / `var` / `sum` / `mean`).
    let n = a.numel();
    let contiguous = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape(a.shape().to_vec())
    } else {
        a.to_contiguous_on(backend)
    };
    let mut host = vec![T::zero(); n];
    backend.copy_to_host(contiguous.storage(), &mut host);

    // Compute the per-axis reduced output shape with `axis` collapsed to 1.
    let out_shape: Vec<usize> = {
        let mut s = contiguous.shape().to_vec();
        s[axis] = 1;
        s
    };
    let out_numel: usize = out_shape.iter().product();

    // Per-output-element linear host fold. Rearrangement: outer-product
    // layout — output linear index splits into (pre, axis=0, post) so
    // each pre-slice owns `post` consecutive output indices, and each
    // post-step walks the `axis` stride. This avoids permutation
    // altogether and matches row-major reader expectation.
    let shape = contiguous.shape();
    let axis_dim = shape[axis];
    let pre_dims = &shape[..axis];
    let post_dims = &shape[axis + 1..];
    let pre_count: usize = pre_dims.iter().product();
    let post_count: usize = post_dims.iter().product();

    let mut out_host = vec![T::zero(); out_numel];
    let inv_p = T::one() / p;

    // Index strides over the outer (pre) and inner (post) dimensions.
    // Output linear index = pre_idx * post_count + post_idx.
    for pre_idx in 0..pre_count {
        for post_idx in 0..post_count {
            // Linear base of the input slice for this (pre, post) pair:
            //   base = pre_idx * (axis_dim * post_count)
            //        + post_idx       (because each "axis-step" moves
            //                          post_count elements)
            let base = pre_idx * (axis_dim * post_count) + post_idx;
            let mut acc = T::zero();
            for k in 0..axis_dim {
                let linear = base + k * post_count;
                acc = acc + host[linear].abs().powf(p);
            }
            let out_idx = pre_idx * post_count + post_idx;
            out_host[out_idx] = acc.powf(inv_p);
        }
    }

    // The output tensor holds the same element type as the input; use
    // `from_slice_on` for the contiguous storage so the dtype and backend
    // are normalised in one place.
    Tensor::from_slice_on(out_shape, &out_host, backend)
}

/// Frobenius (matrix L2) norm over a single 2-D tensor: `sqrt(Σ aᵢⱼ²)`.
///
/// Matches `torch.linalg.matrix_norm(A, ord='fro')` for a 2-D input matrix.
/// Composition on the existing L2 vector norm ([`norm`]) requires no new
/// `BinaryOp` opcodes and no new backend dispatch — the input materialises
/// to contiguous (no copy on the fast path), the squared sum runs through
/// the canonical `mul`+`sum` chain, and the final `sqrt` uses the native
/// `T::sqrt` already exposed by [`crate::unary::sqrt`].
///
/// For ≥3-D tensors see [`frobenius_norm_batched`], which reduces over the
/// last two dimensions per batch.
#[inline]
pub fn frobenius_norm<T: Float, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    norm(a, backend)
}

/// Per-batch Frobenius (matrix L2) norm: reduces over the last two
/// dimensions for every batch slot in the input.
///
/// Matches `torch.linalg.matrix_norm(A, ord='fro')` for inputs of rank
/// `≥ 2`: the norm is computed over the dimensions specified by the
/// canonical last-2-dim pair (`dim = (-2, -1)`), and the leading batch
/// dimensions are kept in the output. Equivalent to applying
/// [`frobenius_norm`] to each `m × n` slice of the batched input.
///
/// # Semantics
///
/// - `ndim == 2`: returns a 0-D scalar Tensor holding the single
///   Frobenius norm (mirrors the scalar return of the flat [`norm`]). The
///   boundary adapter at `coeus_python::matrix_norm` materialises this to
///   a Python `float` for ergonomics.
/// - `ndim >= 3`: returns a Tensor with shape `a.shape[..ndim-2]`
///   holding one Frobenius norm per batch slot.
///
/// Composition, like the other reduction ops, runs in native precision of
/// `T`. The host-side fold is unavoidable here (Frobenius is a reduction
/// over a variable last-two-dim window, and `B::copy_to_host` is the
/// canonical device→host transfer already in use by `norm` / `var` /
/// `var_axis`), so the kernel is lock-free and allocation-light.
///
/// # Panics
/// Panics if the input has rank < 2, mirroring the `torch.linalg.matrix_norm`
/// precondition.
#[inline]
pub fn frobenius_norm_batched<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    let ndim = a.ndim();
    assert!(
        ndim >= 2,
        "frobenius_norm_batched: tensor must have rank >= 2, got ndim={ndim}"
    );
    if ndim == 2 {
        // 2-D → 0-D scalar tensor. Mirrors `torch.linalg.matrix_norm(A)`
        // returning a single-element Tensor, which Python materialises to a
        // plain float at the binding boundary.
        let v = norm(a, backend);
        return Tensor::from_slice_on([], &[v], backend);
    }

    // Materialise contiguous storage so we can iterate with linear
    // arithmetic on the last-two-dim window per batch slot. Same fast-path
    // convention as `norm` / `norm_p_axis` / `var`.
    let contiguous = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape(a.shape().to_vec())
    } else {
        a.to_contiguous_on(backend)
    };
    let n = contiguous.numel();
    let mut host = vec![T::zero(); n];
    backend.copy_to_host(contiguous.storage(), &mut host);

    // Each batch slot owns `last_two = m · n` consecutive elements; per
    // slot we sum squares and sqrt at the end.
    let last_two: usize = contiguous.shape()[ndim - 1] * contiguous.shape()[ndim - 2];
    let pre: usize = n / last_two;
    let mut out_host = Vec::with_capacity(pre);
    for batch_idx in 0..pre {
        let mut acc = T::zero();
        for j in 0..last_two {
            let v = host[batch_idx * last_two + j];
            acc = acc + v * v;
        }
        out_host.push(acc.sqrt());
    }

    // Output shape is the input shape with the last two axes dropped —
    // matches `torch.linalg.matrix_norm`'s default `dim=(-2, -1)`,
    // `keepdim=False`.
    let out_shape: Vec<usize> = contiguous.shape()[..ndim - 2].to_vec();
    Tensor::from_slice_on(out_shape, &out_host, backend)
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

    #[test]
    fn norm_p_axis_axis1_matches_row_references() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 3],
            &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        );
        let got = norm_p_axis(&x, 2.0, 1, &b);
        let want = [
            (1.0_f64 + 4.0 + 9.0).sqrt(),
            (16.0_f64 + 25.0 + 36.0).sqrt(),
        ];
        assert_eq!(got.shape(), &[2, 1]);
        assert!(
            got.as_slice()
                .iter()
                .zip(want)
                .all(|(&g, w)| (g - w).abs() < 1e-12),
            "norm_p_axis(axis=1) = {:?}, want {:?}",
            got.as_slice(),
            want
        );
    }

    #[test]
    fn norm_p_axis_axis0_matches_column_references() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 3],
            &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        );
        let got = norm_p_axis(&x, 1.0, 0, &b);
        let want = [5.0, 7.0, 9.0];
        assert_eq!(got.shape(), &[1, 3]);
        assert_eq!(got.as_slice(), &want);
    }

    #[test]
    fn norm_p_axis_rank1_reduces_to_scalar_tensor() {
        let b = SequentialBackend::new();
        let x = v3();
        // Rank-1 axis=0 must equal global `norm_p` — output is still a
        // size-1 tensor (keepdim convention) and bits agree.
        let got = norm_p_axis(&x, 2.0, 0, &b);
        let n_global = norm_p(&x, 2.0, &b);
        assert_eq!(got.shape(), &[1]);
        assert_eq!(got.as_slice()[0].to_bits(), n_global.to_bits());
    }

    #[test]
    fn norm_p_axis_3d_axis1_matches_manual_per_slice() {
        let b = SequentialBackend::new();
        // Shape [2, 3, 2]: 2 batches × 3 rows × 2 cols. Axis=1 keeps the
        // (batch, _, col) lattice, output shape [2, 1, 2]. Each row sum of
        // |x|^p^(1/p) is the closed form.
        let x = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 3, 2],
            &[
                1.0, 2.0, // batch 0, row 0
                3.0, 4.0, // batch 0, row 1
                5.0, 6.0, // batch 0, row 2
                -1.0, 2.0, // batch 1, row 0
                -3.0, 4.0, // batch 1, row 1
                -5.0, 6.0, // batch 1, row 2
            ],
        );
        let got = norm_p_axis(&x, 3.0, 1, &b);
        assert_eq!(got.shape(), &[2, 1, 2]);
        let want = [
            // batch 0, col 0: rows [1, 3, 5] => (1 + 27 + 125)^(1/3)
            (1.0_f64 + 27.0 + 125.0).cbrt(),
            // batch 0, col 1: rows [2, 4, 6] => (8 + 64 + 216)^(1/3)
            (8.0_f64 + 64.0 + 216.0).cbrt(),
            // batch 1, col 0: rows [1, 3, 5] (abs) => same as batch 0 col 0
            (1.0_f64 + 27.0 + 125.0).cbrt(),
            // batch 1, col 1: rows [2, 4, 6] (abs) => same as batch 0 col 1
            (8.0_f64 + 64.0 + 216.0).cbrt(),
        ];
        for (g, w) in got.as_slice().iter().zip(want) {
            assert!(
                (*g - w).abs() < 1e-9,
                "norm_p_axis(3D, axis=1) = {g}, want {w}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "axis 2 out of bounds")]
    fn norm_p_axis_out_of_range_axis_panics() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 3], &[1.0; 6]);
        let _ = norm_p_axis(&x, 2.0, 2, &b);
    }

    #[test]
    #[should_panic(expected = "axis 1 has zero elements")]
    fn norm_p_axis_zero_size_axis_panics() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 0, 3], &[]);
        let _ = norm_p_axis(&x, 2.0, 1, &b);
    }

    #[test]
    #[should_panic(expected = "ord must be a finite positive number")]
    fn norm_p_axis_non_positive_ord_panics() {
        let b = SequentialBackend::new();
        let x = v3();
        let _ = norm_p_axis(&x, 0.0, 0, &b);
    }

    // ── frobenius_norm / frobenius_norm_batched ─────────────────────────────
    //
    // Reference oracle: `torch.linalg.matrix_norm(A, ord='fro')` returns
    // `sqrt(sum(a_ij ** 2))` over the last two dimensions. For 2-D inputs
    // this collapses to a single scalar; for ≥3-D inputs the per-batch
    // Frobenius norms are returned with the leading batch dimensions
    // preserved.

    fn mat3x3(data: &[f64; 9]) -> Tensor<f64, SequentialBackend> {
        Tensor::<f64, SequentialBackend>::from_slice(vec![3, 3], data)
    }

    #[test]
    fn frobenius_norm_2d_matches_torch_oracle() {
        // `A = reshape(arange(9), (3, 3))` from the torch.linalg.matrix_norm docs:
        //   [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        //   sum of squares = 0+1+4+9+16+25+36+49+64 = 204
        //   frobenius norm = sqrt(204) ≈ 14.2829
        let b = SequentialBackend::new();
        let a = mat3x3(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let got = frobenius_norm(&a, &b);
        let want = (204.0_f64).sqrt();
        assert!(
            (got - want).abs() < 1e-12,
            "frobenius_norm(3x3) = {got}, want {want}"
        );
    }

    #[test]
    fn frobenius_norm_2d_identity_matrix_is_sqrt_3() {
        let b = SequentialBackend::new();
        let id = mat3x3(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let got = frobenius_norm(&id, &b);
        let want = 3.0_f64.sqrt();
        assert!(
            (got - want).abs() < 1e-12,
            "frobenius_norm(I_3) = {got}, want {want}"
        );
    }

    #[test]
    fn frobenius_norm_batched_3d_returns_per_batch_scalars() {
        // `B = A.expand(2, -1, -1)` from the torch docs: two stacked 3×3
        // copies of `A = arange(9).reshape(3, 3)`. Each batch slot has
        // Frobenius norm `sqrt(204)`; output shape is `[2]`.
        let b = SequentialBackend::new();
        let a = mat3x3(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stacked = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 3, 3],
            // batch 0 = a; batch 1 = a
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0,
            ],
        );
        let got = frobenius_norm_batched(&stacked, &b);
        assert_eq!(got.shape(), &[2]);
        let want = (204.0_f64).sqrt();
        for (g, _) in got.as_slice().iter().zip([want, want]) {
            assert!((*g - want).abs() < 1e-12, "got {g}, want {want}");
        }
        // Sanity: the 2-D reference matches the per-batch call.
        let ref_scalar = frobenius_norm(&a, &b);
        assert!(
            (ref_scalar - want).abs() < 1e-12,
            "2-D refr scalar = {ref_scalar}, want {want}"
        );
    }

    #[test]
    fn frobenius_norm_batched_4d_collapses_last_two_dims() {
        // Shape `[2, 2, 3, 3]`: leading 2×2 batch of 3×3 identity matrices.
        // Each batch slot has Frobenius norm `sqrt(3)`; output shape is
        // `[2, 2]`.
        let b = SequentialBackend::new();
        let batch = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 2, 3, 3],
            // each 3×3 slice is the identity
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );
        let got = frobenius_norm_batched(&batch, &b);
        assert_eq!(got.shape(), &[2, 2]);
        let want = 3.0_f64.sqrt();
        for (g, _) in got.as_slice().iter().zip([0.0; 4]) {
            assert!((*g - want).abs() < 1e-12, "got {g}, want {want}");
        }
    }

    #[test]
    fn frobenius_norm_batched_2d_returns_zero_dim_scalar_tensor() {
        // For a 2-D matrix, the batched dispatch must still return a
        // 0-D scalar tensor so the Python adapter can collapse one float.
        let b = SequentialBackend::new();
        let a = mat3x3(&[3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0]);
        let got = frobenius_norm_batched(&a, &b);
        assert_eq!(got.shape(), &[]);
        // sqrt(9 + 16 + 25) = sqrt(50)
        let want = 50.0_f64.sqrt();
        let v = got.as_slice()[0];
        assert!((v - want).abs() < 1e-12, "got {v}, want {want}");
    }

    #[test]
    #[should_panic(expected = "rank >= 2")]
    fn frobenius_norm_batched_1d_panics() {
        let b = SequentialBackend::new();
        let x = v3();
        let _ = frobenius_norm_batched(&x, &b);
    }
}
