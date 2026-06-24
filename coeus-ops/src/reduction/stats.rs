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
// `torch.linalg.vector_norm(x, ord=2)` over a flattened view; pre-1.0 we do
// not advertise a non-`p=2` parameter — extending to ord-p norms would add
// `Pow` to `BinaryOp`, which we selectively defer to MS-66+.
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
