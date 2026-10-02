// ── Vector and matrix norm reductions ──
//
// `norm` is the L2 vector norm over all elements: `sqrt(Σ x²)`, matching
// `torch.linalg.vector_norm(x, ord=2)` (flattened). It is a performance-
// critical short-circuit over the general `norm_p` — no scalar-power kernel.
//
// `norm_p` / `norm_p_axis` support arbitrary finite `p > 0` through provider
// scalar-power dispatch. CPU execution uses Leto `PowfOp`; accelerators use
// the selected Hephaestus scalar-power kernel.
//
// `frobenius_norm` / `frobenius_norm_batched` compose on `mul + sum + sqrt`
// to match `torch.linalg.matrix_norm(A, ord='fro')` for 2-D and ≥3-D
// inputs respectively.

use crate::backend_ops::{BackendOps, ElementwiseOps, ReductionOps, ScalarPowerOps};
use crate::binary;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Euclidean (L2) norm over all elements: `sqrt(sum(x²))`.
///
/// Special case of [`norm_p`] with `p = 2`. Retained as the performance-
/// critical short-circuit when only L2 is needed; the general ord-p variant
/// dispatches scalar powers through the selected provider.
#[inline]
pub fn norm<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    let n = a.numel();
    if n == 0 {
        return Ok(T::from_usize(0));
    }
    let flattened = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([n])
    } else {
        a.to_contiguous_on(backend).reshape([n])
    };
    let sq = binary::mul(&flattened, &flattened, backend);
    Ok(<T as Float>::sqrt(super::sum(&sq, backend)?))
}
/// `L_p` norm over all elements: `(Σ|xᵢ|^p)^(1/p)` for finite `p > 0`.
///
/// Matches `torch.linalg.vector_norm(x, ord=p)` over a flattened view for
/// any `p` in `(0, ∞)`. The complete computation stays on the selected
/// provider; only the scalar-returning API copies its final one-element result
/// to the caller.
///
/// # Panics
/// Panics if `p <= 0`, `p` is not finite, or the input is empty.
#[inline]
pub fn norm_p_tensor<
    T: Float,
    B: ElementwiseOps<T> + ReductionOps<T> + ScalarPowerOps<T> + Default,
>(
    a: &Tensor<T, B>,
    p: T,
    backend: &B,
) -> Tensor<T, B> {
    let n = a.numel();
    assert!(n > 0, "norm_p: empty tensor has no norm");
    assert!(
        p > T::zero() && <T as Float>::is_finite(p),
        "norm_p: ord must be a finite positive number, got {p:?}"
    );
    let magnitudes = crate::abs(a, backend);
    let powered = crate::pow_scalar(&magnitudes, p, backend);
    let flattened = powered.reshape([n]);
    let summed = super::sum_axis(&flattened, 0, backend).expect("norm_p: provider sum");
    crate::pow_scalar(&summed, T::one() / p, backend)
}

/// `L_p` norm over all elements returned as a provider-resident `[1]` tensor.
///
/// The scalar [`norm_p`] API is a compatibility boundary that reads this one
/// output element. Tracked autograd uses this tensor form to retain the norm
/// on the selected provider for backward.
#[inline]
pub fn norm_p<T: Float, B: ElementwiseOps<T> + ReductionOps<T> + ScalarPowerOps<T> + Default>(
    a: &Tensor<T, B>,
    p: T,
    backend: &B,
) -> T {
    let result = norm_p_tensor(a, p, backend);
    let mut scalar = [T::zero()];
    backend.copy_to_host(result.storage(), &mut scalar);
    scalar[0]
}

/// Per-axis `L_p` norm: tensor reduced along `axis` to size 1, with each
/// slice evaluated as `(Σ|xᵢ|^p)^(1/p)` for finite `p > 0`.
///
/// Matches `torch.linalg.vector_norm(x, ord=p, dim=axis)` over a flattened
/// view of every `axis`-slice. The complete computation stays on the selected
/// provider and the reduced axis remains a size-one dimension.
///
/// Output shape is `input.shape` with `axis` reduced to size 1.
///
/// # Panics
/// Panics if `axis` is out of range, the axis has zero elements, `p <= 0`,
/// or `p` is not finite.
#[inline]
pub fn norm_p_axis<
    T: Float,
    B: ElementwiseOps<T> + ReductionOps<T> + ScalarPowerOps<T> + Default,
>(
    a: &Tensor<T, B>,
    p: T,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "norm_p_axis: axis {axis} out of bounds");
    let n_axis = a.shape()[axis];
    assert!(n_axis > 0, "norm_p_axis: axis {axis} has zero elements");
    assert!(
        p > T::zero() && <T as Float>::is_finite(p),
        "norm_p_axis: ord must be a finite positive number, got {p:?}"
    );

    let magnitudes = crate::abs(a, backend);
    let powered = crate::pow_scalar(&magnitudes, p, backend);
    let summed = super::sum_axis(&powered, axis, backend).expect("norm_p_axis: provider sum");
    crate::pow_scalar(&summed, T::one() / p, backend)
}

/// Frobenius (matrix L2) norm over a single 2-D tensor: `sqrt(Σ aᵢⱼ²)`.
///
/// Matches `torch.linalg.matrix_norm(A, ord='fro')` for a 2-D input matrix.
/// Delegates to [`norm`] — no new `BinaryOp` opcodes required.
///
/// For ≥3-D tensors see [`frobenius_norm_batched`], which reduces over the
/// last two dimensions per batch.
#[inline]
pub fn frobenius_norm<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<T, B::Error> {
    norm(a, backend)
}

/// Per-batch Frobenius (matrix L2) norm: reduces over the last two
/// dimensions for every batch slot in the input.
///
/// Matches `torch.linalg.matrix_norm(A, ord='fro')` for inputs of rank
/// `≥ 2`. Leading batch dimensions are kept in the output.
///
/// # Semantics
/// - `ndim == 2`: returns a 0-D scalar Tensor (mirrors the scalar return of
///   [`norm`]). The binding at `coeus_python::matrix_norm` materialises this
///   to a Python `float`.
/// - `ndim >= 3`: returns a Tensor with shape `a.shape[..ndim-2]` holding
///   one Frobenius norm per batch slot.
///
/// # Panics
/// Panics if the input has rank < 2.
#[inline]
pub fn frobenius_norm_batched<T: Float, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let ndim = a.ndim();
    assert!(
        ndim >= 2,
        "frobenius_norm_batched: tensor must have rank >= 2, got ndim={ndim}"
    );
    if ndim == 2 {
        let v = norm(a, backend)?;
        return Ok(Tensor::from_slice_on([], &[v], backend));
    }

    let contiguous = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape(a.shape().to_vec())
    } else {
        a.to_contiguous_on(backend)
    };
    let squared = binary::mul(&contiguous, &contiguous, backend);
    let reduced_last = super::sum_axis(&squared, ndim - 1, backend)?;
    let reduced_matrix = super::sum_axis(&reduced_last, ndim - 2, backend)?;
    let norms = crate::unary::sqrt(&reduced_matrix, backend);

    let out_shape: Vec<usize> = contiguous.shape()[..ndim - 2].to_vec();
    Ok(norms.reshape(out_shape))
}

#[cfg(test)]
#[path = "norms_tests.rs"]
mod tests;
