// ── Vector and matrix norm reductions ──
//
// `norm` is the L2 vector norm over all elements: `sqrt(Σ x²)`, matching
// `torch.linalg.vector_norm(x, ord=2)` (flattened). It is a performance-
// critical short-circuit over the general `norm_p` — no `powf` allocation.
//
// `norm_p` / `norm_p_axis` support arbitrary finite `p > 0` via a host-side
// fold with `T::powf`, avoiding a new `BinaryOp::Pow` opcode (deferred per
// MS-62). Per-element `T::powf` runs at the native precision of `T`.
//
// `frobenius_norm` / `frobenius_norm_batched` compose on `mul + sum + sqrt`
// to match `torch.linalg.matrix_norm(A, ord='fro')` for 2-D and ≥3-D
// inputs respectively.

use crate::backend_ops::BackendOps;
use crate::binary;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// Euclidean (L2) norm over all elements: `sqrt(sum(x²))`.
///
/// Special case of [`norm_p`] with `p = 2`. Retained as the performance-
/// critical short-circuit when only L2 is needed — the ord-p variant
/// uses a host-side fold with `T::powf` and a final per-element `^(1/p)`.
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
        p > T::zero() && <T as Float>::is_finite(p),
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
        acc += <T as Float>::powf(<T as Float>::abs(v), p);
    }
    <T as Float>::powf(acc, T::one() / p)
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
/// Output shape is `input.shape` with `axis` reduced to size 1.
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
        p > T::zero() && <T as Float>::is_finite(p),
        "norm_p_axis: ord must be a finite positive number, got {p:?}"
    );

    let n = a.numel();
    let contiguous = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape(a.shape().to_vec())
    } else {
        a.to_contiguous_on(backend)
    };
    let mut host = vec![T::zero(); n];
    backend.copy_to_host(contiguous.storage(), &mut host);

    let out_shape: Vec<usize> = {
        let mut s = contiguous.shape().to_vec();
        s[axis] = 1;
        s
    };
    let out_numel: usize = out_shape.iter().product();

    let shape = contiguous.shape();
    let axis_dim = shape[axis];
    let pre_dims = &shape[..axis];
    let post_dims = &shape[axis + 1..];
    let pre_count: usize = pre_dims.iter().product();
    let post_count: usize = post_dims.iter().product();

    let mut out_host = vec![T::zero(); out_numel];
    let inv_p = T::one() / p;

    for pre_idx in 0..pre_count {
        for post_idx in 0..post_count {
            let base = pre_idx * (axis_dim * post_count) + post_idx;
            let mut acc = T::zero();
            for k in 0..axis_dim {
                let linear = base + k * post_count;
                acc += <T as Float>::powf(<T as Float>::abs(host[linear]), p);
            }
            let out_idx = pre_idx * post_count + post_idx;
            out_host[out_idx] = <T as Float>::powf(acc, inv_p);
        }
    }

    Tensor::from_slice_on(out_shape, &out_host, backend)
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
    let n = contiguous.numel();
    let mut host = vec![T::zero(); n];
    backend.copy_to_host(contiguous.storage(), &mut host);

    let last_two: usize = contiguous.shape()[ndim - 1] * contiguous.shape()[ndim - 2];
    let pre: usize = n / last_two;
    let mut out_host = Vec::with_capacity(pre);
    for batch_idx in 0..pre {
        let mut acc = T::zero();
        for j in 0..last_two {
            let v = host[batch_idx * last_two + j];
            acc += v * v;
        }
        out_host.push(<T as Float>::sqrt(acc));
    }

    let out_shape: Vec<usize> = contiguous.shape()[..ndim - 2].to_vec();
    Ok(Tensor::from_slice_on(out_shape, &out_host, backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    fn v3() -> Tensor<f64, SequentialBackend> {
        Tensor::from_slice(vec![5], &[1.0f64, -2.0, 3.0, -4.0, 5.0])
    }

    fn ref_p(x: &[f64], p: f64) -> f64 {
        let s: f64 = x.iter().map(|&v| v.abs().powf(p)).sum();
        s.powf(1.0 / p)
    }

    #[test]
    fn norm_p_p2_matches_classical_l2() {
        let b = SequentialBackend::new();
        let x = v3();
        let got = norm_p(&x, 2.0_f64, &b);
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
        let n = norm(&x, &b).expect("valid norm test input");
        let n_p = norm_p(&x, 2.0_f64, &b);
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
        let got = norm_p_axis(&x, 2.0, 0, &b);
        let n_global = norm_p(&x, 2.0, &b);
        assert_eq!(got.shape(), &[1]);
        assert_eq!(got.as_slice()[0].to_bits(), n_global.to_bits());
    }

    #[test]
    fn norm_p_axis_3d_axis1_matches_manual_per_slice() {
        let b = SequentialBackend::new();
        let x = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 3, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0,
            ],
        );
        let got = norm_p_axis(&x, 3.0, 1, &b);
        assert_eq!(got.shape(), &[2, 1, 2]);
        let want = [
            (1.0_f64 + 27.0 + 125.0).cbrt(),
            (8.0_f64 + 64.0 + 216.0).cbrt(),
            (1.0_f64 + 27.0 + 125.0).cbrt(),
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

    fn mat3x3(data: &[f64; 9]) -> Tensor<f64, SequentialBackend> {
        Tensor::<f64, SequentialBackend>::from_slice(vec![3, 3], data)
    }

    #[test]
    fn frobenius_norm_2d_matches_torch_oracle() {
        let b = SequentialBackend::new();
        let a = mat3x3(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let got = frobenius_norm(&a, &b).expect("valid Frobenius norm test input");
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
        let got = frobenius_norm(&id, &b).expect("valid Frobenius norm test input");
        let want = 3.0_f64.sqrt();
        assert!(
            (got - want).abs() < 1e-12,
            "frobenius_norm(I_3) = {got}, want {want}"
        );
    }

    #[test]
    fn frobenius_norm_batched_3d_returns_per_batch_scalars() {
        let b = SequentialBackend::new();
        let a = mat3x3(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stacked = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 3, 3],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0,
            ],
        );
        let got =
            frobenius_norm_batched(&stacked, &b).expect("valid batched Frobenius norm test input");
        assert_eq!(got.shape(), &[2]);
        let want = (204.0_f64).sqrt();
        for g in got.as_slice() {
            assert!((*g - want).abs() < 1e-12, "got {g}, want {want}");
        }
        let ref_scalar = frobenius_norm(&a, &b).expect("valid Frobenius norm test input");
        assert!(
            (ref_scalar - want).abs() < 1e-12,
            "2-D refr scalar = {ref_scalar}, want {want}"
        );
    }

    #[test]
    fn frobenius_norm_batched_4d_collapses_last_two_dims() {
        let b = SequentialBackend::new();
        let batch = Tensor::<f64, SequentialBackend>::from_slice(
            vec![2, 2, 3, 3],
            &[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        );
        let got =
            frobenius_norm_batched(&batch, &b).expect("valid batched Frobenius norm test input");
        assert_eq!(got.shape(), &[2, 2]);
        let want = 3.0_f64.sqrt();
        for g in got.as_slice() {
            assert!((*g - want).abs() < 1e-12, "got {g}, want {want}");
        }
    }

    #[test]
    fn frobenius_norm_batched_2d_returns_zero_dim_scalar_tensor() {
        let b = SequentialBackend::new();
        let a = mat3x3(&[3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0]);
        let got = frobenius_norm_batched(&a, &b).expect("valid batched Frobenius norm test input");
        assert_eq!(got.shape(), &[]);
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
