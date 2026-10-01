// ── Vector arithmetic (dot / cross) ──
//
// `dot` composes on the existing elementwise-multiply and sum reductions —
// the same `mul` + `sum` decomposition `norm` already uses (see
// `reduction/norms.rs`) — so the O(n) work stays on the selected provider for
// every backend that provides on-device `ElementwiseOps`/`ReductionOps`
// (CPU, CUDA, WGPU, ROCm, Metal); only the single resulting scalar crosses
// `BackendOps::copy_to_host`. No new `BinaryOp`/`ReductionOp` opcode.
//
// `cross` has no such decomposition available (the per-channel 3-vector
// product is not an elementwise-then-reduce operation). It dispatches
// through the `CrossOps` capability (ADR 0077): every backend computes it
// via the shared host-fold path (`cross_fold`), except `WgpuBackend`, which
// routes through hephaestus's on-device `CrossProductOps` seam for the
// layout that seam can express (contiguous, `dim` the last axis) and falls
// back to the same host-fold otherwise.

use crate::backend_ops::{BackendOps, CrossOps};
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Flat inner product: `Σ_i aᵢ bᵢ` after flattening.
///
/// Matches `torch.dot(input, tensor)`. Both arguments may be any shape with
/// the same number of elements; empty input returns `T::zero()`.
///
/// # Precision
/// Native `T` precision throughout (no widening accumulator; `Scalar`
/// already enforces native arithmetic).
///
/// # Provider residency
/// Decomposes as `sum(a * b)` over flattened views, reusing the existing
/// elementwise-multiply and sum-reduction seams (`crate::binary::mul`,
/// `super::sum`) exactly as [`super::norm`] already does. Every backend with
/// on-device `ElementwiseOps`/`ReductionOps` (CPU via Leto, CUDA, WGPU, ROCm,
/// Metal via their Hephaestus-backed seams) therefore runs the O(n) multiply
/// and reduction on the selected provider; only the one-element result
/// crosses `B::copy_to_host`, not both full operands.
#[inline]
pub fn dot<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
) -> Result<T, B::Error> {
    let n = a.numel();
    assert_eq!(n, b.numel(), "dot: numel mismatch: a={n}, b={}", b.numel());
    if n == 0 {
        return Ok(T::zero());
    }
    let backend = B::default();
    let flatten = |x: &Tensor<T, B>| -> Result<Tensor<T, B>, B::Error> {
        if x.is_contiguous() && x.layout().offset() == 0 {
            Ok(x.reshape([n]))
        } else {
            Ok(x.to_contiguous_on(&backend)?.reshape([n]))
        }
    };
    let a_flat = flatten(a)?;
    let b_flat = flatten(b)?;
    let product = crate::binary::mul(&a_flat, &b_flat, &backend)?;
    super::sum(&product, &backend)
}

/// Per-channel 3-vector cross product along `dim`.
///
/// Matches `torch.cross(input, other, dim)`: the slice axis must have exactly
/// three elements; the output keeps the same shape (no reduction). Output is
/// `(a_y b_z - a_z b_y, a_z b_x - a_x b_z, a_x b_y - a_y b_x)` per slice — the
/// right-handed `torch.cross` element ordering, matching the convention of
/// `numpy.cross` / `jax.numpy.cross` / `mlx.core.cross`.
///
/// # Precision
/// Single-pass fold in the native precision of `T`; no widening accumulator.
/// Equal-shape precondition is asserted (matches the PyTorch binding).
///
/// # Provider residency
/// Dispatches through [`CrossOps`] (ADR 0077): `WgpuBackend` runs on-device
/// for a contiguous tensor with `dim` the last axis; every other backend
/// (and `WgpuBackend` outside that layout) uses the shared host-fold.
#[inline]
pub fn cross<T: Scalar, B: CrossOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    dim: usize,
) -> Result<Tensor<T, B>, B::Error> {
    let shape = a.shape();
    assert_eq!(
        shape,
        b.shape(),
        "cross: shape mismatch: a={:?}, b={:?}",
        shape,
        b.shape()
    );
    assert!(
        dim < shape.len(),
        "cross: dim {dim} out of bounds for {}-D shape {:?}",
        shape.len(),
        shape
    );
    assert_eq!(
        shape[dim], 3,
        "cross: dim {dim} must have size 3 (got {})",
        shape[dim]
    );

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 3;

    let backend = B::default();
    let a_c = a.to_contiguous_on(&backend)?;
    let b_c = b.to_contiguous_on(&backend)?;

    let storage = backend.cross_storage(a_c.storage(), a_c.layout(), b_c.storage(), dim)?;

    Ok(Tensor::from_raw_parts(
        storage,
        coeus_core::Layout::new(out_shape.into()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor as CoTensor;

    type B = SequentialBackend;

    fn t_1d(data: &[f32]) -> CoTensor<f32, B> {
        CoTensor::<f32, B>::from_slice(vec![data.len()], data)
            .expect("invariant: test backend operation succeeds")
    }

    fn t_2d(rows: usize, cols: usize, data: &[f32]) -> CoTensor<f32, B> {
        CoTensor::<f32, B>::from_slice(vec![rows, cols], data)
            .expect("invariant: test backend operation succeeds")
    }

    fn t_3d(d0: usize, d1: usize, d2: usize, data: &[f32]) -> CoTensor<f32, B> {
        CoTensor::<f32, B>::from_slice(vec![d0, d1, d2], data)
            .expect("invariant: test backend operation succeeds")
    }

    // ── dot ────────────────────────────────────────────────────────────────

    #[test]
    fn dot_1d_matches_sum_pairwise_product() {
        let a = t_1d(&[1.0_f32, 2.0, 3.0]);
        let b = t_1d(&[4.0_f32, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let got = dot::<f32, B>(&a, &b).expect("invariant: test operation succeeds");
        assert_eq!(got, 32.0_f32);
    }

    #[test]
    fn dot_flattens_2d_inputs() {
        // torch.dot flattens: same result regardless of shape as long as
        // numel matches.
        let a = t_2d(2, 3, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t_2d(2, 3, &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
        // Σ aᵢ bᵢ over flattened view: 7 + 16 + 27 + 40 + 55 + 72 = 217
        let got = dot::<f32, B>(&a, &b).expect("invariant: test operation succeeds");
        assert_eq!(got, 217.0_f32);
    }

    #[test]
    fn dot_orthogonal_vectors_is_zero() {
        let a = t_1d(&[1.0_f32, 0.0, 0.0]);
        let b = t_1d(&[0.0_f32, 1.0, 0.0]);
        let got = dot::<f32, B>(&a, &b).expect("invariant: test operation succeeds");
        assert_eq!(got, 0.0_f32);
    }

    #[test]
    fn dot_empty_returns_zero() {
        let a: CoTensor<f32, B> =
            CoTensor::from_slice(vec![0], &[]).expect("invariant: test backend operation succeeds");
        let b: CoTensor<f32, B> =
            CoTensor::from_slice(vec![0], &[]).expect("invariant: test backend operation succeeds");
        let got = dot::<f32, B>(&a, &b).expect("invariant: test operation succeeds");
        assert_eq!(got, 0.0_f32);
    }

    #[test]
    #[should_panic(expected = "numel mismatch")]
    fn dot_numel_mismatch_panics() {
        let a = t_1d(&[1.0_f32, 2.0]);
        let b = t_1d(&[1.0_f32, 2.0, 3.0]);
        let _ = dot::<f32, B>(&a, &b).expect("invariant: test operation succeeds");
    }

    // ── cross ──────────────────────────────────────────────────────────────

    #[test]
    fn cross_3d_basis_vectors_along_last_axis() {
        // cross(e_x, e_y) = e_z  ⇒ [1,0,0] × [0,1,0] = [0,0,1]
        let a = t_1d(&[1.0_f32, 0.0, 0.0]);
        let b = t_1d(&[0.0_f32, 1.0, 0.0]);
        let out = cross::<f32, B>(&a, &b, 0).expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice(), &[0.0_f32, 0.0, 1.0]);
    }

    #[test]
    fn cross_3d_basis_y_x_is_minus_z() {
        // cross(e_y, e_x) = -e_z  ⇒ [0,1,0] × [1,0,0] = [0,0,-1]
        let a = t_1d(&[0.0_f32, 1.0, 0.0]);
        let b = t_1d(&[1.0_f32, 0.0, 0.0]);
        let out = cross::<f32, B>(&a, &b, 0).expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice(), &[0.0_f32, 0.0, -1.0]);
    }

    #[test]
    fn cross_anticommutative_flips_sign() {
        let a = t_1d(&[2.0_f32, 3.0, 4.0]);
        let b = t_1d(&[5.0_f32, 6.0, 7.0]);
        let ab = cross::<f32, B>(&a, &b, 0).expect("invariant: test operation succeeds");
        let ba = cross::<f32, B>(&b, &a, 0).expect("invariant: test operation succeeds");
        let ab_s = ab.as_slice();
        let ba_s = ba.as_slice();
        for i in 0..3 {
            assert_eq!(ab_s[i], -ba_s[i]);
        }
    }

    #[test]
    fn cross_2d_per_row_dim_last() {
        // shape [2, 3] with dim=1 (last axis): per-row cross product.
        //   row 0: [1,0,0] × [0,1,0] = [0,0,1]
        //   row 1: [0,1,0] × [0,0,1] = [1,0,0]
        let a = t_2d(2, 3, &[1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b = t_2d(2, 3, &[0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let out = cross::<f32, B>(&a, &b, 1).expect("invariant: test operation succeeds");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[0.0_f32, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn cross_2d_per_col_dim_first() {
        // shape [3, 3] with dim=0 (first axis is the 3-vector slice axis).
        // Each COLUMN of the tensor is a 3-vector; cross product per column.
        //   col 0: a[:,0]=[1,0,0] × b[:,0]=[0,0,5] = [0, -5, 0]
        //   col 1: a[:,1]=[0,2,0] × b[:,1]=[0,0,0] = [0, 0, 0]
        //   col 2: a[:,2]=[0,0,4] × b[:,2]=[5,0,0] = [0, 20, 0]
        //
        // Output is row-major [3, 3]: column j's three components land at
        // storage[0*3+j], storage[1*3+j], storage[2*3+j].
        let a = t_2d(3, 3, &[1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0]);
        let b = t_2d(3, 3, &[0.0_f32, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]);
        let out = cross::<f32, B>(&a, &b, 0).expect("invariant: test operation succeeds");
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.as_slice(),
            &[0.0_f32, 0.0, 0.0, -5.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn cross_3d_along_middle_axis() {
        // shape [2, 3, 1] dim=1: middle axis has 3 elements; outer axes are
        // the batch (pre=2, post=1). Each batch slot holds one 3-vector.
        //   slot 0: [2,0,0] × [0,3,0] → (0·0−0·3, 0·0−2·0, 2·3−0·0) = [0, 0, 6]
        //   slot 1: [0,0,4] × [5,0,0] → (0·0−4·0, 4·5−0·0, 0·0−0·5) = [0, 20, 0]
        let a_data = vec![2.0_f32, 0.0, 0.0, 0.0, 0.0, 4.0];
        let b_data = vec![0.0_f32, 3.0, 0.0, 5.0, 0.0, 0.0];
        let a = t_3d(2, 3, 1, &a_data);
        let b = t_3d(2, 3, 1, &b_data);
        let out = cross::<f32, B>(&a, &b, 1).expect("invariant: test operation succeeds");
        assert_eq!(out.shape(), &[2, 3, 1]);
        assert_eq!(out.as_slice(), &[0.0_f32, 0.0, 6.0, 0.0, 20.0, 0.0]);
    }

    #[test]
    fn cross_parallel_vectors_is_zero() {
        let a = t_1d(&[2.0_f32, 3.0, 4.0]);
        let out = cross::<f32, B>(&a, &a, 0).expect("invariant: test operation succeeds");
        assert_eq!(out.as_slice(), &[0.0_f32, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "size 3")]
    fn cross_wrong_axis_size_panics() {
        let a = t_1d(&[1.0_f32, 2.0, 3.0, 4.0]);
        let b = t_1d(&[5.0_f32, 6.0, 7.0, 8.0]);
        let _ = cross::<f32, B>(&a, &b, 0).expect("invariant: test operation succeeds");
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn cross_axis_out_of_bounds_panics() {
        let a = t_1d(&[1.0_f32, 2.0, 3.0]);
        let b = t_1d(&[4.0_f32, 5.0, 6.0]);
        let _ = cross::<f32, B>(&a, &b, 5).expect("invariant: test operation succeeds");
    }
}
