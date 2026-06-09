// ── Matrix multiplication kernel ──
//
// Backend-agnostic: uses Layout offset arithmetic for batched N-D support.
// No CPU-addressable storage requirements — works for GPU backends.

use crate::backend_ops::BackendOps;
use coeus_core::{Layout, Scalar, Shape, Strides};
use coeus_tensor::Tensor;

/// Generalized matrix multiplication supporting 2-D and batched N-D inputs.
///
/// # Shape rules
/// - `A`: at least 2-D; inner dim `k = A.shape[-1]`.
/// - `B`: at least 2-D; if exactly 2-D, broadcast over all batch dims of `A`.
/// - `A.shape[-1] == B.shape[-2]`.
/// - Batch dims broadcast: equal or one is 1.
///
/// # Implementation
/// Uses `BackendOps::matmul` with Layout offset arithmetic for each batch
/// slice — no CPU storage access required, so GPU backends work identically.
#[inline]
pub fn matmul<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B> {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    assert!(a_ndim >= 2, "matmul: A must be ≥ 2-D, got {a_ndim}-D");
    assert!(b_ndim >= 2, "matmul: B must be ≥ 2-D, got {b_ndim}-D");

    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[a_ndim - 2];
    let k = a_shape[a_ndim - 1];
    let k2 = b_shape[b_ndim - 2];
    let n = b_shape[b_ndim - 1];
    assert_eq!(k, k2, "matmul: inner dimension mismatch: {} vs {}", k, k2);

    // Fast path for strictly 2-D inputs — zero overhead.
    if a_ndim == 2 && b_ndim == 2 {
        let mut out = Tensor::zeros_on([m, n], backend);
        let (out_storage, out_layout) = out.storage_mut_and_layout();
        backend.matmul(
            a.storage(),
            a.layout(),
            b.storage(),
            b.layout(),
            out_storage,
            out_layout,
        );
        return out;
    }

    // ── Batch dimension resolution ──
    let a_slices: usize = if a_ndim > 2 {
        a_shape[..a_ndim - 2].iter().product()
    } else {
        1
    };
    let b_slices: usize = if b_ndim > 2 {
        b_shape[..b_ndim - 2].iter().product()
    } else {
        1
    };
    assert!(
        a_slices == b_slices || a_slices == 1 || b_slices == 1,
        "matmul: batch dimensions not broadcastable: {} vs {}",
        a_slices,
        b_slices
    );
    let batch_size = a_slices.max(b_slices);

    // ── Build output shape: [batch_size, m, n] ──
    let out_shape = [batch_size, m, n];
    let mut out = Tensor::zeros_on(out_shape.as_slice(), backend);

    let a_mk = m * k;
    let b_kn = k * n;
    let c_mn = m * n;

    // Row-major strides for 2-D sub-layouts
    let a_strides = Strides::from([k, 1].as_slice());
    let b_strides = Strides::from([n, 1].as_slice());
    let c_strides = Strides::from([n, 1].as_slice());
    let a_shape_2d = Shape::from([m, k].as_slice());
    let b_shape_2d = Shape::from([k, n].as_slice());
    let c_shape_2d = Shape::from([m, n].as_slice());

    let a_storage = a.storage();
    let b_storage = b.storage();
    let (out_storage, _) = out.storage_mut_and_layout();

    for bi in 0..batch_size {
        let a_off = (bi % a_slices) * a_mk;
        let b_off = (bi % b_slices) * b_kn;
        let c_off = bi * c_mn;

        let a_layout_2d = Layout::from_shape_strides(a_shape_2d.clone(), a_strides.clone(), a_off);
        let b_layout_2d = Layout::from_shape_strides(b_shape_2d.clone(), b_strides.clone(), b_off);
        let c_layout_2d = Layout::from_shape_strides(c_shape_2d.clone(), c_strides.clone(), c_off);

        backend.matmul(
            a_storage,
            &a_layout_2d,
            b_storage,
            &b_layout_2d,
            out_storage,
            &c_layout_2d,
        );
    }

    out
}
