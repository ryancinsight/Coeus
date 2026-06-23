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

    let a_storage = a.storage();
    let b_storage = b.storage();
    let (out_storage, out_layout) = out.storage_mut_and_layout();

    let a_layout = batch_layout(a.layout(), a_slices, m, k, a_ndim == 2);
    let b_layout = batch_layout(b.layout(), b_slices, k, n, b_ndim == 2);
    let c_layout = Layout::from_shape_strides(
        Shape::from([batch_size, m, n].as_slice()),
        Strides::from([m * n, n, 1].as_slice()),
        out_layout.offset(),
    );

    backend.batched_matmul(
        a_storage,
        &a_layout,
        b_storage,
        &b_layout,
        out_storage,
        &c_layout,
    );

    out
}

fn batch_layout(
    layout: &Layout,
    batches: usize,
    rows: usize,
    cols: usize,
    is_rank2: bool,
) -> Layout {
    let batch_stride = if is_rank2 { 0 } else { rows * cols };
    Layout::from_shape_strides(
        Shape::from([batches, rows, cols].as_slice()),
        Strides::from([batch_stride, cols, 1].as_slice()),
        layout.offset(),
    )
}

/// Accumulating matrix multiplication: `out += a * b`.
#[inline]
pub fn matmul_accumulate<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    out: &mut Tensor<T, B>,
    backend: &B,
) {
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    assert!(
        a_ndim >= 2,
        "matmul_accumulate: A must be ≥ 2-D, got {a_ndim}-D"
    );
    assert!(
        b_ndim >= 2,
        "matmul_accumulate: B must be ≥ 2-D, got {b_ndim}-D"
    );

    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[a_ndim - 2];
    let k = a_shape[a_ndim - 1];
    let k2 = b_shape[b_ndim - 2];
    let n = b_shape[b_ndim - 1];
    assert_eq!(
        k, k2,
        "matmul_accumulate: inner dimension mismatch: {} vs {}",
        k, k2
    );

    if a_ndim == 2 && b_ndim == 2 {
        let (out_storage, out_layout) = out.storage_mut_and_layout();
        backend.matmul_accumulate(
            a.storage(),
            a.layout(),
            b.storage(),
            b.layout(),
            out_storage,
            out_layout,
        );
        return;
    }

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
        "matmul_accumulate: batch dimensions not broadcastable: {} vs {}",
        a_slices,
        b_slices
    );

    let a_layout = batch_layout(a.layout(), a_slices, m, k, a_ndim == 2);
    let b_layout = batch_layout(b.layout(), b_slices, k, n, b_ndim == 2);
    let (out_storage, out_layout) = out.storage_mut_and_layout();

    backend.batched_matmul_accumulate(
        a.storage(),
        &a_layout,
        b.storage(),
        &b_layout,
        out_storage,
        out_layout,
    );
}
