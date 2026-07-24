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
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::matmul;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
/// let c = matmul(&a, &b, &backend).expect("valid matmul doctest inputs");
/// assert_eq!(c.shape(), &[2, 2]);
/// let expected = [58.0, 64.0, 139.0, 154.0];
/// for (got, want) in c.as_slice().iter().zip(expected.iter()) {
///     assert!((got - want).abs() < 1e-4);
/// }
/// ```
#[inline]
pub fn matmul<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
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
        let mut out = Tensor::alloc_on([m, n], backend);
        let (out_storage, out_layout) = out.storage_mut_and_layout();
        backend.matmul(
            a.storage(),
            a.layout(),
            b.storage(),
            b.layout(),
            out_storage,
            out_layout,
        )?;
        return Ok(out);
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
    let a_batch = &a_shape[..a_ndim.saturating_sub(2)];
    let b_batch = &b_shape[..b_ndim.saturating_sub(2)];
    let batch_shape = match (a_slices, b_slices) {
        (1, 1) if a_ndim == 2 && b_ndim > 2 => b_batch.to_vec(),
        (1, 1) if b_ndim == 2 && a_ndim > 2 => a_batch.to_vec(),
        (1, _) => b_batch.to_vec(),
        (_, 1) => a_batch.to_vec(),
        _ => {
            assert_eq!(
                a_batch, b_batch,
                "matmul: non-singleton batch dimensions must match exactly"
            );
            a_batch.to_vec()
        }
    };

    // Preserve logical batch axes while dispatching one flattened batch index
    // to the backend kernel.
    let mut out_shape = batch_shape;
    out_shape.extend([m, n]);
    let mut out = Tensor::alloc_on(out_shape, backend);

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
    )?;

    Ok(out)
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
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
/// use coeus_ops::matmul_accumulate;
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
/// let b = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[5.0, 6.0, 7.0, 8.0]);
/// let mut out = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[10.0, 20.0, 30.0, 40.0]);
/// matmul_accumulate(&a, &b, &mut out, &backend).expect("valid matmul doctest inputs");
/// // out = [[10+19, 20+22], [30+43, 40+50]] = [[29, 42], [73, 90]]
/// let expected = [29.0, 42.0, 73.0, 90.0];
/// for (got, want) in out.as_slice().iter().zip(expected.iter()) {
///     assert!((got - want).abs() < 1e-4);
/// }
/// ```
#[inline]
pub fn matmul_accumulate<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    out: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
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
        )?;
        return Ok(());
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
    assert_eq!(
        out.numel(),
        a_slices.max(b_slices) * m * n,
        "matmul_accumulate: output element count must match batched product"
    );
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    let c_layout = Layout::from_shape_strides(
        Shape::from([a_slices.max(b_slices), m, n].as_slice()),
        Strides::from([m * n, n, 1].as_slice()),
        out_layout.offset(),
    );

    backend.batched_matmul_accumulate(
        a.storage(),
        &a_layout,
        b.storage(),
        &b_layout,
        out_storage,
        &c_layout,
    )?;
    Ok(())
}
