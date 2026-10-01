use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;

use super::validation::{
    csr_index, validate_csr, validate_csr_components, validate_csr_parts, validate_read,
};

/// Sparse Matrix-Vector multiplication (SpMV): y = A x
///
/// Computes multiplication of a sparse CSR matrix `A` and a dense vector `x`.
/// Returns a dense vector `y`.
#[inline]
pub fn spmv<T: Scalar, B: Backend>(
    a: &CsrTensor<T, B>,
    x: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    validate_csr(a, "sparse matrix-vector multiplication")?;
    validate_read(
        x,
        "sparse matrix-vector multiplication",
        Some(&[a.shape()[1]]),
    )?;
    let rows = a.shape()[0];

    // alloc_on: every row r writes y[r] = sum via y_ptr.write — no zero-init needed.
    // SAFETY: `validate_csr` proves every CSR read is within initialized,
    // contiguous storage, and this kernel writes each output row exactly once.
    let mut y = unsafe { Tensor::<T, B>::alloc_on([rows], backend)? };

    let val_slice = a.values().as_slice();
    let col_slice = a.col_indices().as_slice();
    let row_slice = a.row_offsets().as_slice();

    let val_ptr = Ptr(val_slice.as_ptr());
    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let y_ptr = MutPtr(y.as_mut_slice()?.as_mut_ptr());

    let x_slice = x.storage().as_slice();
    let x_ptr = Ptr(x_slice.as_ptr());
    let x_stride = x.layout().strides()[0];
    let x_offset = x.layout().offset();

    backend.parallel_for(0, rows, move |r| unsafe {
        let start = csr_index(row_ptr.read(r));
        let end = csr_index(row_ptr.read(r + 1));
        let mut sum = T::zero();
        for i in start..end {
            let col = csr_index(col_ptr.read(i));
            let val = val_ptr.read(i);
            let xv = x_ptr.read(x_offset + col * x_stride);
            sum += val * xv;
        }
        y_ptr.write(r, sum);
    });

    Ok(y)
}

/// Sparse-Dense Matrix multiplication (SpMM): C = A B
///
/// Multiplies a sparse CSR matrix `A` [M, K] by a dense matrix `B` [K, N].
/// Returns a dense matrix `C` [M, N].
#[inline]
pub fn spmm<T: Scalar, B: Backend>(
    a: &CsrTensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    validate_csr(a, "sparse-dense matrix multiplication")?;
    if b.ndim() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "sparse-dense matrix multiplication",
            rank: b.ndim(),
            max_rank: 2,
        }
        .into());
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    validate_read(b, "sparse-dense matrix multiplication", Some(&[k, n]))?;

    // alloc_on: parallel_for over rows writes every c[r,j] for j in 0..n — no zero-init needed.
    // SAFETY: `validate_csr` and `validate_read` prove all inputs are valid;
    // each row/column pair is written exactly once before success.
    let mut c = unsafe { Tensor::<T, B>::alloc_on([m, n], backend)? };

    let val_slice = a.values().as_slice();
    let col_slice = a.col_indices().as_slice();
    let row_slice = a.row_offsets().as_slice();

    let val_ptr = Ptr(val_slice.as_ptr());
    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let c_ptr = MutPtr(c.as_mut_slice()?.as_mut_ptr());

    let b_slice = b.storage().as_slice();
    let b_ptr = Ptr(b_slice.as_ptr());
    let b_stride_row = b.layout().strides()[0];
    let b_stride_col = b.layout().strides()[1];
    let b_offset = b.layout().offset();

    let c_stride_row = c.layout().strides()[0];
    let c_stride_col = c.layout().strides()[1];
    let c_offset = c.layout().offset();

    backend.parallel_for(0, m, move |r| unsafe {
        let start = csr_index(row_ptr.read(r));
        let end = csr_index(row_ptr.read(r + 1));
        let mut row_accumulator = smallvec::SmallVec::<[T; 256]>::from_elem(T::zero(), n);
        for i in start..end {
            let col = csr_index(col_ptr.read(i));
            let val = val_ptr.read(i);
            let b_col_offset = b_offset + col * b_stride_row;
            for j in 0..n {
                let bv = b_ptr.read(b_col_offset + j * b_stride_col);
                row_accumulator[j] += val * bv;
            }
        }
        for j in 0..n {
            c_ptr.write(
                c_offset + r * c_stride_row + j * c_stride_col,
                row_accumulator[j],
            );
        }
    });

    Ok(c)
}

/// Sparse-Dense Matrix multiplication values backward pass.
/// Computes the gradient with respect to sparse matrix A values.
#[inline]
pub fn spmm_backward_values<T: Scalar, B: Backend>(
    a_col_indices: &Tensor<i64, B>,
    a_row_offsets: &Tensor<i64, B>,
    a_shape: &[usize],
    b: &Tensor<T, B>,
    grad_out: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    if a_shape.len() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "sparse-dense values backward",
            rank: a_shape.len(),
            max_rank: 2,
        }
        .into());
    }
    if a_col_indices.ndim() != 1 || a_row_offsets.ndim() != 1 {
        return Err(coeus_core::BackendError::Storage {
            operation: "sparse-dense values backward",
            reason: "CSR indices and row offsets must be one-dimensional".to_owned(),
        }
        .into());
    }
    let nnz = a_col_indices.shape()[0];
    let rows_plus_one = a_shape[0]
        .checked_add(1)
        .ok_or(coeus_core::BackendError::Overflow {
            operation: "sparse-dense values backward",
            reason: "CSR row count",
        })?;
    if a_row_offsets.shape() != [rows_plus_one] {
        return Err(coeus_core::BackendError::ShapeMismatch {
            operation: "sparse-dense values backward",
            lhs: a_row_offsets.shape().to_vec(),
            rhs: vec![rows_plus_one],
        }
        .into());
    }
    validate_read(a_col_indices, "sparse-dense values backward", Some(&[nnz]))?;
    validate_read(
        a_row_offsets,
        "sparse-dense values backward",
        Some(&[rows_plus_one]),
    )?;
    if b.ndim() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "sparse-dense values backward",
            rank: b.ndim(),
            max_rank: 2,
        }
        .into());
    }
    validate_read(b, "sparse-dense values backward", None)?;
    if grad_out.ndim() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "sparse-dense values backward",
            rank: grad_out.ndim(),
            max_rank: 2,
        }
        .into());
    }
    validate_read(
        grad_out,
        "sparse-dense values backward",
        Some(&[a_shape[0], b.shape()[1]]),
    )?;
    validate_csr_parts(
        a_col_indices,
        a_row_offsets,
        a_shape[1],
        "sparse-dense values backward",
    )?;
    // alloc_on: every i in 0..nnz is written via grad_values_ptr.write(i, sum) — no zero-init needed.
    // SAFETY: the validated CSR row ranges partition `[0, nnz)`, so every
    // gradient slot is written exactly once before the kernel returns.
    let mut grad_values = unsafe { Tensor::<T, B>::alloc_on([nnz], backend)? };
    let m = a_shape[0];
    let n = b.shape()[1];

    let col_slice = a_col_indices.as_slice();
    let row_slice = a_row_offsets.as_slice();
    let b_slice = b.storage().as_slice();
    let grad_out_slice = grad_out.storage().as_slice();

    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let b_ptr = Ptr(b_slice.as_ptr());
    let grad_out_ptr = Ptr(grad_out_slice.as_ptr());
    let grad_values_ptr = MutPtr(grad_values.as_mut_slice()?.as_mut_ptr());

    let b_stride_row = b.layout().strides()[0];
    let b_stride_col = b.layout().strides()[1];
    let b_offset = b.layout().offset();

    let go_stride_row = grad_out.layout().strides()[0];
    let go_stride_col = grad_out.layout().strides()[1];
    let go_offset = grad_out.layout().offset();

    backend.parallel_for(0, m, move |r| {
        // SAFETY: The raw pointers `row_ptr`, `col_ptr`, `b_ptr`, `grad_out_ptr`, and `grad_values_ptr`
        // point to valid memory buffers allocated by the tensor library. The parallel execution index `r`
        // is guaranteed to be within [0, m), which is safe to read. All offset math is bounded by the shapes
        // of B, grad_out, and A.
        unsafe {
            let start = csr_index(row_ptr.read(r));
            let end = csr_index(row_ptr.read(r + 1));
            let go_row_offset = go_offset + r * go_stride_row;
            for i in start..end {
                let col = csr_index(col_ptr.read(i));
                let b_col_offset = b_offset + col * b_stride_row;
                let mut sum = T::zero();
                for j in 0..n {
                    let go_v = grad_out_ptr.read(go_row_offset + j * go_stride_col);
                    let b_v = b_ptr.read(b_col_offset + j * b_stride_col);
                    sum += go_v * b_v;
                }
                grad_values_ptr.write(i, sum);
            }
        }
    });

    Ok(grad_values)
}

/// Sparse-Dense Matrix multiplication dense matrix backward pass.
/// Computes the gradient with respect to dense matrix B.
#[inline]
pub fn spmm_backward_dense<T: Scalar, B: Backend>(
    a_values: &Tensor<T, B>,
    a_col_indices: &Tensor<i64, B>,
    a_row_offsets: &Tensor<i64, B>,
    a_shape: &[usize],
    grad_out: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    if a_shape.len() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "sparse-dense backward",
            rank: a_shape.len(),
            max_rank: 2,
        }
        .into());
    }
    let m = a_shape[0];
    let k = a_shape[1];
    if grad_out.ndim() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "sparse-dense backward",
            rank: grad_out.ndim(),
            max_rank: 2,
        }
        .into());
    }
    let n = grad_out.shape()[1];
    validate_csr_components(
        a_values,
        a_col_indices,
        a_row_offsets,
        a_shape,
        "sparse-dense backward",
    )?;
    validate_read(grad_out, "sparse-dense backward", Some(&[m, n]))?;
    // alloc_on: parallel_for over j writes every grad_b[col,j] for col in 0..k — no zero-init needed.
    // SAFETY: each output column `j` writes every `[col, j]` exactly once;
    // validated CSR metadata bounds all reads and accumulation indices.
    let mut grad_b = unsafe { Tensor::<T, B>::alloc_on([k, n], backend)? };

    let val_slice = a_values.as_slice();
    let col_slice = a_col_indices.as_slice();
    let row_slice = a_row_offsets.as_slice();
    let grad_out_slice = grad_out.storage().as_slice();

    let val_ptr = Ptr(val_slice.as_ptr());
    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let grad_out_ptr = Ptr(grad_out_slice.as_ptr());
    let grad_b_ptr = MutPtr(grad_b.as_mut_slice()?.as_mut_ptr());

    let gb_stride_row = grad_b.layout().strides()[0];
    let gb_stride_col = grad_b.layout().strides()[1];
    let gb_offset = grad_b.layout().offset();

    let go_stride_row = grad_out.layout().strides()[0];
    let go_stride_col = grad_out.layout().strides()[1];
    let go_offset = grad_out.layout().offset();

    backend.parallel_for(0, n, move |j| {
        // SAFETY: The raw pointers `row_ptr`, `col_ptr`, `val_ptr`, `grad_out_ptr`, and `grad_b_ptr`
        // point to valid memory buffers allocated by the tensor library. The parallel execution index `j`
        // is guaranteed to be within [0, n), which is safe to read/write. Since each worker thread processes
        // a unique column index `j`, there are no data race write conflicts on `grad_b`.
        unsafe {
            let mut col_accumulator = smallvec::SmallVec::<[T; 1024]>::from_elem(T::zero(), k);
            for r in 0..m {
                let start = csr_index(row_ptr.read(r));
                let end = csr_index(row_ptr.read(r + 1));
                let go_idx = go_offset + r * go_stride_row + j * go_stride_col;
                let go_v = grad_out_ptr.read(go_idx);
                if go_v == T::zero() {
                    continue;
                }
                for i in start..end {
                    let col = csr_index(col_ptr.read(i));
                    let val = val_ptr.read(i);
                    col_accumulator[col] += val * go_v;
                }
            }
            for col in 0..k {
                let gb_idx = gb_offset + col * gb_stride_row + j * gb_stride_col;
                grad_b_ptr.write(gb_idx, col_accumulator[col]);
            }
        }
    });

    Ok(grad_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::{BackendError, SequentialBackend, Shape};

    fn csr(
        values: &[f32],
        columns: &[i64],
        offsets: &[i64],
        shape: [usize; 2],
    ) -> CsrTensor<f32, SequentialBackend> {
        let backend = SequentialBackend::new();
        CsrTensor::new(
            Shape::from(shape.to_vec()),
            Tensor::from_slice_on(vec![values.len()], values, &backend)
                .expect("invariant: test values allocate"),
            Tensor::from_slice_on(vec![columns.len()], columns, &backend)
                .expect("invariant: test columns allocate"),
            Tensor::from_slice_on(vec![offsets.len()], offsets, &backend)
                .expect("invariant: test offsets allocate"),
        )
    }

    #[test]
    fn malformed_row_offsets_are_rejected_before_pointer_access() {
        let backend = SequentialBackend::new();
        let matrix = csr(&[2.0], &[0], &[0, 2], [1, 1]);
        let x = Tensor::from_slice_on(vec![1], &[3.0], &backend)
            .expect("invariant: test vector allocates");
        let result = spmv(&matrix, &x, &backend);
        assert!(matches!(result, Err(BackendError::Storage { .. })));
    }

    #[test]
    fn invalid_column_indices_are_rejected() {
        let backend = SequentialBackend::new();
        let x = Tensor::from_slice_on(vec![2], &[3.0, 4.0], &backend)
            .expect("invariant: test vector allocates");
        let negative = csr(&[2.0], &[-1], &[0, 1], [1, 2]);
        assert!(matches!(
            spmv(&negative, &x, &backend),
            Err(BackendError::InvalidNumericInput { .. })
        ));
        let out_of_range = csr(&[2.0], &[2], &[0, 1], [1, 2]);
        assert!(matches!(
            spmv(&out_of_range, &x, &backend),
            Err(BackendError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn empty_rows_and_valid_multiplication_keep_values() {
        let backend = SequentialBackend::new();
        let matrix = csr(&[2.0, 3.0], &[0, 2], &[0, 1, 1, 2], [3, 3]);
        let x = Tensor::from_slice_on(vec![3], &[5.0, 7.0, 11.0], &backend)
            .expect("invariant: test vector allocates");
        let y = spmv(&matrix, &x, &backend).expect("invariant: valid CSR multiplies");
        assert_eq!(y.as_slice(), &[10.0, 0.0, 33.0]);

        let dense = Tensor::from_slice_on(vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &backend)
            .expect("invariant: test matrix allocates");
        let product = spmm(&matrix, &dense, &backend).expect("invariant: valid SpMM");
        assert_eq!(product.as_slice(), &[2.0, 4.0, 0.0, 0.0, 15.0, 18.0]);
    }
}
