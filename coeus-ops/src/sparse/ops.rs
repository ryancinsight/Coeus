use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;

/// Sparse Matrix-Vector multiplication (SpMV): y = A x
///
/// Computes multiplication of a sparse CSR matrix `A` and a dense vector `x`.
/// Returns a dense vector `y`.
#[inline]
pub fn spmv<T: Scalar, B: Backend>(
    a: &CsrTensor<T, B>,
    x: &Tensor<T, B>,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    let rows = a.shape()[0];
    let cols = a.shape()[1];
    assert_eq!(x.ndim(), 1, "x must be 1D vector");
    assert_eq!(
        x.shape()[0],
        cols,
        "dimension mismatch: x shape must match CSR column count"
    );

    // alloc_on: every row r writes y[r] = sum via y_ptr.write — no zero-init needed.
    let mut y = Tensor::<T, B>::alloc_on([rows], backend);

    let val_slice = a.values().as_slice();
    let col_slice = a.col_indices().as_slice();
    let row_slice = a.row_offsets().as_slice();

    let val_ptr = Ptr(val_slice.as_ptr());
    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let y_ptr = MutPtr(y.as_mut_slice().as_mut_ptr());

    let x_slice = x.storage().as_slice();
    let x_ptr = Ptr(x_slice.as_ptr());
    let x_stride = x.layout().strides()[0];
    let x_offset = x.layout().offset();

    backend.parallel_for(0, rows, move |r| unsafe {
        let start = row_ptr.read(r) as usize;
        let end = row_ptr.read(r + 1) as usize;
        let mut sum = T::zero();
        for i in start..end {
            let col = col_ptr.read(i) as usize;
            let val = val_ptr.read(i);
            let xv = x_ptr.read(x_offset + col * x_stride);
            sum += val * xv;
        }
        y_ptr.write(r, sum);
    });

    y
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
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    assert_eq!(b.ndim(), 2, "b must be 2D matrix");
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    assert_eq!(
        k, k2,
        "dimension mismatch: CSR column count must match dense row count"
    );

    // alloc_on: parallel_for over rows writes every c[r,j] for j in 0..n — no zero-init needed.
    let mut c = Tensor::<T, B>::alloc_on([m, n], backend);

    let val_slice = a.values().as_slice();
    let col_slice = a.col_indices().as_slice();
    let row_slice = a.row_offsets().as_slice();

    let val_ptr = Ptr(val_slice.as_ptr());
    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let c_ptr = MutPtr(c.as_mut_slice().as_mut_ptr());

    let b_slice = b.storage().as_slice();
    let b_ptr = Ptr(b_slice.as_ptr());
    let b_stride_row = b.layout().strides()[0];
    let b_stride_col = b.layout().strides()[1];
    let b_offset = b.layout().offset();

    let c_stride_row = c.layout().strides()[0];
    let c_stride_col = c.layout().strides()[1];
    let c_offset = c.layout().offset();

    backend.parallel_for(0, m, move |r| unsafe {
        let start = row_ptr.read(r) as usize;
        let end = row_ptr.read(r + 1) as usize;
        let mut row_accumulator = smallvec::SmallVec::<[T; 256]>::from_elem(T::zero(), n);
        for i in start..end {
            let col = col_ptr.read(i) as usize;
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

    c
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
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    let nnz = a_col_indices.numel();
    // alloc_on: every i in 0..nnz is written via grad_values_ptr.write(i, sum) — no zero-init needed.
    let mut grad_values = Tensor::<T, B>::alloc_on([nnz], backend);
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
    let grad_values_ptr = MutPtr(grad_values.as_mut_slice().as_mut_ptr());

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
            let start = row_ptr.read(r) as usize;
            let end = row_ptr.read(r + 1) as usize;
            let go_row_offset = go_offset + r * go_stride_row;
            for i in start..end {
                let col = col_ptr.read(i) as usize;
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

    grad_values
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
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    let m = a_shape[0];
    let k = a_shape[1];
    let n = grad_out.shape()[1];
    // alloc_on: parallel_for over j writes every grad_b[col,j] for col in 0..k — no zero-init needed.
    let mut grad_b = Tensor::<T, B>::alloc_on([k, n], backend);

    let val_slice = a_values.as_slice();
    let col_slice = a_col_indices.as_slice();
    let row_slice = a_row_offsets.as_slice();
    let grad_out_slice = grad_out.storage().as_slice();

    let val_ptr = Ptr(val_slice.as_ptr());
    let col_ptr = Ptr(col_slice.as_ptr());
    let row_ptr = Ptr(row_slice.as_ptr());
    let grad_out_ptr = Ptr(grad_out_slice.as_ptr());
    let grad_b_ptr = MutPtr(grad_b.as_mut_slice().as_mut_ptr());

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
                let start = row_ptr.read(r) as usize;
                let end = row_ptr.read(r + 1) as usize;
                let go_idx = go_offset + r * go_stride_row + j * go_stride_col;
                let go_v = grad_out_ptr.read(go_idx);
                if go_v == T::zero() {
                    continue;
                }
                for i in start..end {
                    let col = col_ptr.read(i) as usize;
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

    grad_b
}
