use coeus_core::{Backend, CpuAddressableStorageMut, Scalar};
use coeus_sparse::{CooTensor, CsrTensor};
use coeus_tensor::Tensor;

/// Convert a dense tensor to Sparse Coordinate List (COO) format.
pub fn dense_to_coo<T: Scalar, B: Backend>(dense: &Tensor<T, B>, backend: &B) -> CooTensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let temp_dense;
    let dense_ref = if dense.is_contiguous() && dense.layout().offset() == 0 {
        dense
    } else {
        temp_dense = dense.to_contiguous_on(backend);
        &temp_dense
    };
    let shape = dense_ref.shape();
    let rank = shape.len();
    let numel = dense_ref.numel();
    let slice = dense_ref.as_slice();

    let mut indices_vec = Vec::new();
    let mut values_vec = Vec::new();

    let mut index = smallvec::SmallVec::<[usize; 4]>::from_elem(0, rank);
    for i in 0..numel {
        let val = slice[i];
        if val != T::zero() {
            for &idx in &index {
                indices_vec.push(idx as i64);
            }
            values_vec.push(val);
        }
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }

    let nnz = values_vec.len();
    let mut indices = Tensor::<i64, B>::zeros_on([rank, nnz], backend);
    let mut values = Tensor::<T, B>::zeros_on([nnz], backend);

    let indices_slice = indices.as_mut_slice();
    for col in 0..nnz {
        for row in 0..rank {
            indices_slice[row * nnz + col] = indices_vec[col * rank + row];
        }
    }
    values.as_mut_slice().copy_from_slice(&values_vec);

    CooTensor::new(dense.shape_cloned(), indices, values)
}

/// Convert a Coordinate List (COO) tensor back to a dense tensor.
pub fn coo_to_dense<T: Scalar, B: Backend>(coo: &CooTensor<T, B>, backend: &B) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let shape = coo.shape().clone();
    let mut dense = Tensor::<T, B>::zeros_on(shape, backend);

    let nnz = coo.nnz();
    let rank = coo.shape().len();
    let temp_idx;
    let indices = if coo.indices().is_contiguous() && coo.indices().layout().offset() == 0 {
        coo.indices()
    } else {
        temp_idx = coo.indices().to_contiguous_on(backend);
        &temp_idx
    };
    let temp_val;
    let values = if coo.values().is_contiguous() && coo.values().layout().offset() == 0 {
        coo.values()
    } else {
        temp_val = coo.values().to_contiguous_on(backend);
        &temp_val
    };

    let idx_slice = indices.as_slice();
    let val_slice = values.as_slice();

    for col in 0..nnz {
        let mut logical_idx = smallvec::SmallVec::<[usize; 4]>::from_elem(0, rank);
        for row in 0..rank {
            logical_idx[row] = idx_slice[row * nnz + col] as usize;
        }
        let val = val_slice[col];
        dense.set(&logical_idx, dense.get(&logical_idx) + val);
    }

    dense
}

/// Convert a 2D Coordinate List (COO) tensor to Compressed Sparse Row (CSR) format.
pub fn coo_to_csr<T: Scalar, B: Backend>(coo: &CooTensor<T, B>, backend: &B) -> CsrTensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    assert_eq!(coo.shape().len(), 2, "COO to CSR requires 2D shape");
    let rows = coo.shape()[0];
    let nnz = coo.nnz();

    let temp_idx;
    let indices = if coo.indices().is_contiguous() && coo.indices().layout().offset() == 0 {
        coo.indices()
    } else {
        temp_idx = coo.indices().to_contiguous_on(backend);
        &temp_idx
    };
    let temp_val;
    let values = if coo.values().is_contiguous() && coo.values().layout().offset() == 0 {
        coo.values()
    } else {
        temp_val = coo.values().to_contiguous_on(backend);
        &temp_val
    };

    let idx_slice: &[i64] = indices.as_slice();
    let val_slice: &[T] = values.as_slice();

    let mut triples = Vec::with_capacity(nnz);
    for col in 0..nnz {
        let r = idx_slice[col] as usize;
        let c = idx_slice[nnz + col] as usize;
        let val = val_slice[col];
        triples.push((r, c, val));
    }

    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut csr_values = Tensor::<T, B>::zeros_on([nnz], backend);
    let mut csr_col_indices = Tensor::<i64, B>::zeros_on([nnz], backend);
    let mut csr_row_offsets = Tensor::<i64, B>::zeros_on([rows + 1], backend);

    let val_mut = csr_values.as_mut_slice();
    let col_mut = csr_col_indices.as_mut_slice();
    let row_mut = csr_row_offsets.as_mut_slice();

    let mut current_row = 0;
    row_mut[0] = 0;

    for (i, &(r, c, val)) in triples.iter().enumerate() {
        val_mut[i] = val;
        col_mut[i] = c as i64;
        while current_row < r {
            current_row += 1;
            row_mut[current_row] = i as i64;
        }
    }
    while current_row < rows {
        current_row += 1;
        row_mut[current_row] = nnz as i64;
    }

    CsrTensor::new(
        coo.shape().clone(),
        csr_values,
        csr_col_indices,
        csr_row_offsets,
    )
}

/// Convert a 2D dense tensor to Compressed Sparse Row (CSR) format.
pub fn dense_to_csr<T: Scalar, B: Backend>(dense: &Tensor<T, B>, backend: &B) -> CsrTensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    assert_eq!(dense.ndim(), 2, "Dense to CSR requires 2D tensor");
    let coo = dense_to_coo(dense, backend);
    coo_to_csr(&coo, backend)
}

/// Convert a Compressed Sparse Row (CSR) tensor back to a dense tensor.
pub fn csr_to_dense<T: Scalar, B: Backend>(csr: &CsrTensor<T, B>, backend: &B) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let rows = csr.shape()[0];
    let cols = csr.shape()[1];
    let mut dense = Tensor::<T, B>::zeros_on([rows, cols], backend);

    let temp_val;
    let val_cont = if csr.values().is_contiguous() && csr.values().layout().offset() == 0 {
        csr.values()
    } else {
        temp_val = csr.values().to_contiguous_on(backend);
        &temp_val
    };
    let temp_col;
    let col_cont = if csr.col_indices().is_contiguous() && csr.col_indices().layout().offset() == 0
    {
        csr.col_indices()
    } else {
        temp_col = csr.col_indices().to_contiguous_on(backend);
        &temp_col
    };
    let temp_row;
    let row_cont = if csr.row_offsets().is_contiguous() && csr.row_offsets().layout().offset() == 0
    {
        csr.row_offsets()
    } else {
        temp_row = csr.row_offsets().to_contiguous_on(backend);
        &temp_row
    };

    let val_slice = val_cont.as_slice();
    let col_slice = col_cont.as_slice();
    let row_slice = row_cont.as_slice();

    for r in 0..rows {
        let start = row_slice[r] as usize;
        let end = row_slice[r + 1] as usize;
        for i in start..end {
            let c = col_slice[i] as usize;
            dense.set(&[r, c], val_slice[i]);
        }
    }

    dense
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn test_sparse_conversions_and_ops() {
        let backend = SequentialBackend::new();
        // Dense matrix:
        // [ 1.0  0.0  0.0 ]
        // [ 0.0  0.0  2.0 ]
        // [ 3.0  0.0  0.0 ]
        let dense_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0];
        let dense =
            Tensor::<f32, SequentialBackend>::from_slice_on(vec![3, 3], &dense_data, &backend);

        // Convert to COO
        let coo = dense_to_coo(&dense, &backend);
        assert_eq!(coo.nnz(), 3);

        // Convert COO back to dense
        let dense_recon = coo_to_dense(&coo, &backend);
        assert_eq!(dense_recon.as_slice(), dense.as_slice());

        // Convert COO to CSR
        let csr = coo_to_csr(&coo, &backend);
        assert_eq!(csr.nnz(), 3);

        // Convert CSR back to dense
        let dense_recon_csr = csr_to_dense(&csr, &backend);
        assert_eq!(dense_recon_csr.as_slice(), dense.as_slice());
    }

    #[test]
    fn test_sparse_offset_views() {
        let backend = SequentialBackend::new();
        // Dense matrix:
        // [ 99.0  99.0  99.0 ] <- ignored row
        // [  1.0   0.0   0.0 ] <- starts here
        // [  0.0   0.0   2.0 ]
        // [  3.0   0.0   0.0 ]
        // [ 99.0  99.0  99.0 ] <- ignored row
        let dense_data = vec![
            99.0f32, 99.0, 99.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 99.0, 99.0, 99.0,
        ];
        let dense_all =
            Tensor::<f32, SequentialBackend>::from_slice_on(vec![5, 3], &dense_data, &backend);

        // Slice to [3, 3] starting at row 1 (offset 3)
        let dense = dense_all.slice(&[(1, 4), (0, 3)]);
        assert_eq!(dense.layout().offset(), 3);
        assert!(dense.is_contiguous());

        // Convert to COO (verifies dense_to_coo checks)
        let coo = dense_to_coo(&dense, &backend);
        assert_eq!(coo.nnz(), 3);

        // Convert COO back to dense (verifies coo_to_dense checks)
        let dense_recon = coo_to_dense(&coo, &backend);
        assert_eq!(dense_recon.as_slice(), dense.as_slice());

        // Convert COO to CSR (verifies coo_to_csr checks)
        let csr = coo_to_csr(&coo, &backend);
        assert_eq!(csr.nnz(), 3);

        // Convert CSR back to dense (verifies csr_to_dense checks)
        let dense_recon_csr = csr_to_dense(&csr, &backend);
        assert_eq!(dense_recon_csr.as_slice(), dense.as_slice());
    }
}
