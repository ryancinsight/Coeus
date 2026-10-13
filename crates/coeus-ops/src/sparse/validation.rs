use coeus_core::{Backend, BackendError, CpuAddressableStorage, Scalar, Storage};
use coeus_sparse::{CooTensor, CsrTensor};
use coeus_tensor::Tensor;

fn error<B: Backend>(error: BackendError) -> B::Error {
    error.into()
}

#[inline]
pub(crate) fn csr_index(index: i64) -> usize {
    usize::try_from(index)
        .expect("invariant: CSR validation checked nonnegative platform-sized indices")
}

fn checked_numel<B: Backend>(shape: &[usize], operation: &'static str) -> Result<usize, B::Error> {
    shape.iter().try_fold(1usize, |product, &extent| {
        product.checked_mul(extent).ok_or_else(|| {
            error::<B>(BackendError::Overflow {
                operation,
                reason: "tensor element count",
            })
        })
    })
}

fn is_contiguous<B: Backend>(
    shape: &[usize],
    strides: &[usize],
    operation: &'static str,
) -> Result<bool, B::Error> {
    if shape.len() != strides.len() {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "layout shape and stride ranks differ".to_owned(),
        }));
    }
    let mut expected_stride = 1usize;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        if extent > 1 && stride != expected_stride {
            return Ok(false);
        }
        expected_stride = expected_stride.checked_mul(extent).ok_or_else(|| {
            error::<B>(BackendError::Overflow {
                operation,
                reason: "contiguous stride calculation",
            })
        })?;
    }
    Ok(true)
}

fn validate_tensor<T: Scalar, B: Backend>(
    tensor: &Tensor<T, B>,
    operation: &'static str,
    expected_shape: Option<&[usize]>,
    initialized: bool,
    require_contiguous: bool,
) -> Result<(), B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    if let Some(expected) = expected_shape {
        if tensor.shape() != expected {
            return Err(error::<B>(BackendError::ShapeMismatch {
                operation,
                lhs: tensor.shape().to_vec(),
                rhs: expected.to_vec(),
            }));
        }
    }
    let layout = tensor.layout();
    let numel = checked_numel::<B>(layout.shape(), operation)?;
    if layout.shape().len() != layout.strides().len() {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "layout shape and stride ranks differ".to_owned(),
        }));
    }
    let storage_len = Storage::len(tensor.storage());
    if numel == 0 {
        if layout.offset() > storage_len {
            return Err(error::<B>(BackendError::Storage {
                operation,
                reason: "empty layout offset exceeds storage length".to_owned(),
            }));
        }
    } else {
        let mut last = layout.offset();
        for (&extent, &stride) in layout.shape().iter().zip(layout.strides()) {
            let term = (extent - 1).checked_mul(stride).ok_or_else(|| {
                error::<B>(BackendError::Overflow {
                    operation,
                    reason: "tensor layout span",
                })
            })?;
            last = last.checked_add(term).ok_or_else(|| {
                error::<B>(BackendError::Overflow {
                    operation,
                    reason: "tensor layout span",
                })
            })?;
        }
        if last >= storage_len {
            return Err(error::<B>(BackendError::Storage {
                operation,
                reason: format!("layout span ends at {last}, storage length is {storage_len}"),
            }));
        }
    }
    if initialized && Storage::try_as_slice(tensor.storage()).is_none() {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "input storage is not initialized".to_owned(),
        }));
    }
    let contiguous = is_contiguous::<B>(layout.shape(), layout.strides(), operation)?;
    if require_contiguous && !contiguous {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "sparse kernels require contiguous CSR component storage".to_owned(),
        }));
    }
    Ok(())
}

pub(crate) fn validate_csr<T: Scalar, B: Backend>(
    csr: &CsrTensor<T, B>,
    operation: &'static str,
) -> Result<(), B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    validate_csr_components(
        csr.values(),
        csr.col_indices(),
        csr.row_offsets(),
        csr.shape(),
        operation,
    )
    .map(|_| ())
}

pub(crate) fn validate_csr_components<T: Scalar, B: Backend>(
    values: &Tensor<T, B>,
    col_indices: &Tensor<i64, B>,
    row_offsets: &Tensor<i64, B>,
    shape: &[usize],
    operation: &'static str,
) -> Result<usize, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    if shape.len() != 2 {
        return Err(error::<B>(BackendError::UnsupportedRank {
            operation,
            rank: shape.len(),
            max_rank: 2,
        }));
    }
    for tensor in [values.shape(), col_indices.shape(), row_offsets.shape()] {
        if tensor.len() != 1 {
            return Err(error::<B>(BackendError::UnsupportedRank {
                operation,
                rank: tensor.len(),
                max_rank: 1,
            }));
        }
    }
    let rows = shape[0];
    let cols = shape[1];
    let nnz = values.shape()[0];
    validate_tensor(values, operation, Some(&[nnz]), true, true)?;
    validate_tensor(col_indices, operation, Some(&[nnz]), true, true)?;
    validate_tensor(
        row_offsets,
        operation,
        Some(&[rows.checked_add(1).ok_or_else(|| {
            error::<B>(BackendError::Overflow {
                operation,
                reason: "CSR row count",
            })
        })?]),
        true,
        true,
    )?;

    validate_csr_parts(col_indices, row_offsets, cols, operation)?;
    Ok(nnz)
}

pub(crate) fn validate_csr_parts<B: Backend>(
    col_indices: &Tensor<i64, B>,
    row_offsets: &Tensor<i64, B>,
    cols: usize,
    operation: &'static str,
) -> Result<(), B::Error>
where
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    if col_indices.ndim() != 1 || row_offsets.ndim() != 1 {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "CSR indices and row offsets must be one-dimensional".to_owned(),
        }));
    }
    let nnz = col_indices.shape()[0];
    let columns = col_indices.as_slice();
    for (position, &column) in columns.iter().enumerate() {
        if column < 0 {
            return Err(error::<B>(BackendError::InvalidNumericInput {
                operation,
                reason: format!("negative CSR column index {column} at position {position}"),
            }));
        }
        let column = usize::try_from(column).map_err(|_| {
            error::<B>(BackendError::IndexOutOfRange {
                operation,
                position,
                index: usize::MAX,
                bound: cols,
            })
        })?;
        if column >= cols {
            return Err(error::<B>(BackendError::IndexOutOfRange {
                operation,
                position,
                index: column,
                bound: cols,
            }));
        }
    }

    let offsets = row_offsets.as_slice();
    if offsets.first().copied() != Some(0) {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "CSR row offsets must start at zero".to_owned(),
        }));
    }
    let mut previous = 0usize;
    for (row, &offset) in offsets.iter().enumerate() {
        let offset = usize::try_from(offset).map_err(|_| {
            error::<B>(BackendError::InvalidNumericInput {
                operation,
                reason: format!("negative CSR row offset at position {row}"),
            })
        })?;
        if offset < previous || offset > nnz {
            return Err(error::<B>(BackendError::Storage {
                operation,
                reason: format!("CSR row offset {offset} at position {row} is outside 0..={nnz}"),
            }));
        }
        previous = offset;
    }
    if previous != nnz {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: format!("CSR final row offset {previous} does not equal nnz {nnz}"),
        }));
    }
    Ok(())
}

pub(crate) fn validate_read<T: Scalar, B: Backend>(
    tensor: &Tensor<T, B>,
    operation: &'static str,
    expected_shape: Option<&[usize]>,
) -> Result<(), B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    validate_tensor(tensor, operation, expected_shape, true, true)
}

pub(crate) fn validate_readable_layout<T: Scalar, B: Backend>(
    tensor: &Tensor<T, B>,
    operation: &'static str,
    expected_shape: Option<&[usize]>,
) -> Result<(), B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    validate_tensor(tensor, operation, expected_shape, true, false)
}

pub(crate) fn validate_coo_layout<T: Scalar, B: Backend>(
    coo: &CooTensor<T, B>,
    operation: &'static str,
) -> Result<usize, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    if coo.values().ndim() != 1 {
        return Err(error::<B>(BackendError::UnsupportedRank {
            operation,
            rank: coo.values().ndim(),
            max_rank: 1,
        }));
    }
    let nnz = coo.values().shape()[0];
    let indices_shape = [coo.shape().len(), nnz];
    validate_tensor(coo.values(), operation, Some(&[nnz]), true, false)?;
    validate_tensor(coo.indices(), operation, Some(&indices_shape), true, false)?;
    Ok(nnz)
}

pub(crate) fn validate_coo_indices<B: Backend>(
    indices: &[i64],
    shape: &[usize],
    nnz: usize,
    operation: &'static str,
) -> Result<(), B::Error> {
    let expected_len = shape.len().checked_mul(nnz).ok_or_else(|| {
        error::<B>(BackendError::Overflow {
            operation,
            reason: "COO index element count",
        })
    })?;
    if indices.len() != expected_len {
        return Err(error::<B>(BackendError::Storage {
            operation,
            reason: "COO index storage length does not match its shape".to_owned(),
        }));
    }
    if nnz == 0 {
        return Ok(());
    }
    for (position, &index) in indices.iter().enumerate() {
        let bound = shape[position / nnz];
        coo_index::<B>(index, position, bound, operation)?;
    }
    Ok(())
}

pub(crate) fn coo_index<B: Backend>(
    index: i64,
    position: usize,
    bound: usize,
    operation: &'static str,
) -> Result<usize, B::Error> {
    if index < 0 {
        return Err(error::<B>(BackendError::InvalidNumericInput {
            operation,
            reason: format!("negative COO index {index} at position {position}"),
        }));
    }
    let coordinate = usize::try_from(index).map_err(|_| {
        error::<B>(BackendError::IndexOutOfRange {
            operation,
            position,
            index: usize::MAX,
            bound,
        })
    })?;
    if coordinate >= bound {
        return Err(error::<B>(BackendError::IndexOutOfRange {
            operation,
            position,
            index: coordinate,
            bound,
        }));
    }
    Ok(coordinate)
}

#[cfg(test)]
mod tests {
    use coeus_core::{BackendError, Layout, SequentialBackend};
    use coeus_tensor::Tensor;

    use super::validate_read;

    #[test]
    fn rejects_overflowing_shape_before_numel_or_slice_access() {
        let mut tensor = Tensor::<f32, SequentialBackend>::from_slice([1], &[1.0])
            .expect("tensor allocation succeeds");
        let (_, layout) = tensor.storage_and_layout_mut();
        *layout = Layout::from_shape_strides([usize::MAX, 2].into(), smallvec::smallvec![2, 1], 0);

        assert!(matches!(
            validate_read(&tensor, "sparse validation", None),
            Err(BackendError::Overflow {
                operation: "sparse validation",
                ..
            })
        ));
    }

    #[test]
    fn rejects_mismatched_layout_rank_before_contiguous_read() {
        let mut tensor = Tensor::<f32, SequentialBackend>::from_slice([1, 1], &[1.0])
            .expect("tensor allocation succeeds");
        let (_, layout) = tensor.storage_and_layout_mut();
        *layout = Layout::from_shape_strides([1, 1].into(), smallvec::smallvec![1], 0);

        assert!(matches!(
            validate_read(&tensor, "sparse validation", None),
            Err(BackendError::Storage {
                operation: "sparse validation",
                ..
            })
        ));
    }
}
