//! Tensor/storage adaptation for the provider-owned CSR row recurrence.

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use leto_ops::{
    CsrProduct, CsrView, DynamicWidth, FixedWidth, ProductWidth, ScalarStrategy, StoredOrder,
};

use super::super::validation::{validate_csr, validate_read};
use crate::backend_ops::error::map_leto_error;

/// Compute `y = A*x` in stored CSR order with separate native-precision
/// multiplication and addition, retaining the caller's backend row dispatch.
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
    const OPERATION: &str = "sparse matrix-vector multiplication";
    let view = validate_csr(a, OPERATION)?;
    validate_read(x, OPERATION, Some(&[a.shape()[1]]))?;
    // SAFETY: fallible mutable access initializes CPU storage before creating
    // &mut [T]. The bound provider overwrites every output row before success.
    let mut output = unsafe { Tensor::<T, B>::alloc_on([a.shape()[0]], backend)? };
    multiply_into::<T, B, FixedWidth<1>>(
        view,
        x.as_slice(),
        1,
        output.as_mut_slice()?,
        backend,
        OPERATION,
    )?;
    Ok(output)
}

/// Compute `C = A*B` in stored CSR order with separate native-precision
/// multiplication and addition, retaining the caller's backend row dispatch.
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
    const OPERATION: &str = "sparse-dense matrix multiplication";
    let view = validate_csr(a, OPERATION)?;
    if b.ndim() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: OPERATION,
            rank: b.ndim(),
            max_rank: 2,
        }
        .into());
    }
    let [rows, shared] = [a.shape()[0], a.shape()[1]];
    let width = b.shape()[1];
    validate_read(b, OPERATION, Some(&[shared, width]))?;
    // SAFETY: mutable access initializes typed storage; the provider then
    // writes every row, including empty rows. alloc_on retains its original
    // checked count/allocation errors before provider preparation.
    let mut output = unsafe { Tensor::<T, B>::alloc_on([rows, width], backend)? };
    multiply_into::<T, B, DynamicWidth>(
        view,
        b.as_slice(),
        width,
        output.as_mut_slice()?,
        backend,
        OPERATION,
    )?;
    Ok(output)
}

// Expose the checked fixed vector width to backend callback specialization.
#[inline(always)]
fn multiply_into<T: Scalar, B: Backend, W: ProductWidth>(
    view: CsrView<'_, T, i64, StoredOrder>,
    dense: &[T],
    width: usize,
    output: &mut [T],
    backend: &B,
    operation: &'static str,
) -> Result<(), B::Error> {
    let product = CsrProduct::<_, _, _, ScalarStrategy, W>::prepare(view, dense, width)
        .map_err(|error| map_leto_error(operation, error))?;
    let bound = product
        .bind_output(output)
        .map_err(|error| map_leto_error(operation, error))?;
    // SAFETY: Backend's unsafe contract (ADR 0081) visits exactly the in-range
    // rows once on success, at most once on panic, and joins all admitted work
    // before return/unwind. No task escapes this closure. The provider retains
    // immutable inputs and the exclusive output borrow, and proves disjoint
    // complete row writes; Coeus performs no sparse pointer math or arithmetic.
    unsafe {
        bound.dispatch_with(|rows, task| {
            backend.parallel_for(0, rows, move |row| task.run(row));
        });
    }
    Ok(())
}
