use coeus_core::{ComputeBackend, Scalar, Storage, StorageMut};
use coeus_tensor::Tensor;
use std::borrow::Cow;

/// Retrieve a tensor's data as a host slice (using Cow to avoid heap allocation if the storage is already CPU-addressable and contiguous).
pub(crate) fn get_tensor_host_data<'a, T: Scalar, B: ComputeBackend>(
    tensor: &'a Tensor<T, B>,
    backend: &B,
) -> Result<Cow<'a, [T]>, B::Error> {
    let numel = tensor.numel();
    if numel == 0 {
        return Ok(Cow::Borrowed(&[]));
    }
    // Zero-copy shortcut if storage is CPU addressable, contiguous, and zero offset.
    if tensor.is_contiguous() && tensor.layout().offset() == 0 {
        if let Some(slice) = tensor.storage().try_as_slice() {
            return Ok(Cow::Borrowed(&slice[..numel]));
        }
    }

    let mut host_data = vec![T::zero(); numel];
    if tensor.is_contiguous() && tensor.layout().offset() == 0 {
        backend.copy_to_host(tensor.storage(), &mut host_data)?;
    } else {
        let storage_len = Storage::len(tensor.storage());
        let mut full_host_storage = vec![T::zero(); storage_len];
        backend.copy_to_host(tensor.storage(), &mut full_host_storage)?;

        let ndim = tensor.ndim();
        let shape = tensor.shape();
        let strides = tensor.strides();
        let mut offset = tensor.layout().offset();
        let mut index = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
        for val in &mut host_data {
            *val = full_host_storage[offset];
            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < shape[d] {
                    offset += strides[d];
                    break;
                }
                offset -= (shape[d] - 1) * strides[d];
                index[d] = 0;
            }
        }
    }
    Ok(Cow::Owned(host_data))
}

/// Copy a host slice back into a tensor (leveraging try_as_mut_slice for CPU zero-copy).
pub(crate) fn copy_host_slice_to_tensor<T: Scalar, B: ComputeBackend>(
    host_data: &[T],
    tensor: &mut Tensor<T, B>,
    backend: &B,
) -> Result<(), B::Error> {
    if tensor.is_contiguous() && tensor.layout().offset() == 0 {
        if let Some(slice) = tensor.storage_mut()?.try_as_mut_slice()? {
            slice[..host_data.len()].copy_from_slice(host_data);
        } else {
            backend.copy_to_device(host_data, tensor.storage_mut()?)?;
        }
    } else {
        let storage_len = Storage::len(tensor.storage());
        let mut full_host_storage = vec![T::zero(); storage_len];
        backend.copy_to_host(tensor.storage(), &mut full_host_storage)?;

        let ndim = tensor.ndim();
        let shape = tensor.shape();
        let strides = tensor.strides();
        let mut offset = tensor.layout().offset();
        let mut index = smallvec::SmallVec::<[usize; 4]>::from_elem(0, ndim);
        for &val in host_data {
            full_host_storage[offset] = val;
            for d in (0..ndim).rev() {
                index[d] += 1;
                if index[d] < shape[d] {
                    offset += strides[d];
                    break;
                }
                offset -= (shape[d] - 1) * strides[d];
                index[d] = 0;
            }
        }
        backend.copy_to_device(&full_host_storage, tensor.storage_mut()?)?;
    }
    Ok(())
}

/// Borrow or retrieve a tensor's data as a raw byte slice on host and apply a function.
pub(crate) fn with_tensor_host_bytes<T: Scalar, B: ComputeBackend, F, R>(
    tensor: &Tensor<T, B>,
    backend: &B,
    f: F,
) -> Result<R, B::Error>
where
    F: FnOnce(&[u8]) -> Result<R, B::Error>,
{
    let numel = tensor.numel();
    if numel == 0 {
        return f(&[]);
    }
    let bytes_len = numel * std::mem::size_of::<T>();
    let host_data = get_tensor_host_data(tensor, backend)?;
    let raw_ptr = host_data.as_ptr() as *const u8;
    // SAFETY: `host_data` owns or borrows exactly `numel` initialized `T` values;
    // the byte view is used only for the duration of `f` and has the same extent.
    let raw_slice = unsafe { std::slice::from_raw_parts(raw_ptr, bytes_len) };
    f(raw_slice)
}

/// Try to receive data directly into the tensor's mut slice, or fallback to host buffer.
pub(crate) fn recv_tensor_data<T: Scalar, B: ComputeBackend, F>(
    tensor: &mut Tensor<T, B>,
    backend: &B,
    recv_fn: F,
) -> Result<(), B::Error>
where
    F: FnOnce(&mut [u8]),
{
    let numel = tensor.numel();
    if numel == 0 {
        return Ok(());
    }
    let bytes_len = numel * std::mem::size_of::<T>();

    if tensor.is_contiguous() && tensor.layout().offset() == 0 {
        if let Some(slice) = tensor.storage_mut()?.try_as_mut_slice()? {
            let raw_ptr = slice.as_mut_ptr() as *mut u8;
            // SAFETY: `slice` is a mutable view of initialized `numel` values and
            // the receiver writes exactly its byte extent.
            let raw_slice = unsafe { std::slice::from_raw_parts_mut(raw_ptr, bytes_len) };
            recv_fn(raw_slice);
            return Ok(());
        }
    }

    let mut host_data = vec![T::zero(); numel];
    let raw_ptr = host_data.as_mut_ptr() as *mut u8;
    // SAFETY: the byte slice covers the allocated `numel`-element host buffer;
    // the receiver initializes that exact extent before it is copied to storage.
    let raw_slice = unsafe { std::slice::from_raw_parts_mut(raw_ptr, bytes_len) };
    recv_fn(raw_slice);
    copy_host_slice_to_tensor(&host_data, tensor, backend)
}

/// Helper to read raw bytes from a source into a mutable host slice.
pub(crate) fn recv_slice_data<T: Scalar, F>(data: &mut [T], recv_fn: F)
where
    F: FnOnce(&mut [u8]),
{
    if data.is_empty() {
        return;
    }
    let bytes_len = std::mem::size_of_val(data);
    let raw_ptr = data.as_mut_ptr() as *mut u8;
    let raw_slice = unsafe { std::slice::from_raw_parts_mut(raw_ptr, bytes_len) };
    recv_fn(raw_slice);
}
