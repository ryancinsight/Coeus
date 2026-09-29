use std::marker::PhantomData;

use coeus_core::{BackendError, ComputeBackend, Layout, Scalar, Shape, Storage};

use super::{Tensor, TensorTransferError};

impl<T: Scalar, B: ComputeBackend> Tensor<T, B> {
    #[inline(always)]
    fn from_storage_and_shape(storage: B::DeviceBuffer<T>, shape: Shape) -> Self {
        Self {
            storage,
            layout: Layout::new(shape),
            _backend: PhantomData,
        }
    }

    /// Allocate a tensor with the given shape without initializing the elements.
    ///
    /// # Safety
    /// The returned tensor's contents are unspecified. Callers **must** write
    /// every element before reading.
    #[inline]
    pub fn alloc_on<S: Into<Shape>>(shape: S, backend: &B) -> Result<Self, B::Error> {
        let shape = shape.into();
        let numel = Self::checked_numel(&shape, "Tensor::alloc_on")?;
        Ok(Self::from_storage_and_shape(
            backend.allocate(numel)?,
            shape,
        ))
    }

    /// Create a new tensor filled with zeros on the given backend.
    #[inline]
    pub fn zeros_on<S: Into<Shape>>(shape: S, backend: &B) -> Result<Self, B::Error> {
        let shape = shape.into();
        let numel = Self::checked_numel(&shape, "Tensor::zeros_on")?;
        Ok(Self::from_storage_and_shape(
            backend.allocate_zeroed(numel)?,
            shape,
        ))
    }

    /// Create a new tensor filled with ones on the given backend.
    #[inline]
    pub fn ones_on<S: Into<Shape>>(shape: S, backend: &B) -> Result<Self, B::Error> {
        let shape = shape.into();
        let numel = Self::checked_numel(&shape, "Tensor::ones_on")?;
        let mut storage = backend.allocate(numel)?;
        backend.fill(&mut storage, T::one())?;
        Ok(Self::from_storage_and_shape(storage, shape))
    }

    /// Create a new tensor filled with a constant value on the given backend.
    #[inline]
    pub fn full_on<S: Into<Shape>>(shape: S, value: T, backend: &B) -> Result<Self, B::Error> {
        let shape = shape.into();
        let numel = Self::checked_numel(&shape, "Tensor::full_on")?;
        let mut storage = backend.allocate(numel)?;
        backend.fill(&mut storage, value)?;
        Ok(Self::from_storage_and_shape(storage, shape))
    }

    /// Create from a slice of data and a shape on the given backend.
    #[inline]
    pub fn from_slice_on<S: Into<Shape>>(
        shape: S,
        data: &[T],
        backend: &B,
    ) -> Result<Self, B::Error> {
        let shape = shape.into();
        let numel = Self::checked_numel(&shape, "Tensor::from_slice_on")?;
        if numel != data.len() {
            return Err(BackendError::Storage {
                operation: "Tensor::from_slice_on",
                reason: format!(
                    "shape requires {numel} elements but source contains {}",
                    data.len()
                ),
            }
            .into());
        }
        let mut storage = backend.allocate(numel)?;
        backend.copy_to_device(data, &mut storage)?;
        Ok(Self::from_storage_and_shape(storage, shape))
    }

    /// Construct a tensor from its raw storage and layout parts.
    #[inline]
    pub fn from_raw_parts(storage: B::DeviceBuffer<T>, layout: Layout) -> Self {
        Self {
            storage,
            layout,
            _backend: PhantomData,
        }
    }

    /// Copy tensor memory to a new backend using explicit backend references.
    pub fn to_backend_on<NewB: ComputeBackend>(
        &self,
        src_backend: &B,
        dst_backend: &NewB,
    ) -> Result<Tensor<T, NewB>, TensorTransferError<B::Error, NewB::Error>> {
        if std::any::TypeId::of::<B>() == std::any::TypeId::of::<NewB>() {
            let cloned_storage = self.storage.clone();
            // SAFETY: B and NewB are the same type, so their device buffers match.
            let dst_storage = unsafe {
                assert_eq!(
                    std::mem::size_of::<B::DeviceBuffer<T>>(),
                    std::mem::size_of::<NewB::DeviceBuffer<T>>()
                );
                let dst: NewB::DeviceBuffer<T> = std::mem::transmute_copy(&cloned_storage);
                std::mem::forget(cloned_storage);
                dst
            };
            return Ok(Tensor {
                storage: dst_storage,
                layout: self.layout.clone(),
                _backend: PhantomData,
            });
        }

        let numel = self.numel();
        let mut dst_storage = dst_backend
            .allocate(numel)
            .map_err(TensorTransferError::Destination)?;
        let host_data = if let Some(host_slice) = self.storage.try_as_slice() {
            let start = self.layout.offset();
            if self.is_contiguous() {
                host_slice[start..start + numel].to_vec()
            } else {
                coeus_leto::contiguous_values(&self.layout, host_slice).map_err(|error| {
                    TensorTransferError::Source(
                        BackendError::Storage {
                            operation: "tensor backend transfer",
                            reason: error.to_string(),
                        }
                        .into(),
                    )
                })?
            }
        } else {
            let storage_len = Storage::len(&self.storage);
            let mut full_host_storage = vec![T::zero(); storage_len];
            src_backend
                .copy_to_host(&self.storage, &mut full_host_storage)
                .map_err(TensorTransferError::Source)?;
            let start = self.layout.offset();
            if self.is_contiguous() {
                full_host_storage[start..start + numel].to_vec()
            } else {
                coeus_leto::contiguous_values(&self.layout, &full_host_storage).map_err(
                    |error| {
                        TensorTransferError::Source(
                            BackendError::Storage {
                                operation: "tensor backend transfer",
                                reason: error.to_string(),
                            }
                            .into(),
                        )
                    },
                )?
            }
        };
        dst_backend
            .copy_to_device(&host_data, &mut dst_storage)
            .map_err(TensorTransferError::Destination)?;
        Ok(Tensor {
            storage: dst_storage,
            layout: Layout::new(self.shape_cloned()),
            _backend: PhantomData,
        })
    }
}
