// ── Tensor ──
// Core N-dimensional tensor type.

use std::marker::PhantomData;

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Layout,
    MoiraiBackend, Scalar, Shape, Storage, StorageMut, Strides,
};
use thiserror::Error;

/// Failure while transferring a tensor between different backends.
#[derive(Debug, Error)]
pub enum TensorTransferError<SourceError, DestinationError>
where
    SourceError: std::error::Error + 'static,
    DestinationError: std::error::Error + 'static,
{
    /// The source backend could not copy its storage to the host.
    #[error("source backend transfer failed")]
    Source(#[source] SourceError),
    /// The destination backend could not allocate or populate storage.
    #[error("destination backend transfer failed")]
    Destination(#[source] DestinationError),
}

/// Generic N-dimensional tensor.
///
/// # Type parameters
/// - `T`: scalar element type (f32, f64, etc.)
/// - `B`: execution backend (default: `MoiraiBackend`)
///
/// # COW semantics
/// Mutation triggers copy-on-write if storage is shared.
/// Views (slice, transpose) share the underlying storage.
///
/// # Examples
///
/// Create a 2×3 tensor from a flat slice and inspect its shape:
///
/// ```
/// use coeus_tensor::Tensor;
///
/// let t: Tensor<f32> = Tensor::from_slice([2, 3], &[1., 2., 3., 4., 5., 6.])
///     .expect("example tensor allocation succeeds");
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.numel(), 6);
/// assert_eq!(t.as_slice(), &[1., 2., 3., 4., 5., 6.]);
/// ```
///
/// Zero-copy views share storage:
///
/// ```
/// use coeus_tensor::Tensor;
///
/// let t: Tensor<f32> = Tensor::from_slice([2, 3], &[1., 2., 3., 4., 5., 6.])
///     .expect("example tensor allocation succeeds");
/// let row = t.slice(&[(0, 1), (0, 3)]); // first row
/// assert_eq!(row.shape(), &[1, 3]);
/// assert_eq!(row.as_slice(), &[1., 2., 3.]);
/// ```
pub struct Tensor<T: Scalar, B: ComputeBackend = MoiraiBackend> {
    pub(crate) storage: B::DeviceBuffer<T>,
    pub(crate) layout: Layout,
    pub(crate) _backend: PhantomData<B>,
}

impl<T: Scalar, B: ComputeBackend> Clone for Tensor<T, B> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.clone(),
            _backend: PhantomData,
        }
    }
}

// ── Basic accessors ──

impl<T: Scalar, B: ComputeBackend> Tensor<T, B> {
    #[inline]
    fn checked_numel(shape: &Shape, operation: &'static str) -> Result<usize, B::Error> {
        shape.iter().try_fold(1usize, |product, &dimension| {
            product.checked_mul(dimension).ok_or_else(|| {
                BackendError::Overflow {
                    operation,
                    reason: "shape element count exceeds usize",
                }
                .into()
            })
        })
    }

    /// Number of dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_core::SequentialBackend;
    ///
    /// let t = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 4], &[0.0; 24]);
    /// assert_eq!(t.ndim(), 3);
    /// ```
    #[inline]
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Total number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_core::SequentialBackend;
    ///
    /// let t = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 4], &[0.0; 24]);
    /// assert_eq!(t.numel(), 24);
    /// ```
    #[inline]
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Shape as slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_core::SequentialBackend;
    ///
    /// let t = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &[0.0; 6]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Clone shape.
    #[inline]
    pub fn shape_cloned(&self) -> Shape {
        self.layout.shape_cloned()
    }

    /// Strides as slice.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    /// Clone strides.
    #[inline]
    pub fn strides_cloned(&self) -> Strides {
        self.layout.strides_cloned()
    }

    /// Reference to layout.
    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Reference to storage.
    #[inline]
    pub fn storage(&self) -> &B::DeviceBuffer<T> {
        &self.storage
    }

    /// Mutable reference to storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut B::DeviceBuffer<T> {
        self.storage.make_unique();
        &mut self.storage
    }

    /// Mutable reference to storage and reference to layout.
    #[inline]
    pub fn storage_mut_and_layout(&mut self) -> (&mut B::DeviceBuffer<T>, &Layout) {
        self.storage.make_unique();
        (&mut self.storage, &self.layout)
    }

    /// Mutable references to storage and layout without eagerly making the
    /// storage unique.
    ///
    /// Backend operations that mutate the existing allocation must preserve
    /// copy-on-write through [`StorageMut::make_unique`]. Operations that
    /// replace the allocation can instead read the shared storage and install
    /// a compact result without first copying the source allocation.
    #[inline]
    pub fn storage_and_layout_mut(&mut self) -> (&mut B::DeviceBuffer<T>, &mut Layout) {
        (&mut self.storage, &mut self.layout)
    }

    /// True if the layout is row-major contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Materialize logical tensor values in row-major order on the host.
    ///
    /// The backend performs the device-to-host transfer when its storage is
    /// not directly CPU-addressable. Views are compacted according to their
    /// layout, so offsets and strides do not leak into the returned buffer.
    pub fn to_vec_on(&self, backend: &B) -> Result<Vec<T>, B::Error> {
        if let Some(host_slice) = self.storage.try_as_slice() {
            return coeus_leto::contiguous_values(&self.layout, host_slice).map_err(|error| {
                BackendError::Storage {
                    operation: "tensor host materialization",
                    reason: error.to_string(),
                }
                .into()
            });
        }

        let mut physical = vec![T::zero(); Storage::len(&self.storage)];
        backend.copy_to_host(&self.storage, &mut physical)?;
        coeus_leto::contiguous_values(&self.layout, &physical).map_err(|error| {
            BackendError::Storage {
                operation: "tensor host materialization",
                reason: error.to_string(),
            }
            .into()
        })
    }

    /// Expose logical row-major host values, borrowing when storage permits.
    ///
    /// Contiguous CPU-addressable tensors borrow their storage. Offset,
    /// strided, and device-backed tensors materialize through the backend.
    pub fn host_cow_on<'a>(&'a self, backend: &B) -> Result<std::borrow::Cow<'a, [T]>, B::Error> {
        if self.is_contiguous() {
            if let Some(host_slice) = self.storage.try_as_slice() {
                let start = self.layout.offset();
                return Ok(std::borrow::Cow::Borrowed(
                    &host_slice[start..start + self.numel()],
                ));
            }
        }
        Ok(std::borrow::Cow::Owned(self.to_vec_on(backend)?))
    }
}

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Borrow data as contiguous slice.
    ///
    /// # Panics
    /// If the tensor is not contiguous.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        assert!(self.is_contiguous(), "as_slice requires contiguous tensor");
        let start = self.layout.offset();
        let len = self.numel();
        &self.storage.as_slice()[start..start + len]
    }

    /// Get element at logical index.
    #[inline]
    pub fn get(&self, index: &[usize]) -> T
    where
        T: Copy,
    {
        let off = self.layout.physical_index(index);
        self.storage.as_slice()[off]
    }
}

// ── Mutation & Scalar-bound operations ──

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Mutably borrow data as contiguous slice.
    ///
    /// Triggers COW if storage is shared.
    ///
    /// # Panics
    /// If the tensor is not contiguous.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            self.is_contiguous(),
            "as_mut_slice requires contiguous tensor"
        );
        let start = self.layout.offset();
        let len = self.numel();
        &mut self.storage.as_mut_slice()[start..start + len]
    }

    /// Set element at logical index (triggers COW if shared).
    #[inline]
    pub fn set(&mut self, index: &[usize], val: T) {
        let off = self.layout.physical_index(index);
        self.storage.as_mut_slice()[off] = val;
    }
}

impl<T: Scalar, B: ComputeBackend + Default> Tensor<T, B> {
    /// Expose logical row-major host values on `B::default()`.
    #[inline]
    pub fn host_cow(&self) -> Result<std::borrow::Cow<'_, [T]>, B::Error> {
        self.host_cow_on(&B::default())
    }

    /// Materialize logical tensor values in row-major order on the host.
    #[inline]
    pub fn to_vec(&self) -> Result<Vec<T>, B::Error> {
        self.to_vec_on(&B::default())
    }

    /// Make this tensor contiguous in-place on the given backend.
    #[inline]
    pub fn make_contiguous_on(&mut self, backend: &B) -> Result<(), B::Error> {
        if self.is_contiguous() {
            return Ok(());
        }
        *self = self.to_contiguous_on(backend)?;
        Ok(())
    }

    /// Full (non-view) copy of the tensor, compact and contiguous on the given backend.
    #[inline]
    pub fn to_contiguous_on(&self, backend: &B) -> Result<Self, B::Error> {
        if self.is_contiguous() && self.layout.offset() == 0 {
            return Ok(self.clone());
        }
        let values = self.to_vec_on(backend)?;
        Self::from_slice_on(self.shape_cloned(), &values, backend)
    }

    /// Make this tensor contiguous in-place.
    #[inline]
    pub fn make_contiguous(&mut self) -> Result<(), B::Error> {
        self.make_contiguous_on(&B::default())
    }

    /// Full (non-view) copy of the tensor, compact and contiguous.
    #[inline]
    pub fn to_contiguous(&self) -> Result<Self, B::Error> {
        self.to_contiguous_on(&B::default())
    }
}

// ── Generic constructors & device transfers ──

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
    /// every element before reading. This is used internally by kernel dispatch
    /// functions that unconditionally overwrite the output (e.g., `elementwise_unary`,
    /// `elementwise_binary`) to avoid a redundant zero-fill pass.
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
    ///
    /// # Errors
    /// Returns a storage error when the shape size does not match `data`,
    /// overflows `usize`, or the backend cannot allocate or copy the values.
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
    ///
    /// # Performance
    /// - Zero-copy slice cast (bytemuck) if source is host addressable.
    /// - Intermediate host-buffer allocation scaled to `numel()` rather than the full physical buffer layout.
    pub fn to_backend_on<NewB: ComputeBackend>(
        &self,
        src_backend: &B,
        dst_backend: &NewB,
    ) -> Result<Tensor<T, NewB>, TensorTransferError<B::Error, NewB::Error>> {
        if std::any::TypeId::of::<B>() == std::any::TypeId::of::<NewB>() {
            let cloned_storage = self.storage.clone();
            // SAFETY: Since B and NewB are the same type, B::DeviceBuffer<T> and NewB::DeviceBuffer<T> are the same type.
            // We transmute the cloned device buffer to the destination device buffer type.
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

        if let Some(host_slice) = self.storage.try_as_slice() {
            let start = self.layout.offset();
            if self.is_contiguous() {
                dst_backend
                    .copy_to_device(&host_slice[start..start + numel], &mut dst_storage)
                    .map_err(TensorTransferError::Destination)?;
            } else {
                let host_data =
                    coeus_leto::contiguous_values(&self.layout, host_slice).map_err(|error| {
                        TensorTransferError::Source(
                            BackendError::Storage {
                                operation: "tensor backend transfer",
                                reason: error.to_string(),
                            }
                            .into(),
                        )
                    })?;
                dst_backend
                    .copy_to_device(&host_data, &mut dst_storage)
                    .map_err(TensorTransferError::Destination)?;
            }
        } else {
            let storage_len = Storage::len(&self.storage);
            let mut full_host_storage = vec![T::zero(); storage_len];
            src_backend
                .copy_to_host(&self.storage, &mut full_host_storage)
                .map_err(TensorTransferError::Source)?;

            if self.is_contiguous() {
                let start = self.layout.offset();
                dst_backend
                    .copy_to_device(&full_host_storage[start..start + numel], &mut dst_storage)
                    .map_err(TensorTransferError::Destination)?;
            } else {
                let host_data = coeus_leto::contiguous_values(&self.layout, &full_host_storage)
                    .map_err(|error| {
                        TensorTransferError::Source(
                            BackendError::Storage {
                                operation: "tensor backend transfer",
                                reason: error.to_string(),
                            }
                            .into(),
                        )
                    })?;
                dst_backend
                    .copy_to_device(&host_data, &mut dst_storage)
                    .map_err(TensorTransferError::Destination)?;
            }
        }

        Ok(Tensor {
            storage: dst_storage,
            layout: Layout::new(self.shape_cloned()),
            _backend: PhantomData,
        })
    }
}

impl<T: Scalar, B: ComputeBackend + Default> Tensor<T, B> {
    /// Create a new tensor filled with zeros.
    #[inline]
    pub fn zeros<S: Into<Shape>>(shape: S) -> Result<Self, B::Error> {
        Self::zeros_on(shape, &B::default())
    }

    /// Create a new tensor filled with ones.
    #[inline]
    pub fn ones<S: Into<Shape>>(shape: S) -> Result<Self, B::Error> {
        Self::ones_on(shape, &B::default())
    }

    /// Create a new tensor filled with a constant value.
    #[inline]
    pub fn full<S: Into<Shape>>(shape: S, value: T) -> Result<Self, B::Error> {
        Self::full_on(shape, value, &B::default())
    }

    /// Create from a slice of data and a shape.
    ///
    /// # Errors
    /// Returns a storage error when the shape does not match `data` or the
    /// backend cannot allocate or copy the values.
    #[inline]
    pub fn from_slice<S: Into<Shape>>(shape: S, data: &[T]) -> Result<Self, B::Error> {
        Self::from_slice_on(shape, data, &B::default())
    }

    /// Create a 1-D tensor from a vector.
    #[inline]
    pub fn from_vec(data: Vec<T>) -> Result<Self, B::Error> {
        let n = data.len();
        Self::from_slice([n], &data)
    }

    /// Copy tensor memory to a new backend.
    pub fn to_backend<NewB: ComputeBackend + Default>(
        &self,
        backend: &NewB,
    ) -> Result<Tensor<T, NewB>, TensorTransferError<B::Error, NewB::Error>> {
        self.to_backend_on(&B::default(), backend)
    }
}
