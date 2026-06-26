// ── Tensor ──
// Core N-dimensional tensor type.

use std::marker::PhantomData;

use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, MoiraiBackend, Scalar,
    Shape, Storage, StorageMut, Strides,
};

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
/// let t: Tensor<f32> = Tensor::from_slice([2, 3], &[1., 2., 3., 4., 5., 6.]);
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
/// let t: Tensor<f32> = Tensor::from_slice([2, 3], &[1., 2., 3., 4., 5., 6.]);
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

    /// True if the layout is row-major contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
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
    /// Make this tensor contiguous in-place on the given backend.
    #[inline]
    pub fn make_contiguous_on(&mut self, backend: &B) {
        if self.is_contiguous() {
            return;
        }
        *self = self.to_contiguous_on(backend);
    }

    /// Full (non-view) copy of the tensor, compact and contiguous on the given backend.
    #[inline]
    pub fn to_contiguous_on(&self, backend: &B) -> Self {
        if self.is_contiguous() && self.layout.offset() == 0 {
            return self.clone();
        }
        // Direct compaction for CPU-addressable storage to avoid recursion and host-to-device transfers
        if let Some(src_slice) = self.storage.try_as_slice() {
            let values = coeus_leto::contiguous_values(&self.layout, src_slice)
                .expect("coeus-leto contiguous materialization failed");
            return Self::from_slice_on(self.shape_cloned(), &values, backend);
        }
        let host_backend = MoiraiBackend::new();
        let host_tensor = self.to_backend_on(backend, &host_backend);
        let host_contiguous = host_tensor.to_contiguous();
        host_contiguous.to_backend_on(&host_backend, backend)
    }

    /// Make this tensor contiguous in-place.
    #[inline]
    pub fn make_contiguous(&mut self) {
        self.make_contiguous_on(&B::default());
    }

    /// Full (non-view) copy of the tensor, compact and contiguous.
    #[inline]
    pub fn to_contiguous(&self) -> Self {
        self.to_contiguous_on(&B::default())
    }
}

// ── Generic constructors & device transfers ──

impl<T: Scalar, B: ComputeBackend> Tensor<T, B> {
    /// Create a new tensor filled with zeros on the given backend.
    #[inline]
    pub fn zeros_on<S: Into<Shape>>(shape: S, backend: &B) -> Self {
        let shape = shape.into();
        let numel: usize = shape.iter().product();
        let mut storage = backend.allocate(numel);
        backend.fill(&mut storage, T::zero());
        let layout = Layout::new(shape);
        Self {
            storage,
            layout,
            _backend: PhantomData,
        }
    }

    /// Create a new tensor filled with ones on the given backend.
    #[inline]
    pub fn ones_on<S: Into<Shape>>(shape: S, backend: &B) -> Self {
        let shape = shape.into();
        let numel: usize = shape.iter().product();
        let mut storage = backend.allocate(numel);
        backend.fill(&mut storage, T::one());
        let layout = Layout::new(shape);
        Self {
            storage,
            layout,
            _backend: PhantomData,
        }
    }

    /// Create a new tensor filled with a constant value on the given backend.
    #[inline]
    pub fn full_on<S: Into<Shape>>(shape: S, value: T, backend: &B) -> Self {
        let shape = shape.into();
        let numel: usize = shape.iter().product();
        let mut storage = backend.allocate(numel);
        backend.fill(&mut storage, value);
        let layout = Layout::new(shape);
        Self {
            storage,
            layout,
            _backend: PhantomData,
        }
    }

    /// Create from a slice of data and a shape on the given backend.
    ///
    /// # Panics
    /// If `data.len() != shape.numel()`.
    #[inline]
    pub fn from_slice_on<S: Into<Shape>>(shape: S, data: &[T], backend: &B) -> Self {
        let shape = shape.into();
        let numel: usize = shape.iter().product();
        assert_eq!(numel, data.len(), "data size mismatch for shape");
        let mut storage = backend.allocate(numel);
        backend.copy_to_device(data, &mut storage);
        let layout = Layout::new(shape);
        Self {
            storage,
            layout,
            _backend: PhantomData,
        }
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
    ) -> Tensor<T, NewB> {
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
            return Tensor {
                storage: dst_storage,
                layout: self.layout.clone(),
                _backend: PhantomData,
            };
        }

        let numel = self.numel();
        let mut dst_storage = dst_backend.allocate(numel);

        if let Some(host_slice) = self.storage.try_as_slice() {
            let start = self.layout.offset();
            if self.is_contiguous() {
                dst_backend.copy_to_device(&host_slice[start..start + numel], &mut dst_storage);
            } else {
                let host_data = coeus_leto::contiguous_values(&self.layout, host_slice)
                    .expect("coeus-leto backend transfer materialization failed");
                dst_backend.copy_to_device(&host_data, &mut dst_storage);
            }
        } else {
            let storage_len = Storage::len(&self.storage);
            let mut full_host_storage = vec![T::zero(); storage_len];
            src_backend.copy_to_host(&self.storage, &mut full_host_storage);

            if self.is_contiguous() {
                let start = self.layout.offset();
                dst_backend
                    .copy_to_device(&full_host_storage[start..start + numel], &mut dst_storage);
            } else {
                let host_data = coeus_leto::contiguous_values(&self.layout, &full_host_storage)
                    .expect("coeus-leto backend transfer materialization failed");
                dst_backend.copy_to_device(&host_data, &mut dst_storage);
            }
        }

        Tensor {
            storage: dst_storage,
            layout: Layout::new(self.shape_cloned()),
            _backend: PhantomData,
        }
    }
}

impl<T: Scalar, B: ComputeBackend + Default> Tensor<T, B> {
    /// Create a new tensor filled with zeros.
    #[inline]
    pub fn zeros<S: Into<Shape>>(shape: S) -> Self {
        Self::zeros_on(shape, &B::default())
    }

    /// Create a new tensor filled with ones.
    #[inline]
    pub fn ones<S: Into<Shape>>(shape: S) -> Self {
        Self::ones_on(shape, &B::default())
    }

    /// Create a new tensor filled with a constant value.
    #[inline]
    pub fn full<S: Into<Shape>>(shape: S, value: T) -> Self {
        Self::full_on(shape, value, &B::default())
    }

    /// Create from a slice of data and a shape.
    ///
    /// # Panics
    /// If `data.len() != shape.numel()`.
    #[inline]
    pub fn from_slice<S: Into<Shape>>(shape: S, data: &[T]) -> Self {
        Self::from_slice_on(shape, data, &B::default())
    }

    /// Create a 1-D tensor from a vector.
    #[inline]
    pub fn from_vec(data: Vec<T>) -> Self {
        let n = data.len();
        Self::from_slice([n], &data)
    }

    /// Copy tensor memory to a new backend.
    pub fn to_backend<NewB: ComputeBackend + Default>(&self, backend: &NewB) -> Tensor<T, NewB> {
        self.to_backend_on(&B::default(), backend)
    }
}
