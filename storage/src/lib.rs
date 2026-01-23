//! # Coeus Storage Abstractions
//!
//! Provides memory layout and storage primitives for multi-dimensional tensors.
//!
//! ## Architecture
//!
//! Storage abstractions separate memory layout concerns from compute backend logic,
//! enabling flexible tensor implementations with zero-cost abstractions.
//!
//! ### Storage Trait Hierarchy
//!
//! ```text
//! Storage<T: DataType>
//! ├── DenseStorage<T>       // Contiguous memory, row-major (C-contiguous)
//! ├── QuantizedStorage<T>   // Packed quantized values (4/8/16-bit)
//! │   ├── QuantizedStorage4<T>  // 4-bit quantization
//! │   ├── QuantizedStorage8<T>  // 8-bit quantization
//! │   └── QuantizedStorage16<T> // 16-bit quantization
//! ├── StridedStorage<T>     // Custom strides for views/transpose
//! ├── SparseStorage<T>      // CSR/CSC/COO formats
//! │   ├── CsrStorage<T>     // Compressed Sparse Row
//! │   ├── CscStorage<T>     // Compressed Sparse Column
//! │   └── CooStorage<T>     // Coordinate format
//! └── DistributedStorage<T> // Multi-device tensor storage
//! ```
//!
//! ## Memory Layout
//!
//! **Row-Major Ordering** (default, matches NumPy/PyTorch):
//! ```text
//! Shape [2, 3]:
//! [[a, b, c],    Memory: [a, b, c, d, e, f]
//!  [d, e, f]]    
//!
//! Stride: [3, 1]  (row stride=3, col stride=1)
//! ```
//!
//! **Column-Major Ordering** (opt-in, for BLAS/Fortran interop):
//! ```text
//! Shape [2, 3]:
//! [[a, b, c],    Memory: [a, d, b, e, c, f]
//!  [d, e, f]]    
//!
//! Stride: [1, 2]  (row stride=1, col stride=2)
//! ```
//!
//! ## Safety
//!
//! All storage operations are memory-safe. Shape/stride validation prevents
//! out-of-bounds access, and Rust ownership ensures no data races.

#![no_std]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub use alloc::{vec, vec::Vec};
pub use dtype::DataType;
pub use num_traits;

// Core infrastructure
pub mod error;
pub mod iter;
pub mod shape;
pub mod traits; // Unified storage operation traits

// Storage implementations
pub mod dense;
pub mod quantized;
pub mod strided;

// Sparse storage
pub mod sparse;
pub mod sparse_indexing;

// Distributed storage
pub mod broadcast;
pub mod distributed;

pub use dense::DenseStorage;
pub use distributed::{DistributedStorage, ReduceOperation, ShardingStrategy};
pub use strided::StridedStorage;

// Re-export sparse types from sparse module for API stability
pub use sparse::{CsrStorage, CscStorage, CooStorage, SparseFormat, SparseStorage};

/// Trait for dynamic downcasting of storage types
pub trait AsAny {
    /// Get as Any reference for downcasting
    fn as_any(&self) -> &dyn core::any::Any;
}

/// Trait for storage types that can be created from vectors
///
/// This enables generic tensor creation methods that work with any storage
/// type supporting vector-based initialization.
pub trait StorageFromVec<T: crate::DataType>: Storage<T> {
    /// Create storage from a vector with the given dimensions
    ///
    /// # Errors
    /// Returns error if dimensions don't match the vector length
    fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized;

    /// Create storage filled with zeros
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
        T: num_traits::Zero;

    /// Create storage filled with ones
    ///
    /// # Errors
    /// Returns error if dimensions are invalid
    fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
        T: num_traits::One;
}

/// Trait for converting storage to dense representation
///
/// This enables gradient operations that require dense storage internally,
/// while maintaining generic storage type support at the API level.
pub trait StorageToDense<T: crate::DataType>: Storage<T> {
    /// Convert this storage to dense representation
    ///
    /// # Errors
    /// Returns error if conversion fails
    fn to_dense(&self) -> crate::Result<DenseStorage<T>>;
}
pub use error::StorageError;
pub use quantized::{QuantizedStorage, QuantizedStorage16, QuantizedStorage4, QuantizedStorage8};
pub use shape::Shape;
// Sparse types already exported above (line 89)
pub use sparse_indexing::{SparseBooleanIndex, SparseFancyIndex};


/// Result type for storage operations
pub type Result<T> = core::result::Result<T, StorageError>;

/// Storage format enumeration for runtime type identification
///
/// Enables zero-cost dispatch based on storage format without dynamic typing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageFormat {
    /// Dense contiguous storage (row-major or column-major)
    Dense,
    /// Compressed Sparse Row format
    Csr,
    /// Compressed Sparse Column format
    Csc,
    /// Coordinate format (COO)
    Coo,
    /// Quantized storage (4/8/16-bit)
    Quantized,
    /// Strided storage (for views and transposes)
    Strided,
    /// Distributed storage (multi-device)
    Distributed,
}

impl StorageFormat {
    /// Returns true if this format is sparse
    #[must_use]
    pub fn is_sparse(self) -> bool {
        matches!(self, Self::Csr | Self::Csc | Self::Coo)
    }

    /// Returns true if this format is dense
    #[must_use]
    pub fn is_dense(self) -> bool {
        matches!(self, Self::Dense | Self::Strided)
    }

    /// Returns true if this format is quantized
    #[must_use]
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Quantized)
    }
}

/// Core storage trait for tensor memory layouts.
///
/// Defines the interface all storage backends must implement, enabling
/// zero-cost abstraction over different memory layouts.
///
/// # Type Parameters
///
/// * `T` - Element type implementing `DataType` trait
///
/// # Safety
///
/// Implementations must guarantee:
/// - Valid memory access within bounds
/// - Correct stride calculations
/// - No aliasing violations for mutable access
pub trait Storage<T: DataType>: Send + Sync + Clone + core::fmt::Debug + 'static {
    /// Returns the storage format type for runtime dispatch
    fn format(&self) -> StorageFormat;

    /// Returns a reference to the underlying data as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use storage::{DenseStorage, Storage};
    /// use dtype::float::Float32;
    ///
    /// let storage = DenseStorage::from_slice(&[Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)], &[3]).unwrap();
    /// assert_eq!(storage.as_slice().len(), 3);
    /// ```
    fn as_slice(&self) -> &[T];

    /// Returns a mutable reference to the underlying data as a slice.
    ///
    /// # Safety
    ///
    /// Caller must ensure no aliasing violations if storage is shared.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns the shape of this storage.
    fn shape(&self) -> &Shape;

    /// Returns the strides for each dimension.
    ///
    /// Strides define how many elements to skip to move along each axis.
    fn strides(&self) -> &[usize];

    /// Returns the total number of elements.
    fn len(&self) -> usize {
        self.shape().size()
    }

    /// Returns true if the storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the storage is contiguous (no gaps in memory).
    fn is_contiguous(&self) -> bool;

    /// Returns a reference to self as a Storage trait object.
    fn as_storage_ref(&self) -> &Self;

    /// Creates storage filled with a constant value.
    ///
    /// # Arguments
    /// * `dims` - Shape dimensions
    /// * `value` - Value to fill storage with
    ///
    /// # Errors
    ///
    /// Returns error if shape specification is invalid.
    fn full(dims: &[usize], value: T) -> Result<Self>
    where
        Self: Sized;

    /// Apply a function to the stored structure.
    ///
    /// For sparse storage, this only applies to mapped values.
    /// The function `f` MUST preserve zero (f(0) == 0) for sparse storage correctness.
    fn map_structure<F>(&self, f: F) -> Result<Self>
    where
        Self: Sized,
        F: FnMut(T) -> T;
}
