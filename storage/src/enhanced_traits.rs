//! Enhanced storage traits for zero-cost generic dispatch
//!
//! This module defines the enhanced trait hierarchy that enables compile-time
//! dispatch for any storage format (dense, sparse CSR/CSC/COO) on any backend
//! with any datatype.

use crate::{DataType, Shape};
use alloc::vec::Vec;
use core::fmt::Debug;

/// Core storage trait that all storage types must implement
///
/// This trait provides the minimal interface required for any storage format
/// to participate in the tensor system with zero-cost abstractions.
pub trait Storage<T: DataType>: Clone + Send + Sync + Debug + 'static {
    /// Get the shape of the storage
    fn shape(&self) -> &Shape;
    
    /// Get the total number of elements
    fn len(&self) -> usize;
    
    /// Check if the storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the number of non-zero elements (for sparse formats)
    fn nnz(&self) -> usize {
        self.len() // Default implementation for dense
    }
    
    /// Get the sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    fn sparsity(&self) -> f64 {
        let total = self.shape().total_elements();
        if total == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total as f64)
        }
    }
}

/// Dense storage trait for contiguous memory layouts
///
/// Provides direct access to underlying data for maximum performance
/// with zero-cost abstractions.
pub trait DenseStorage<T: DataType>: Storage<T> {
    /// Get a reference to the underlying data slice
    fn as_slice(&self) -> &[T];
    
    /// Get a mutable reference to the underlying data slice
    fn as_mut_slice(&mut self) -> &mut [T];
    
    /// Create from a vector with given shape
    fn from_vec(data: Vec<T>, shape: &[usize]) -> crate::Result<Self>
    where
        Self: Sized;
    
    /// Convert to vector, consuming the storage
    fn into_vec(self) -> Vec<T>;
}

/// CSR (Compressed Sparse Row) storage trait
///
/// Efficient for row-based operations and matrix-vector multiplication.
/// Memory layout: O(nnz) for data and indices, O(rows+1) for indptr.
pub trait CsrStorage<T: DataType>: Storage<T> {
    /// Get the non-zero values
    fn data(&self) -> &[T];
    
    /// Get the column indices for each non-zero value
    fn indices(&self) -> &[usize];
    
    /// Get the row pointers (length = rows + 1)
    fn indptr(&self) -> &[usize];
    
    /// Create from CSR components
    fn from_csr(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
    ) -> crate::Result<Self>
    where
        Self: Sized;
    
    /// Get the number of rows
    fn rows(&self) -> usize {
        if self.indptr().is_empty() {
            0
        } else {
            self.indptr().len() - 1
        }
    }
    
    /// Get the number of columns
    fn cols(&self) -> usize {
        self.shape().dims().get(1).copied().unwrap_or(0)
    }
}

/// CSC (Compressed Sparse Column) storage trait
///
/// Efficient for column-based operations. Mirror of CSR but column-oriented.
/// Memory layout: O(nnz) for data and indices, O(cols+1) for indptr.
pub trait CscStorage<T: DataType>: Storage<T> {
    /// Get the non-zero values
    fn data(&self) -> &[T];
    
    /// Get the row indices for each non-zero value
    fn indices(&self) -> &[usize];
    
    /// Get the column pointers (length = cols + 1)
    fn indptr(&self) -> &[usize];
    
    /// Create from CSC components
    fn from_csc(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
    ) -> crate::Result<Self>
    where
        Self: Sized;
    
    /// Get the number of rows
    fn rows(&self) -> usize {
        self.shape().dims().get(0).copied().unwrap_or(0)
    }
    
    /// Get the number of columns
    fn cols(&self) -> usize {
        if self.indptr().is_empty() {
            0
        } else {
            self.indptr().len() - 1
        }
    }
}

/// COO (Coordinate) storage trait
///
/// Most flexible format for construction and format conversion.
/// Memory layout: O(3*nnz) for data, row_indices, and col_indices.
pub trait CooStorage<T: DataType>: Storage<T> {
    /// Get the non-zero values
    fn data(&self) -> &[T];
    
    /// Get the row indices for each non-zero value
    fn row_indices(&self) -> &[usize];
    
    /// Get the column indices for each non-zero value
    fn col_indices(&self) -> &[usize];
    
    /// Create from COO components
    fn from_coo(
        data: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: &[usize],
    ) -> crate::Result<Self>
    where
        Self: Sized;
    
    /// Get the number of rows
    fn rows(&self) -> usize {
        self.shape().dims().get(0).copied().unwrap_or(0)
    }
    
    /// Get the number of columns
    fn cols(&self) -> usize {
        self.shape().dims().get(1).copied().unwrap_or(0)
    }
    
    /// Check if the COO format is sorted by row then column
    fn is_sorted(&self) -> bool {
        let rows = self.row_indices();
        let cols = self.col_indices();
        
        for i in 1..rows.len() {
            if rows[i] < rows[i-1] || (rows[i] == rows[i-1] && cols[i] < cols[i-1]) {
                return false;
            }
        }
        true
    }
}

/// Conversion traits for zero-cost format transformations
pub trait ToDense<T: DataType> {
    type Output: DenseStorage<T>;
    
    /// Convert to dense format
    fn to_dense(&self) -> crate::Result<Self::Output>;
}

pub trait ToCsr<T: DataType> {
    type Output: CsrStorage<T>;
    
    /// Convert to CSR format
    fn to_csr(&self) -> crate::Result<Self::Output>;
}

pub trait ToCsc<T: DataType> {
    type Output: CscStorage<T>;
    
    /// Convert to CSC format
    fn to_csc(&self) -> crate::Result<Self::Output>;
}

pub trait ToCoo<T: DataType> {
    type Output: CooStorage<T>;
    
    /// Convert to COO format
    fn to_coo(&self) -> crate::Result<Self::Output>;
}

/// Generic storage creation trait
pub trait StorageFromVec<T: DataType>: Storage<T> {
    /// Create storage from a vector with given shape
    fn from_vec(data: Vec<T>, shape: &[usize]) -> crate::Result<Self>
    where
        Self: Sized;
}

/// Storage type enumeration for dynamic dispatch when needed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageType {
    /// Dense contiguous storage
    Dense,
    /// Compressed Sparse Row
    Csr,
    /// Compressed Sparse Column
    Csc,
    /// Coordinate format
    Coo,
    /// Quantized storage
    Quantized,
    /// Strided storage
    Strided,
}

/// Trait for getting storage type information
pub trait StorageInfo<T: DataType>: Storage<T> {
    /// Get the storage type
    fn storage_type(&self) -> StorageType;
    
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
    
    /// Check if storage format is optimal for given operation
    fn is_optimal_for(&self, operation: &str) -> bool {
        match (self.storage_type(), operation) {
            (StorageType::Dense, _) => true, // Dense is always acceptable
            (StorageType::Csr, "matmul" | "spmv" | "row_sum") => true,
            (StorageType::Csc, "matvec" | "col_sum") => true,
            (StorageType::Coo, "construction" | "conversion") => true,
            _ => false,
        }
    }
}

/// Arithmetic operations trait for storage-specific implementations
pub trait StorageArithmetic<T: DataType>: Storage<T> {
    /// Element-wise addition
    fn add(&self, other: &Self) -> crate::Result<Self>;
    
    /// Element-wise subtraction
    fn sub(&self, other: &Self) -> crate::Result<Self>;
    
    /// Element-wise multiplication
    fn mul(&self, other: &Self) -> crate::Result<Self>;
    
    /// Element-wise division
    fn div(&self, other: &Self) -> crate::Result<Self>;
    
    /// Scalar addition
    fn add_scalar(&self, scalar: T) -> crate::Result<Self>;
    
    /// Scalar multiplication
    fn mul_scalar(&self, scalar: T) -> crate::Result<Self>;
}

/// Linear algebra operations trait
pub trait StorageLinearAlgebra<T: DataType>: Storage<T> {
    /// Matrix multiplication
    fn matmul(&self, other: &Self) -> crate::Result<Self>;
    
    /// Matrix-vector multiplication
    fn matvec(&self, vec: &[T]) -> crate::Result<Vec<T>>;
    
    /// Transpose operation
    fn transpose(&self) -> crate::Result<Self>;
}

/// Reduction operations trait
pub trait StorageReduction<T: DataType>: Storage<T> {
    /// Sum all elements
    fn sum(&self) -> T;
    
    /// Mean of all elements
    fn mean(&self) -> T
    where
        T: num_traits::Float;
    
    /// Maximum element
    fn max(&self) -> Option<T>
    where
        T: PartialOrd;
    
    /// Minimum element
    fn min(&self) -> Option<T>
    where
        T: PartialOrd;
    
    /// Sum along axis
    fn sum_axis(&self, axis: usize) -> crate::Result<Self>;
    
    /// Mean along axis
    fn mean_axis(&self, axis: usize) -> crate::Result<Self>
    where
        T: num_traits::Float;
}

/// Activation operations trait
pub trait StorageActivation<T: DataType>: Storage<T> {
    /// ReLU activation
    fn relu(&self) -> crate::Result<Self>
    where
        T: PartialOrd + num_traits::Zero;
    
    /// Sigmoid activation
    fn sigmoid(&self) -> crate::Result<Self>
    where
        T: num_traits::Float;
    
    /// Tanh activation
    fn tanh(&self) -> crate::Result<Self>
    where
        T: num_traits::Float;
    
    /// GELU activation
    fn gelu(&self) -> crate::Result<Self>
    where
        T: num_traits::Float;
    
    /// Softmax activation
    fn softmax(&self, axis: usize) -> crate::Result<Self>
    where
        T: num_traits::Float;
}

/// Unified storage operations trait combining all operation categories
pub trait StorageOps<T: DataType>:
    Storage<T>
    + StorageInfo<T>
    + StorageArithmetic<T>
    + StorageLinearAlgebra<T>
    + StorageReduction<T>
    + StorageActivation<T>
{
}

// Blanket implementation for any type that implements all required traits
impl<S, T> StorageOps<T> for S
where
    S: Storage<T>
        + StorageInfo<T>
        + StorageArithmetic<T>
        + StorageLinearAlgebra<T>
        + StorageReduction<T>
        + StorageActivation<T>,
    T: DataType,
{
}