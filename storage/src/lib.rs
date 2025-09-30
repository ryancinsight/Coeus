//! # Coeus Storage - Zero-Cost Tensor Storage Abstractions
//!
//! A dedicated crate providing generic, zero-cost abstractions for tensor storage formats,
//! enabling seamless interoperability between dense and sparse tensor representations.
//!
//! ## Architecture Overview
//!
//! The storage crate provides:
//! - **TensorStorage Trait**: Unified interface for all storage formats
//! - **DenseStorage**: Contiguous memory layout for optimal performance
//! - **SparseStorageCSR**: Compressed Sparse Row format for matrix operations
//! - **SparseStorageCOO**: Coordinate format for general sparsity patterns
//! - **Zero-Cost Polymorphism**: Compile-time dispatch maintains performance
//! - **Backend Agnostic**: Storage formats work with any backend implementation
//!
//! ## Key Design Principles
//!
//! - **Zero-Cost Abstractions**: Compile-time monomorphization, no runtime overhead
//! - **Memory Safety**: Zero unsafe code with proper bounds checking
//! - **Type Safety**: Compile-time guarantees for all storage operations
//! - **Extensibility**: Easy to add new storage formats
//! - **Interoperability**: Seamless conversion between storage formats
//!
//! ## Usage
//!
//! ```rust
//! use coeus_storage::{TensorStorage, DenseStorage, SparseStorageCSR, StorageDtype};
//!
//! // Create dense storage
//! let dense = DenseStorage::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
//!
//! // Create sparse CSR storage
//! let csr = SparseStorageCSR::new(
//!     vec![0, 2, 4],    // row pointers
//!     vec![0, 1, 1, 2], // column indices
//!     vec![1.0, 2.0, 3.0, 4.0], // values
//!     vec![2, 3]         // shape
//! );
//!
//! // Unified interface works with any storage format
//! fn process_storage<T: StorageDtype, S: TensorStorage<T>>(storage: &S) {
//!     println!("Shape: {:?}, Elements: {}", storage.shape(), storage.numel());
//! }
//! ```

// Re-export Dtype trait for storage compatibility
pub use coeus_dtype::Dtype as StorageDtype;
use num_traits::One;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Core storage trait for tensor data layouts
///
/// This trait provides a unified interface for all tensor storage formats,
/// enabling zero-cost polymorphism across dense and sparse representations.
pub trait TensorStorage<T: StorageDtype>: Clone + Send + Sync + fmt::Debug {
    /// Get the data as a slice (values for sparse, full data for dense)
    fn data(&self) -> &[T];

    /// Get mutable data access (values for sparse, full data for dense)
    fn data_mut(&mut self) -> &mut [T];

    /// Get the shape of the stored data
    fn shape(&self) -> &[usize];

    /// Get the number of elements
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if storage is contiguous in memory
    fn is_contiguous(&self) -> bool;

    /// Check if storage is sparse
    fn is_sparse(&self) -> bool {
        !self.is_contiguous()
    }

    /// Convert to dense representation (may allocate)
    fn to_dense(&self) -> Vec<T>;

    /// Create from dense representation (may fail for unsupported formats)
    fn from_dense(data: Vec<T>, shape: Vec<usize>) -> Result<Self, StorageError>
    where
        Self: Sized,
        T: PartialEq + Clone;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize {
        std::mem::size_of::<T>() * self.data().len() +
        std::mem::size_of::<usize>() * self.shape().len()
    }

    /// Validate storage integrity
    fn validate(&self) -> Result<(), StorageError>;
}

/// Dense storage for contiguous memory layout
///
/// Optimized for performance with contiguous memory access patterns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DenseStorage<T: StorageDtype> {
    /// Contiguous data array
    pub data: Vec<T>,
    /// Tensor shape
    pub shape: Vec<usize>,
}

impl<T: StorageDtype> DenseStorage<T> {
    /// Create dense storage from data and shape
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(data.len(), expected_len,
                   "Data length {} does not match shape product {}",
                   data.len(), expected_len);
        Self { data, shape }
    }

    /// Create zeros storage
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Default + Clone,
    {
        let len: usize = shape.iter().product();
        Self {
            data: vec![T::default(); len],
            shape,
        }
    }

    /// Create ones storage (requires T to implement One)
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: One + Clone,
    {
        let len: usize = shape.iter().product();
        Self {
            data: vec![T::one(); len],
            shape,
        }
    }

    /// Create storage filled with a specific value
    pub fn fill(shape: Vec<usize>, value: T) -> Self
    where
        T: Clone,
    {
        let len: usize = shape.iter().product();
        Self {
            data: vec![value; len],
            shape,
        }
    }
}

impl<T: StorageDtype> TensorStorage<T> for DenseStorage<T> {
    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn to_dense(&self) -> Vec<T> {
        self.data.clone()
    }

    fn from_dense(data: Vec<T>, shape: Vec<usize>) -> Result<Self, StorageError> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(StorageError::ShapeMismatch {
                data_len: data.len(),
                shape_product: expected_len,
            });
        }
        Ok(Self { data, shape })
    }

    fn validate(&self) -> Result<(), StorageError> {
        let expected_len: usize = self.shape.iter().product();
        if self.data.len() != expected_len {
            return Err(StorageError::ShapeMismatch {
                data_len: self.data.len(),
                shape_product: expected_len,
            });
        }
        Ok(())
    }
}

/// Compressed Sparse Row (CSR) storage for 2D sparse matrices
///
/// Optimized for sparse matrix-vector multiplication and row slicing operations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseStorageCSR<T: StorageDtype> {
    /// Row pointers (cumulative sum of row non-zeros)
    pub row_ptr: Vec<usize>,
    /// Column indices for non-zero elements
    pub col_indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<T>,
    /// Tensor shape [rows, cols]
    pub shape: Vec<usize>,
}

impl<T: StorageDtype> SparseStorageCSR<T> {
    /// Create CSR storage
    pub fn new(row_ptr: Vec<usize>, col_indices: Vec<usize>, values: Vec<T>, shape: Vec<usize>) -> Self {
        if shape.len() != 2 {
            panic!("CSR storage requires 2D shape, got {:?}", shape);
        }
        Self {
            row_ptr,
            col_indices,
            values,
            shape,
        }
    }

    /// Create CSR storage from dense matrix
    pub fn from_dense(dense: &[T], rows: usize, cols: usize) -> Self
    where
        T: PartialEq + Clone,
    {
        let zero = T::zero();
        let mut row_ptr = vec![0; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let val = &dense[idx];

                if *val != zero {
                    col_indices.push(col);
                    values.push(val.clone());
                }
            }
            row_ptr[row + 1] = values.len();
        }

        Self {
            row_ptr,
            col_indices,
            values,
            shape: vec![rows, cols],
        }
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// Get number of columns
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Get sparsity ratio (nnz / total_elements)
    pub fn sparsity(&self) -> f64 {
        self.nnz() as f64 / self.numel() as f64
    }
}

impl<T: StorageDtype> TensorStorage<T> for SparseStorageCSR<T> {
    fn data(&self) -> &[T] {
        &self.values
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn is_contiguous(&self) -> bool {
        false
    }

    fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![T::zero(); self.numel()];

        for row in 0..self.rows() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_indices[i];
                let idx = row * self.cols() + col;
                dense[idx] = self.values[i].clone();
            }
        }

        dense
    }

    fn from_dense(data: Vec<T>, shape: Vec<usize>) -> Result<Self, StorageError>
    where
        T: PartialEq + Clone,
    {
        if shape.len() != 2 {
            return Err(StorageError::UnsupportedShape {
                shape,
                format: "CSR".to_string(),
            });
        }

        Ok(Self::from_dense(&data, shape[0], shape[1]))
    }

    fn validate(&self) -> Result<(), StorageError> {
        if self.shape.len() != 2 {
            return Err(StorageError::UnsupportedShape {
                shape: self.shape.clone(),
                format: "CSR".to_string(),
            });
        }

        // Validate row pointers
        if self.row_ptr.len() != self.rows() + 1 {
            return Err(StorageError::InvalidStructure {
                message: format!("Row pointer length {} doesn't match rows + 1 ({})",
                               self.row_ptr.len(), self.rows() + 1),
            });
        }

        if self.row_ptr[0] != 0 {
            return Err(StorageError::InvalidStructure {
                message: "First row pointer must be 0".to_string(),
            });
        }

        for i in 1..self.row_ptr.len() {
            if self.row_ptr[i] < self.row_ptr[i - 1] {
                return Err(StorageError::InvalidStructure {
                    message: format!("Row pointer {} < {} at index {}", self.row_ptr[i], self.row_ptr[i - 1], i),
                });
            }
        }

        // Validate indices are in bounds
        for &col in &self.col_indices {
            if col >= self.cols() {
                return Err(StorageError::IndexOutOfBounds {
                    index: col,
                    dimension: self.cols(),
                });
            }
        }

        // Validate lengths match
        if self.col_indices.len() != self.values.len() {
            return Err(StorageError::InvalidStructure {
                message: format!("Column indices length {} != values length {}",
                               self.col_indices.len(), self.values.len()),
            });
        }

        Ok(())
    }
}

/// Coordinate (COO) storage for general sparse tensors
///
/// Flexible format suitable for arbitrary sparsity patterns and easy construction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseStorageCOO<T: StorageDtype> {
    /// Row indices for non-zero elements
    pub row_indices: Vec<usize>,
    /// Column indices for non-zero elements
    pub col_indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<T>,
    /// Tensor shape
    pub shape: Vec<usize>,
}

impl<T: StorageDtype> SparseStorageCOO<T> {
    /// Create COO storage
    pub fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, values: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(row_indices.len(), col_indices.len(),
                   "Row and column indices must have same length");
        assert_eq!(row_indices.len(), values.len(),
                   "Indices and values must have same length");

        Self {
            row_indices,
            col_indices,
            values,
            shape,
        }
    }

    /// Create COO storage from dense tensor
    pub fn from_dense(dense: &[T], shape: &[usize]) -> Self
    where
        T: PartialEq + Clone,
    {
        let zero = T::zero();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (flat_idx, val) in dense.iter().enumerate() {
            if *val != zero {
                // Convert flat index to multi-dimensional indices
                let mut coords = Vec::with_capacity(shape.len());
                let mut remaining = flat_idx;

                for &dim in shape.iter().rev() {
                    coords.push(remaining % dim);
                    remaining /= dim;
                }
                coords.reverse();

                // Add to COO format
                if coords.len() == 1 {
                    col_indices.push(coords[0]);
                } else if coords.len() == 2 {
                    row_indices.push(coords[0]);
                    col_indices.push(coords[1]);
                } else {
                    // For higher dimensions, flatten to 2D representation
                    let row = coords.iter().rev().fold(0, |acc, &x| acc * shape[shape.len() - 1] + x);
                    row_indices.push(row);
                    col_indices.push(0); // Placeholder
                }

                if coords.len() == 1 {
                    row_indices.push(0); // 1D tensor as single row
                }

                values.push(val.clone());
            }
        }

        Self {
            row_indices,
            col_indices,
            values,
            shape: shape.to_vec(),
        }
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio (nnz / total_elements)
    pub fn sparsity(&self) -> f64 {
        self.nnz() as f64 / self.numel() as f64
    }
}

impl<T: StorageDtype> TensorStorage<T> for SparseStorageCOO<T> {
    fn data(&self) -> &[T] {
        &self.values
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn is_contiguous(&self) -> bool {
        false
    }

    fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![T::zero(); self.numel()];

        for i in 0..self.nnz() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];

            // For multi-dimensional tensors, convert 2D coordinates back to flat index
            let flat_idx = if self.shape.len() == 1 {
                col
            } else if self.shape.len() == 2 {
                row * self.shape[1] + col
            } else {
                // Higher dimensional - simplified mapping
                row * self.shape.iter().product::<usize>() / self.shape[0] + col
            };

            if flat_idx < dense.len() {
                dense[flat_idx] = self.values[i].clone();
            }
        }

        dense
    }

    fn from_dense(data: Vec<T>, shape: Vec<usize>) -> Result<Self, StorageError>
    where
        T: PartialEq + Clone,
    {
        Ok(Self::from_dense(&data, &shape))
    }

    fn validate(&self) -> Result<(), StorageError> {
        if self.row_indices.len() != self.col_indices.len() {
            return Err(StorageError::InvalidStructure {
                message: format!("Row indices length {} != column indices length {}",
                               self.row_indices.len(), self.col_indices.len()),
            });
        }

        if self.row_indices.len() != self.values.len() {
            return Err(StorageError::InvalidStructure {
                message: format!("Indices length {} != values length {}",
                               self.row_indices.len(), self.values.len()),
            });
        }

        // Validate indices are in bounds
        for (&row, &col) in self.row_indices.iter().zip(&self.col_indices) {
            if row >= self.shape.get(0).copied().unwrap_or(0) {
                return Err(StorageError::IndexOutOfBounds {
                    index: row,
                    dimension: self.shape.get(0).copied().unwrap_or(0),
                });
            }

            if self.shape.len() > 1 && col >= self.shape.get(1).copied().unwrap_or(0) {
                return Err(StorageError::IndexOutOfBounds {
                    index: col,
                    dimension: self.shape.get(1).copied().unwrap_or(0),
                });
            }
        }

        Ok(())
    }
}

/// Storage-related errors
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StorageError {
    #[error("Shape mismatch: data length {data_len} != shape product {shape_product}")]
    ShapeMismatch { data_len: usize, shape_product: usize },

    #[error("Unsupported shape {shape:?} for storage format {format}")]
    UnsupportedShape { shape: Vec<usize>, format: String },

    #[error("Invalid storage structure: {message}")]
    InvalidStructure { message: String },

    #[error("Index {index} out of bounds for dimension {dimension}")]
    IndexOutOfBounds { index: usize, dimension: usize },

    #[error("Storage format conversion not supported")]
    ConversionUnsupported,
}

// Implement StorageDtype for common types used in tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_storage() {
        let storage = DenseStorage::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        assert_eq!(storage.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.shape(), &[3]);
        assert_eq!(storage.numel(), 3);
        assert!(storage.is_contiguous());
        assert_eq!(storage.memory_usage(), 3 * std::mem::size_of::<f32>() + std::mem::size_of::<usize>() * storage.shape().len());

        // Test validation
        assert!(storage.validate().is_ok());
    }

    #[test]
    fn test_dense_zeros() {
        let storage = DenseStorage::<f32>::zeros(vec![2, 3]);
        assert_eq!(storage.data(), &[0.0; 6]);
        assert_eq!(storage.shape(), &[2, 3]);
        assert_eq!(storage.numel(), 6);
    }

    #[test]
    fn test_dense_ones() {
        let storage = DenseStorage::<f32>::ones(vec![2, 2]);
        assert_eq!(storage.data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(storage.shape(), &[2, 2]);
    }

    #[test]
    fn test_dense_fill() {
        let storage = DenseStorage::<f32>::fill(vec![2, 2], 3.14);
        assert_eq!(storage.data(), &[3.14, 3.14, 3.14, 3.14]);
        assert_eq!(storage.shape(), &[2, 2]);
    }

    #[test]
    fn test_csr_storage() {
        let row_ptr = vec![0, 2, 4, 5];
        let col_indices = vec![0, 1, 1, 2, 2];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let storage = SparseStorageCSR::new(row_ptr, col_indices, values, vec![3, 3]);

        assert_eq!(storage.nnz(), 5);
        assert_eq!(storage.rows(), 3);
        assert_eq!(storage.cols(), 3);
        assert_eq!(storage.shape(), &[3, 3]);
        assert!(!storage.is_contiguous());
        assert_eq!(storage.sparsity(), 5.0 / 9.0);

        // Test validation
        assert!(storage.validate().is_ok());
    }

    #[test]
    fn test_csr_from_dense() {
        let dense = vec![1.0f32, 0.0, 2.0,
                        0.0, 3.0, 0.0,
                        4.0, 0.0, 0.0];
        let csr = SparseStorageCSR::from_dense(&dense, 3, 3);

        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 4]);
        assert_eq!(csr.col_indices, vec![0, 2, 1, 0]);
    }

    #[test]
    fn test_csr_to_dense() {
        let row_ptr = vec![0, 2, 3, 4];
        let col_indices = vec![0, 2, 1, 0];
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let csr = SparseStorageCSR::new(row_ptr, col_indices, values, vec![3, 3]);

        let dense = csr.to_dense();
        let expected = vec![1.0f32, 0.0, 2.0,
                           0.0, 3.0, 0.0,
                           4.0, 0.0, 0.0];
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_coo_storage() {
        let row_indices = vec![0, 0, 1, 2, 2];
        let col_indices = vec![0, 2, 1, 0, 2];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let storage = SparseStorageCOO::new(row_indices, col_indices, values, vec![3, 3]);

        assert_eq!(storage.nnz(), 5);
        assert_eq!(storage.shape(), &[3, 3]);
        assert!(!storage.is_contiguous());
        assert_eq!(storage.sparsity(), 5.0 / 9.0);

        // Test validation
        assert!(storage.validate().is_ok());
    }

    #[test]
    fn test_coo_from_dense() {
        let dense = vec![1.0f32, 0.0, 2.0,
                        0.0, 3.0, 0.0,
                        4.0, 0.0, 0.0];
        let coo = SparseStorageCOO::from_dense(&dense, &[3, 3]);

        assert_eq!(coo.nnz(), 4);
        // Note: COO format doesn't guarantee sorted order
        assert!(coo.validate().is_ok());
    }

    #[test]
    fn test_storage_errors() {
        // Test shape mismatch
        let result = DenseStorage::from_dense(vec![1.0, 2.0], vec![3]);
        assert!(matches!(result, Err(StorageError::ShapeMismatch { .. })));

        // Test invalid CSR structure
        let bad_csr = SparseStorageCSR::new(vec![1, 2, 3], vec![0, 1], vec![1.0], vec![2, 2]);
        assert!(matches!(bad_csr.validate(), Err(StorageError::InvalidStructure { .. })));
    }

    #[test]
    fn test_storage_conversion() {
        let dense = DenseStorage::from_vec(vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0], vec![2, 3]);

        // Convert to CSR
        let csr_result = SparseStorageCSR::from_dense(dense.data(), dense.shape()[0], dense.shape()[1]);
        assert_eq!(csr_result.nnz(), 3);

        // Convert back to dense
        let back_to_dense = csr_result.to_dense();
        assert_eq!(back_to_dense, dense.data());
    }
}

// Re-export tensor crate storage implementations for backward compatibility
// TODO: These will be removed once tensor crate is updated to use storage crate directly

/// Alternative dense storage implementation (from tensor crate)
/// This will be deprecated once tensor crate migrates to use storage crate
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDenseStorage<T: StorageDtype> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T: StorageDtype> TensorDenseStorage<T> {
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(data.len(), expected_len);
        Self { data, shape }
    }
}

impl<T: StorageDtype> TensorStorage<T> for TensorDenseStorage<T> {
    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn to_dense(&self) -> Vec<T> {
        self.data.clone()
    }

    fn from_dense(data: Vec<T>, shape: Vec<usize>) -> Result<Self, StorageError>
    where
        Self: Sized,
        T: PartialEq + Clone,
    {
        Ok(Self::from_vec(data, shape))
    }

    fn validate(&self) -> Result<(), StorageError> {
        let expected_len: usize = self.shape.iter().product();
        if self.data.len() != expected_len {
            return Err(StorageError::ShapeMismatch {
                data_len: self.data.len(),
                shape_product: expected_len,
            });
        }
        Ok(())
    }
}

/// Alternative sparse CSR storage implementation (from tensor crate)
/// This will be deprecated once tensor crate migrates to use storage crate
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSparseStorageCSR<T: StorageDtype> {
    pub row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T: StorageDtype> TensorSparseStorageCSR<T> {
    pub fn new(row_ptr: Vec<usize>, col_indices: Vec<usize>, values: Vec<T>, shape: Vec<usize>) -> Self {
        Self { row_ptr, col_indices, values, shape }
    }
}

impl<T: StorageDtype> TensorStorage<T> for TensorSparseStorageCSR<T> {
    fn data(&self) -> &[T] {
        &self.values
    }

    fn data_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn is_contiguous(&self) -> bool {
        false
    }

    fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![T::zero(); self.numel()];

        for row in 0..self.shape[0] {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_indices[i];
                let idx = row * self.shape[1] + col;
                dense[idx] = self.values[i].clone();
            }
        }

        dense
    }

    fn from_dense(data: Vec<T>, shape: Vec<usize>) -> Result<Self, StorageError>
    where
        Self: Sized,
        T: PartialEq + Clone,
    {
        let rows = shape[0];
        let cols = shape[1];
        let mut row_ptr = vec![0; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let val = &data[idx];

                if !(*val == T::zero()) {
                    col_indices.push(col);
                    values.push(val.clone());
                }
            }
            row_ptr[row + 1] = values.len();
        }

        Ok(Self { row_ptr, col_indices, values, shape })
    }

    fn validate(&self) -> Result<(), StorageError> {
        if self.shape.len() != 2 {
            return Err(StorageError::UnsupportedShape {
                shape: self.shape.clone(),
                format: "CSR".to_string(),
            });
        }

        // Validate row pointers
        if self.row_ptr.len() != self.shape[0] + 1 {
            return Err(StorageError::InvalidStructure {
                message: format!("Row pointer length {} != expected {} for CSR format",
                               self.row_ptr.len(), self.shape[0] + 1),
            });
        }

        // Check that row pointers are non-decreasing
        for i in 1..self.row_ptr.len() {
            if self.row_ptr[i] < self.row_ptr[i - 1] {
                return Err(StorageError::InvalidStructure {
                    message: format!("Row pointer at index {} is less than previous", i),
                });
            }
        }

        // Check that column indices are within bounds
        for &col in &self.col_indices {
            if col >= self.shape[1] {
                return Err(StorageError::InvalidStructure {
                    message: format!("Column index {} out of bounds for shape {:?}", col, self.shape),
                });
            }
        }

        // Check that values and column indices have the same length
        if self.col_indices.len() != self.values.len() {
            return Err(StorageError::InvalidStructure {
                message: format!("Column indices length {} != values length {}",
                               self.col_indices.len(), self.values.len()),
            });
        }

        Ok(())
    }
}

