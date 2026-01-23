//! Enhanced CSR (Compressed Sparse Row) storage implementation
//!
//! This module provides the enhanced CSR sparse storage implementation optimized
//! for row-based operations and matrix-vector multiplication with zero-cost abstractions.

use crate::{AsAny, DataType, Result, Shape, Storage, StorageError, StorageFromVec, StorageToDense, DenseStorage, StorageFormat};
use alloc::{vec, vec::Vec};
use core::fmt;

/// Enhanced CSR (Compressed Sparse Row) storage implementation
///
/// Memory layout:
/// - `data`: Non-zero values (length = nnz)
/// - `indices`: Column indices for each non-zero value (length = nnz)
/// - `indptr`: Row pointers (length = rows + 1)
///
/// Efficient for:
/// - Row-based operations
/// - Matrix-vector multiplication (Ax)
/// - Sparse matrix-matrix multiplication
#[derive(Clone, Debug, PartialEq)]
pub struct CsrStorage<T: DataType> {
    data: Vec<T>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    shape: Shape,
}

impl<T: DataType> CsrStorage<T> {
    /// Create new CSR storage from components
    pub fn new(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
    ) -> Result<Self> {
        // Validate input
        if data.len() != indices.len() {
            return Err(StorageError::ShapeMismatch {
                expected: data.len(),
                actual: indices.len(),
            });
        }
        
        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSR storage requires 2D shape",
            });
        }
        
        let rows = shape[0];
        let cols = shape[1];
        
        if indptr.len() != rows + 1 {
            return Err(StorageError::ShapeMismatch {
                expected: rows + 1,
                actual: indptr.len(),
            });
        }
        
        // Validate indptr is non-decreasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(StorageError::InvalidShape {
                    reason: "indptr must be non-decreasing",
                });
            }
        }
        
        // Validate indices are within bounds
        for &col_idx in &indices {
            if col_idx >= cols {
                return Err(StorageError::IndexOutOfBounds {
                    index: col_idx,
                    bound: cols,
                });
            }
        }
        
        Ok(Self {
            data,
            indices,
            indptr,
            shape: Shape::new(shape)?,
        })
    }
    
    /// Create empty CSR storage with given shape
    pub fn empty(shape: &[usize]) -> Result<Self> {
        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSR storage requires 2D shape",
            });
        }
        
        let rows = shape[0];
        Self::new(Vec::new(), Vec::new(), vec![0; rows + 1], shape)
    }
    
    /// Create identity matrix in CSR format
    pub fn eye(size: usize) -> Result<Self>
    where
        T: num_traits::One,
    {
        let data = vec![T::one(); size];
        let indices: Vec<usize> = (0..size).collect();
        let indptr: Vec<usize> = (0..=size).collect();
        
        Self::new(data, indices, indptr, &[size, size])
    }
    
    /// Create CSR storage from dense matrix, keeping only non-zero elements
    pub fn from_dense(dense: &DenseStorage<T>) -> Result<Self>
    where
        T: num_traits::Zero + PartialEq,
    {
        let shape_dims = dense.shape().dims();
        if shape_dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Can only convert 2D dense storage to CSR",
            });
        }
        
        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let dense_data = dense.as_slice();
        
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let value = dense_data[idx];
                if value != T::zero() {
                    data.push(value);
                    indices.push(col);
                }
            }
            indptr.push(data.len());
        }
        
        Self::new(data, indices, indptr, shape_dims)
    }
    
    /// Convert CSR storage to dense format
    pub fn to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero,
    {
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];
        let mut dense_data = vec![T::zero(); rows * cols];
        
        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            for idx in start..end {
                let col = self.indices[idx];
                let value = self.data[idx];
                dense_data[row * cols + col] = value;
            }
        }
        
        DenseStorage::from_vec(dense_data, self.shape.dims())
    }
    
    /// Convert to CSC format
    pub fn to_csc(&self) -> Result<super::super::csc::CscStorage<T>> {
        let (rows, cols) = self.dims();
        
        // Count non-zeros per column
        let mut col_counts = vec![0; cols];
        for &col in &self.indices {
            col_counts[col] += 1;
        }
        
        // Build column pointers
        let mut indptr = vec![0; cols + 1];
        for col in 0..cols {
            indptr[col + 1] = indptr[col] + col_counts[col];
        }
        
        // Fill CSC data
        let mut data = vec![T::default(); self.data.len()];
        let mut indices = vec![0; self.indices.len()];
        let mut col_positions = indptr[..cols].to_vec();
        
        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            for idx in start..end {
                let col = self.indices[idx];
                let value = self.data[idx];
                let pos = col_positions[col];
                
                data[pos] = value;
                indices[pos] = row;
                col_positions[col] += 1;
            }
        }
        
        super::super::csc::CscStorage::new(data, indices, indptr, &[rows, cols])
    }
    
    /// Convert to COO format
    pub fn to_coo(&self) -> Result<super::super::coo::CooStorage<T>> {
        let (rows, _cols) = self.dims();
        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        
        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            for idx in start..end {
                data.push(self.data[idx]);
                row_indices.push(row);
                col_indices.push(self.indices[idx]);
            }
        }
        
        super::super::coo::CooStorage::new(data, row_indices, col_indices, self.shape.dims())
    }
    
    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
    
    /// Get matrix dimensions
    pub fn dims(&self) -> (usize, usize) {
        let shape_dims = self.shape.dims();
        (shape_dims[0], shape_dims[1])
    }
    
    /// Get reference to non-zero values
    pub fn data(&self) -> &[T] {
        &self.data
    }
    
    /// Get reference to column indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
    
    /// Get reference to row pointers
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }
    
    /// Get value at (row, col), returns zero if not stored
    pub fn get(&self, row: usize, col: usize) -> T
    where
        T: num_traits::Zero,
    {
        let (rows, cols) = self.dims();
        if row >= rows || col >= cols {
            return T::zero();
        }
        
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        
        // Binary search for column index
        for idx in start..end {
            match self.indices[idx].cmp(&col) {
                core::cmp::Ordering::Equal => return self.data[idx],
                core::cmp::Ordering::Greater => break,
                core::cmp::Ordering::Less => continue,
            }
        }
        
        T::zero()
    }
    
    /// Matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &[T], y: &mut [T]) -> Result<()>
    where
        T: num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy,
    {
        let (rows, cols) = self.dims();
        
        if x.len() != cols {
            return Err(StorageError::ShapeMismatch {
                expected: cols,
                actual: x.len(),
            });
        }
        
        if y.len() != rows {
            return Err(StorageError::ShapeMismatch {
                expected: rows,
                actual: y.len(),
            });
        }
        
        // Initialize output to zero
        for y_elem in y.iter_mut() {
            *y_elem = T::zero();
        }
        
        // Compute y = A * x
        for row in 0..rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            for idx in start..end {
                let col = self.indices[idx];
                let value = self.data[idx];
                y[row] = y[row] + value * x[col];
            }
        }
        
        Ok(())
    }
    
    /// Transpose the CSR matrix (returns CSC format)
    pub fn transpose(&self) -> Result<super::super::csc::CscStorage<T>> {
        self.to_csc()
    }
    
    /// Element-wise addition with another CSR matrix
    pub fn add(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Zero + core::ops::Add<Output = T> + Copy + PartialEq,
    {
        if self.shape != other.shape {
            return Err(StorageError::BroadcastError {
                shape_a: self.shape.dims().to_vec(),
                shape_b: other.shape.dims().to_vec(),
                dimension: 0,
            });
        }
        
        let (rows, _cols) = self.dims();
        let mut result_data = Vec::new();
        let mut result_indices = Vec::new();
        let mut result_indptr = vec![0];
        
        for row in 0..rows {
            let a_start = self.indptr[row];
            let a_end = self.indptr[row + 1];
            let b_start = other.indptr[row];
            let b_end = other.indptr[row + 1];
            
            let mut a_idx = a_start;
            let mut b_idx = b_start;
            
            while a_idx < a_end || b_idx < b_end {
                let (col, sum) = if a_idx < a_end && b_idx < b_end {
                    let a_col = self.indices[a_idx];
                    let b_col = other.indices[b_idx];
                    
                    match a_col.cmp(&b_col) {
                        core::cmp::Ordering::Less => {
                            let val = self.data[a_idx];
                            a_idx += 1;
                            (a_col, val)
                        }
                        core::cmp::Ordering::Greater => {
                            let val = other.data[b_idx];
                            b_idx += 1;
                            (b_col, val)
                        }
                        core::cmp::Ordering::Equal => {
                            let val = self.data[a_idx] + other.data[b_idx];
                            a_idx += 1;
                            b_idx += 1;
                            (a_col, val)
                        }
                    }
                } else if a_idx < a_end {
                    let val = self.data[a_idx];
                    let col = self.indices[a_idx];
                    a_idx += 1;
                    (col, val)
                } else {
                    let val = other.data[b_idx];
                    let col = other.indices[b_idx];
                    b_idx += 1;
                    (col, val)
                };
                
                // Only store non-zero results
                if sum != T::zero() {
                    result_data.push(sum);
                    result_indices.push(col);
                }
            }
            
            result_indptr.push(result_data.len());
        }
        
        Self::new(result_data, result_indices, result_indptr, self.shape.dims())
    }
}

impl<T: DataType> Storage<T> for CsrStorage<T> {
    fn len(&self) -> usize {
        self.shape.dims().iter().product()
    }
    
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    fn shape(&self) -> &Shape {
        &self.shape
    }
    
    fn format(&self) -> StorageFormat {
        StorageFormat::Csr
    }
    
    fn strides(&self) -> &[usize] {
        // CSR doesn't have regular strides like dense storage
        &[]
    }
    
    fn is_contiguous(&self) -> bool {
        // CSR is not contiguous in the dense sense
        false
    }
    
    fn as_slice(&self) -> &[T] {
        // Return the non-zero values
        &self.data
    }
    
    fn as_mut_slice(&mut self) -> &mut [T] {
        // Return the non-zero values
        &mut self.data
    }
    
    fn as_storage_ref(&self) -> &Self {
        self
    }
    
    fn full(dims: &[usize], value: T) -> Result<Self>
    where
        Self: Sized,
    {
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSR storage requires 2D shape",
            });
        }
        
        let rows = dims[0];
        let cols = dims[1];
        let total_elements = rows * cols;
        
        if value == T::zero() {
            // All zeros - create empty sparse matrix
            Self::empty(dims)
        } else {
            // All non-zeros - create dense-like sparse matrix
            let data = vec![value; total_elements];
            let indices: Vec<usize> = (0..cols).cycle().take(total_elements).collect();
            let indptr: Vec<usize> = (0..=rows).map(|i| i * cols).collect();
            
            Self::new(data, indices, indptr, dims)
        }
    }
}

impl<T: DataType> StorageToDense<T> for CsrStorage<T>
where
    T: num_traits::Zero,
{
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        self.to_dense()
    }
}

impl<T: DataType> StorageFromVec<T> for CsrStorage<T>
where
    T: num_traits::Zero + PartialEq,
{
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        // Create dense storage first, then convert to CSR
        let dense = DenseStorage::from_vec(data, shape)?;
        Self::from_dense(&dense)
    }
    
    fn zeros(dims: &[usize]) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Zero,
    {
        Self::empty(dims)
    }
    
    fn ones(dims: &[usize]) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::One,
    {
        Self::full(dims, T::one())
    }
}

impl<T: DataType> AsAny for CsrStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> fmt::Display for CsrStorage<T>
where
    T: fmt::Display + num_traits::Zero,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.dims();
        writeln!(f, "CsrStorage {}x{} (nnz={})", rows, cols, self.nnz())?;
        
        // Display as dense matrix for readability (up to reasonable size)
        if rows <= 10 && cols <= 10 {
            for row in 0..rows {
                write!(f, "[")?;
                for col in 0..cols {
                    if col > 0 { write!(f, ", ")?; }
                    write!(f, "{}", self.get(row, col))?;
                }
                writeln!(f, "]")?;
            }
        } else {
            writeln!(f, "Matrix too large to display")?;
        }
        
        Ok(())
    }
}