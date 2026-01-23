//! Enhanced CSC (Compressed Sparse Column) storage implementation
//!
//! This module provides the enhanced CSC sparse storage implementation optimized
//! for column-based operations and matrix-vector multiplication with zero-cost abstractions.

use crate::{AsAny, DataType, Result, Shape, Storage, StorageError, StorageFromVec, StorageToDense, DenseStorage, StorageFormat};
use alloc::{vec, vec::Vec};
use core::fmt;

/// Enhanced CSC (Compressed Sparse Column) storage implementation
///
/// Memory layout:
/// - `data`: Non-zero values (length = nnz)
/// - `indices`: Row indices for each non-zero value (length = nnz)
/// - `indptr`: Column pointers (length = cols + 1)
///
/// Efficient for:
/// - Column-based operations
/// - Matrix-vector multiplication (A^T x)
/// - Sparse matrix-matrix multiplication
#[derive(Clone, Debug, PartialEq)]
pub struct CscStorage<T: DataType> {
    data: Vec<T>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    shape: Shape,
}

impl<T: DataType> CscStorage<T> {
    /// Create new CSC storage from components
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
                reason: "CSC storage requires 2D shape",
            });
        }
        
        let rows = shape[0];
        let cols = shape[1];
        
        if indptr.len() != cols + 1 {
            return Err(StorageError::ShapeMismatch {
                expected: cols + 1,
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
        for &row_idx in &indices {
            if row_idx >= rows {
                return Err(StorageError::IndexOutOfBounds {
                    index: row_idx,
                    bound: rows,
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
    
    /// Create empty CSC storage with given shape
    pub fn empty(shape: &[usize]) -> Result<Self> {
        if shape.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "CSC storage requires 2D shape",
            });
        }
        
        let cols = shape[1];
        Self::new(Vec::new(), Vec::new(), vec![0; cols + 1], shape)
    }
    
    /// Create identity matrix in CSC format
    pub fn eye(size: usize) -> Result<Self>
    where
        T: num_traits::One,
    {
        let data = vec![T::one(); size];
        let indices: Vec<usize> = (0..size).collect();
        let indptr: Vec<usize> = (0..=size).collect();
        
        Self::new(data, indices, indptr, &[size, size])
    }
    
    /// Create CSC storage from dense matrix, keeping only non-zero elements
    pub fn from_dense(dense: &DenseStorage<T>) -> Result<Self>
    where
        T: num_traits::Zero + PartialEq,
    {
        let shape_dims = dense.shape().dims();
        if shape_dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "Can only convert 2D dense storage to CSC",
            });
        }
        
        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let dense_data = dense.as_slice();
        
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        
        for col in 0..cols {
            for row in 0..rows {
                let idx = row * cols + col;
                let value = dense_data[idx];
                if value != T::zero() {
                    data.push(value);
                    indices.push(row);
                }
            }
            indptr.push(data.len());
        }
        
        Self::new(data, indices, indptr, shape_dims)
    }
    
    /// Convert CSC storage to dense format
    pub fn to_dense(&self) -> Result<DenseStorage<T>>
    where
        T: num_traits::Zero,
    {
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];
        let mut dense_data = vec![T::zero(); rows * cols];
        
        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];
            
            for idx in start..end {
                let row = self.indices[idx];
                let value = self.data[idx];
                dense_data[row * cols + col] = value;
            }
        }
        
        DenseStorage::from_vec(dense_data, self.shape.dims())
    }
    
    /// Convert to CSR format
    pub fn to_csr(&self) -> Result<super::super::csr::CsrStorage<T>> {
        let (rows, cols) = self.dims();
        
        // Count non-zeros per row
        let mut row_counts = vec![0; rows];
        for &row in &self.indices {
            row_counts[row] += 1;
        }
        
        // Build row pointers
        let mut indptr = vec![0; rows + 1];
        for row in 0..rows {
            indptr[row + 1] = indptr[row] + row_counts[row];
        }
        
        // Fill CSR data
        let mut data = vec![T::default(); self.data.len()];
        let mut indices = vec![0; self.indices.len()];
        let mut row_positions = indptr[..rows].to_vec();
        
        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];
            
            for idx in start..end {
                let row = self.indices[idx];
                let value = self.data[idx];
                let pos = row_positions[row];
                
                data[pos] = value;
                indices[pos] = col;
                row_positions[row] += 1;
            }
        }
        
        super::super::csr::CsrStorage::new(data, indices, indptr, &[rows, cols])
    }
    
    /// Convert to COO format
    pub fn to_coo(&self) -> Result<super::super::coo::CooStorage<T>> {
        let (_rows, cols) = self.dims();
        let mut data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        
        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];
            
            for idx in start..end {
                data.push(self.data[idx]);
                row_indices.push(self.indices[idx]);
                col_indices.push(col);
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
    
    /// Get reference to row indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
    
    /// Get reference to column pointers
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
        
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        
        // Binary search for row index
        for idx in start..end {
            match self.indices[idx].cmp(&row) {
                core::cmp::Ordering::Equal => return self.data[idx],
                core::cmp::Ordering::Greater => break,
                core::cmp::Ordering::Less => continue,
            }
        }
        
        T::zero()
    }
    
    /// Matrix-vector multiplication: y = A^T * x (transpose multiplication)
    pub fn matvec_transpose(&self, x: &[T], y: &mut [T]) -> Result<()>
    where
        T: num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy,
    {
        let (rows, cols) = self.dims();
        
        if x.len() != rows {
            return Err(StorageError::ShapeMismatch {
                expected: rows,
                actual: x.len(),
            });
        }
        
        if y.len() != cols {
            return Err(StorageError::ShapeMismatch {
                expected: cols,
                actual: y.len(),
            });
        }
        
        // Initialize output to zero
        for y_elem in y.iter_mut() {
            *y_elem = T::zero();
        }
        
        // Compute y = A^T * x
        for col in 0..cols {
            let start = self.indptr[col];
            let end = self.indptr[col + 1];
            
            for idx in start..end {
                let row = self.indices[idx];
                let value = self.data[idx];
                y[col] = y[col] + value * x[row];
            }
        }
        
        Ok(())
    }
    
    /// Transpose the CSC matrix (returns CSR format)
    pub fn transpose(&self) -> Result<super::super::csr::CsrStorage<T>> {
        self.to_csr()
    }
    
    /// Element-wise addition with another CSC matrix
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
        
        let (_rows, cols) = self.dims();
        let mut result_data = Vec::new();
        let mut result_indices = Vec::new();
        let mut result_indptr = vec![0];
        
        for col in 0..cols {
            let a_start = self.indptr[col];
            let a_end = self.indptr[col + 1];
            let b_start = other.indptr[col];
            let b_end = other.indptr[col + 1];
            
            let mut a_idx = a_start;
            let mut b_idx = b_start;
            
            while a_idx < a_end || b_idx < b_end {
                let (row, sum) = if a_idx < a_end && b_idx < b_end {
                    let a_row = self.indices[a_idx];
                    let b_row = other.indices[b_idx];
                    
                    match a_row.cmp(&b_row) {
                        core::cmp::Ordering::Less => {
                            let val = self.data[a_idx];
                            a_idx += 1;
                            (a_row, val)
                        }
                        core::cmp::Ordering::Greater => {
                            let val = other.data[b_idx];
                            b_idx += 1;
                            (b_row, val)
                        }
                        core::cmp::Ordering::Equal => {
                            let val = self.data[a_idx] + other.data[b_idx];
                            a_idx += 1;
                            b_idx += 1;
                            (a_row, val)
                        }
                    }
                } else if a_idx < a_end {
                    let val = self.data[a_idx];
                    let row = self.indices[a_idx];
                    a_idx += 1;
                    (row, val)
                } else {
                    let val = other.data[b_idx];
                    let row = other.indices[b_idx];
                    b_idx += 1;
                    (row, val)
                };
                
                // Only store non-zero results
                if sum != T::zero() {
                    result_data.push(sum);
                    result_indices.push(row);
                }
            }
            
            result_indptr.push(result_data.len());
        }
        
        Self::new(result_data, result_indices, result_indptr, self.shape.dims())
    }
}

impl<T: DataType> Storage<T> for CscStorage<T> {
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
        StorageFormat::Csc
    }
    
    fn strides(&self) -> &[usize] {
        // CSC doesn't have regular strides like dense storage
        &[]
    }
    
    fn is_contiguous(&self) -> bool {
        // CSC is not contiguous in the dense sense
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
                reason: "CSC storage requires 2D shape",
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
            let indices: Vec<usize> = (0..rows).cycle().take(total_elements).collect();
            let indptr: Vec<usize> = (0..=cols).map(|i| i * rows).collect();
            
            Self::new(data, indices, indptr, dims)
        }
    }
}

impl<T: DataType> StorageToDense<T> for CscStorage<T>
where
    T: num_traits::Zero,
{
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        self.to_dense()
    }
}

impl<T: DataType> StorageFromVec<T> for CscStorage<T>
where
    T: num_traits::Zero + PartialEq,
{
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        // Create dense storage first, then convert to CSC
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

impl<T: DataType> AsAny for CscStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> fmt::Display for CscStorage<T>
where
    T: fmt::Display + num_traits::Zero,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.dims();
        writeln!(f, "CscStorage {}x{} (nnz={})", rows, cols, self.nnz())?;
        
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