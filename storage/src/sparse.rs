//! Sparse storage implementations
//!
//! Provides memory-efficient storage for tensors with many zero elements.
//! Supports CSR, CSC, and COO sparse matrix formats.

use crate::sparse_arithmetic::SparseMatMul;
use crate::{AsAny, DataType, Result, Shape, Storage, StorageError};
use alloc::{vec, vec::Vec};

/// Enum representing different sparse matrix formats
#[derive(Debug, Clone, PartialEq)]
pub enum SparseFormat {
    /// Compressed Sparse Row: efficient for row-based operations
    Csr,
    /// Compressed Sparse Column: efficient for column-based operations
    Csc,
    /// Coordinate format: flexible for construction and conversion
    Coo,
}

/// Sparse storage using compressed sparse row (CSR) format.
///
/// CSR format stores only non-zero elements with row pointers and column indices.
/// Memory usage: O(nnz) where nnz is number of non-zero elements.
///
/// # Format Layout
/// - `data`: Non-zero values in row-major order
/// - `indices`: Column indices for each non-zero element
/// - `indptr`: Row pointers (start index of each row in data/indices)
///
/// # Examples
/// ```
/// use storage::CsrStorage;
/// use dtype::float::Float32;
///
/// // 3x3 matrix: [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
/// // Non-zeros: (0,0)=1, (0,2)=2, (1,1)=3, (2,0)=4, (2,2)=5
/// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
///                 Float32::new(4.0), Float32::new(5.0)];
/// let indices = vec![0, 2, 1, 0, 2];  // column indices
/// let indptr = vec![0, 2, 3, 5];       // row pointers
/// let storage = CsrStorage::new(data, indices, indptr, &[3, 3]).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CsrStorage<T: DataType> {
    /// Non-zero values in row-major order
    data: Vec<T>,
    /// Column indices for each non-zero element
    indices: Vec<usize>,
    /// Row pointers (start index of each row in data/indices)
    indptr: Vec<usize>,
    /// Matrix shape [rows, cols]
    shape: Shape,
}

/// Sparse storage using compressed sparse column (CSC) format.
///
/// CSC format stores only non-zero elements with column pointers and row indices.
/// Memory usage: O(nnz) where nnz is number of non-zero elements.
///
/// # Format Layout
/// - `data`: Non-zero values in column-major order
/// - `indices`: Row indices for each non-zero element
/// - `indptr`: Column pointers (start index of each column in data/indices)
#[derive(Debug, Clone, PartialEq)]
pub struct CscStorage<T: DataType> {
    /// Non-zero values in column-major order
    data: Vec<T>,
    /// Row indices for each non-zero element
    indices: Vec<usize>,
    /// Column pointers (start index of each column in data/indices)
    indptr: Vec<usize>,
    /// Matrix shape [rows, cols]
    shape: Shape,
}

/// Sparse storage using coordinate (COO) format.
///
/// COO format stores coordinates (row, col) for each non-zero element.
/// Most flexible format for construction and conversion between formats.
///
/// # Format Layout
/// - `data`: Non-zero values
/// - `row_indices`: Row indices for each non-zero element
/// - `col_indices`: Column indices for each non-zero element
#[derive(Debug, Clone, PartialEq)]
pub struct CooStorage<T: DataType> {
    /// Non-zero values
    data: Vec<T>,
    /// Row indices for each non-zero element
    row_indices: Vec<usize>,
    /// Column indices for each non-zero element
    col_indices: Vec<usize>,
    /// Matrix shape [rows, cols]
    shape: Shape,
}

impl<T: DataType> CsrStorage<T> {
    /// Creates CSR storage from components.
    ///
    /// # Arguments
    /// * `data` - Non-zero values in row-major order
    /// * `indices` - Column indices for each non-zero element
    /// * `indptr` - Row pointers (must have `shape[0]` + 1 elements)
    /// * `shape` - Matrix shape [rows, cols]
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or indices out of bounds.
    pub fn new(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
    ) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.ndim() != 2 {
            return Err(StorageError::ShapeMismatch {
                expected: 2,
                actual: shape.ndim(),
            });
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Validate indptr length
        if indptr.len() != rows + 1 {
            return Err(StorageError::ShapeMismatch {
                expected: rows + 1,
                actual: indptr.len(),
            });
        }

        // Validate data and indices have same length
        if data.len() != indices.len() {
            return Err(StorageError::ShapeMismatch {
                expected: data.len(),
                actual: indices.len(),
            });
        }

        // Validate indices are within bounds
        for &col_idx in &indices {
            if col_idx >= cols {
                return Err(StorageError::ShapeMismatch {
                    expected: cols,
                    actual: col_idx,
                });
            }
        }

        // Validate indptr is non-decreasing and within bounds
        let mut prev = 0;
        for &ptr in &indptr {
            if ptr < prev || ptr > data.len() {
                return Err(StorageError::ShapeMismatch {
                    expected: data.len(),
                    actual: ptr,
                });
            }
            prev = ptr;
        }

        Ok(Self {
            data,
            indices,
            indptr,
            shape,
        })
    }

    /// Returns the number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the sparsity ratio (fraction of zero elements, 0.0 = dense, 1.0 = all zeros).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f64 {
        let total = self.shape.size() as f64;
        let nnz = self.nnz() as f64;
        if total == 0.0 {
            0.0
        } else {
            1.0 - (nnz / total)
        }
    }
}

impl<T: DataType> CscStorage<T> {
    /// Creates CSC storage from components.
    ///
    /// # Arguments
    /// * `data` - Non-zero values in column-major order
    /// * `indices` - Row indices for each non-zero element
    /// * `indptr` - Column pointers (must have `shape[1]` + 1 elements)
    /// * `shape` - Matrix shape [rows, cols]
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or indices out of bounds.
    pub fn new(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
    ) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.ndim() != 2 {
            return Err(StorageError::ShapeMismatch {
                expected: 2,
                actual: shape.ndim(),
            });
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Validate indptr length
        if indptr.len() != cols + 1 {
            return Err(StorageError::ShapeMismatch {
                expected: cols + 1,
                actual: indptr.len(),
            });
        }

        // Validate data and indices have same length
        if data.len() != indices.len() {
            return Err(StorageError::ShapeMismatch {
                expected: data.len(),
                actual: indices.len(),
            });
        }

        // Validate indices are within bounds
        for &row_idx in &indices {
            if row_idx >= rows {
                return Err(StorageError::ShapeMismatch {
                    expected: rows,
                    actual: row_idx,
                });
            }
        }

        // Validate indptr is non-decreasing and within bounds
        let mut prev = 0;
        for &ptr in &indptr {
            if ptr < prev || ptr > data.len() {
                return Err(StorageError::ShapeMismatch {
                    expected: data.len(),
                    actual: ptr,
                });
            }
            prev = ptr;
        }

        Ok(Self {
            data,
            indices,
            indptr,
            shape,
        })
    }

    /// Returns the number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the sparsity ratio (fraction of zero elements, 0.0 = dense, 1.0 = all zeros).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f64 {
        let total = self.shape.size() as f64;
        let nnz = self.nnz() as f64;
        if total == 0.0 {
            0.0
        } else {
            1.0 - (nnz / total)
        }
    }
}

impl<T: DataType> CooStorage<T> {
    /// Creates COO storage from components.
    ///
    /// # Arguments
    /// * `data` - Non-zero values
    /// * `row_indices` - Row indices for each non-zero element
    /// * `col_indices` - Column indices for each non-zero element
    /// * `shape` - Matrix shape [rows, cols]
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or indices out of bounds.
    pub fn new(
        data: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: &[usize],
    ) -> Result<Self> {
        let shape = Shape::new(shape)?;
        if shape.ndim() != 2 {
            return Err(StorageError::ShapeMismatch {
                expected: 2,
                actual: shape.ndim(),
            });
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        // Validate all vectors have same length
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(StorageError::ShapeMismatch {
                expected: data.len(),
                actual: row_indices.len().max(col_indices.len()),
            });
        }

        // Validate indices are within bounds
        for (&row, &col) in row_indices.iter().zip(&col_indices) {
            if row >= rows || col >= cols {
                return Err(StorageError::ShapeMismatch {
                    expected: rows.max(cols),
                    actual: row.max(col),
                });
            }
        }

        Ok(Self {
            data,
            row_indices,
            col_indices,
            shape,
        })
    }

    /// Returns the number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the sparsity ratio (fraction of zero elements, 0.0 = dense, 1.0 = all zeros).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f64 {
        let total = self.shape.size() as f64;
        let nnz = self.nnz() as f64;
        if total == 0.0 {
            0.0
        } else {
            1.0 - (nnz / total)
        }
    }

    /// Sorts the COO storage by row, then by column for efficient conversion.
    pub fn sort(&mut self) {
        // Create indices for sorting
        let mut indices: Vec<usize> = (0..self.data.len()).collect();

        // Sort by row, then by column
        indices.sort_by(|&a, &b| {
            let row_a = self.row_indices[a];
            let row_b = self.row_indices[b];
            let col_a = self.col_indices[a];
            let col_b = self.col_indices[b];

            row_a.cmp(&row_b).then(col_a.cmp(&col_b))
        });

        // Reorder all arrays according to sorted indices
        let mut new_data = Vec::with_capacity(self.data.len());
        let mut new_row_indices = Vec::with_capacity(self.row_indices.len());
        let mut new_col_indices = Vec::with_capacity(self.col_indices.len());

        for &idx in &indices {
            new_data.push(self.data[idx]);
            new_row_indices.push(self.row_indices[idx]);
            new_col_indices.push(self.col_indices[idx]);
        }

        self.data = new_data;
        self.row_indices = new_row_indices;
        self.col_indices = new_col_indices;
    }

    /// Returns the row indices as a slice.
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Returns the column indices as a slice.
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Converts COO to CSR format.
    #[must_use]
    pub fn to_csr(&self) -> CsrStorage<T>
    where
        T: Clone,
    {
        let rows = self.shape.dims()[0];
        let nnz = self.nnz();

        // Sort COO by row, then by column for CSR construction
        let mut indices = (0..nnz).collect::<Vec<_>>();
        indices.sort_by(|&a, &b| {
            let row_a = self.row_indices[a];
            let row_b = self.row_indices[b];
            let col_a = self.col_indices[a];
            let col_b = self.col_indices[b];

            row_a.cmp(&row_b).then(col_a.cmp(&col_b))
        });

        // Build CSR components
        let mut indptr = vec![0; rows + 1];
        let mut csr_indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);

        // Count elements per row and build indptr
        for &idx in &indices {
            let row = self.row_indices[idx];
            let col = self.col_indices[idx];

            indptr[row + 1] += 1;
            csr_indices.push(col);
            data.push(self.data[idx]);
        }

        // Convert counts to cumulative sums
        for i in 1..=rows {
            indptr[i] += indptr[i - 1];
        }

        CsrStorage {
            data,
            indices: csr_indices,
            indptr,
            shape: self.shape.clone(),
        }
    }

    /// Converts COO to CSC format.
    #[must_use]
    pub fn to_csc(&self) -> CscStorage<T>
    where
        T: Clone,
    {
        let cols = self.shape.dims()[1];
        let nnz = self.nnz();

        // Sort COO by column, then by row for CSC construction
        let mut indices = (0..nnz).collect::<Vec<_>>();
        indices.sort_by(|&a, &b| {
            let col_a = self.col_indices[a];
            let col_b = self.col_indices[b];
            let row_a = self.row_indices[a];
            let row_b = self.row_indices[b];

            col_a.cmp(&col_b).then(row_a.cmp(&row_b))
        });

        // Build CSC components
        let mut indptr = vec![0; cols + 1];
        let mut csc_indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);

        // Count elements per column and build indptr
        for &idx in &indices {
            let row = self.row_indices[idx];
            let col = self.col_indices[idx];

            indptr[col + 1] += 1;
            csc_indices.push(row);
            data.push(self.data[idx]);
        }

        // Convert counts to cumulative sums
        for i in 1..=cols {
            indptr[i] += indptr[i - 1];
        }

        CscStorage {
            data,
            indices: csc_indices,
            indptr,
            shape: self.shape.clone(),
        }
    }

    /// Transposes the COO matrix (rows become columns, columns become rows).
    ///
    /// # Panics
    /// Panics if the transposed shape cannot be created (should never happen for valid matrices).
    #[must_use]
    pub fn transpose(&self) -> CooStorage<T>
    where
        T: Clone,
    {
        let transposed_data = self.data.clone();
        let transposed_row_indices = self.col_indices.clone();
        let transposed_col_indices = self.row_indices.clone();
        let transposed_shape = Shape::new(&[self.shape.dims()[1], self.shape.dims()[0]]).unwrap();

        CooStorage {
            data: transposed_data,
            row_indices: transposed_row_indices,
            col_indices: transposed_col_indices,
            shape: transposed_shape,
        }
    }
}

impl<T: DataType> CsrStorage<T> {
    /// Returns the column indices as a slice.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the row pointers as a slice.
    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Converts CSR to COO format.
    #[must_use]
    pub fn to_coo(&self) -> CooStorage<T>
    where
        T: Clone,
    {
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for row in 0..self.shape.dims()[0] {
            let row_start = self.indptr[row];
            let row_end = self.indptr[row + 1];

            for idx in row_start..row_end {
                row_indices.push(row);
                col_indices.push(self.indices[idx]);
                data.push(self.data[idx]);
            }
        }

        CooStorage {
            data,
            row_indices,
            col_indices,
            shape: self.shape.clone(),
        }
    }

    /// Converts CSR to CSC format.
    #[must_use]
    pub fn to_csc(&self) -> CscStorage<T>
    where
        T: Clone,
    {
        // First convert to COO, then to CSC
        let coo = self.to_coo();
        coo.to_csc()
    }

    /// Transposes the CSR matrix (rows become columns, columns become rows).
    #[must_use]
    pub fn transpose(&self) -> CsrStorage<T>
    where
        T: Clone,
    {
        // Convert to COO, transpose, then back to CSR
        let coo = self.to_coo();
        coo.transpose().to_csr()
    }
}

impl<T: DataType> CscStorage<T> {
    /// Returns the row indices as a slice.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the column pointers as a slice.
    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Converts CSC to COO format.
    #[must_use]
    pub fn to_coo(&self) -> CooStorage<T>
    where
        T: Clone,
    {
        let mut row_indices = Vec::with_capacity(self.nnz());
        let mut col_indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for col in 0..self.shape.dims()[1] {
            let col_start = self.indptr[col];
            let col_end = self.indptr[col + 1];

            for idx in col_start..col_end {
                row_indices.push(self.indices[idx]);
                col_indices.push(col);
                data.push(self.data[idx]);
            }
        }

        CooStorage {
            data,
            row_indices,
            col_indices,
            shape: self.shape.clone(),
        }
    }

    /// Converts CSC to CSR format.
    #[must_use]
    pub fn to_csr(&self) -> CsrStorage<T>
    where
        T: Clone,
    {
        // First convert to COO, then to CSR
        let coo = self.to_coo();
        coo.to_csr()
    }

    /// Transposes the CSC matrix (rows become columns, columns become rows).
    #[must_use]
    pub fn transpose(&self) -> CscStorage<T>
    where
        T: Clone,
    {
        // Convert to COO, transpose, then back to CSC
        let coo = self.to_coo();
        coo.transpose().to_csc()
    }
}

// Storage trait implementations for sparse formats

impl<T: DataType> AsAny for CsrStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> Storage<T> for CsrStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        // Sparse storage doesn't have meaningful strides in the dense sense
        // Return empty slice for compatibility
        &[]
    }

    fn is_contiguous(&self) -> bool {
        // Sparse storage is not contiguous by definition
        false
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }
}

impl<T: DataType> AsAny for CscStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> Storage<T> for CscStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        // Sparse storage doesn't have meaningful strides
        &[]
    }

    fn is_contiguous(&self) -> bool {
        false
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }
}

impl<T: DataType> AsAny for CooStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> Storage<T> for CooStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        // Sparse storage doesn't have meaningful strides
        &[]
    }

    fn is_contiguous(&self) -> bool {
        false
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }
}

// StorageFromVec implementations for sparse storage types

impl<T: DataType> crate::StorageFromVec<T> for CsrStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // For sparse storage, we need to convert from dense representation
        // Extract non-zero elements and build CSR structure
        let rows = dims[0];
        let cols = dims[1];

        if data.len() != rows * cols {
            return Err(crate::StorageError::ShapeMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }

        let mut csr_data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0; rows + 1];

        for row in 0..rows {
            for col in 0..cols {
                let flat_idx = row * cols + col;
                let val = &data[flat_idx];

                if !val.is_zero() {
                    csr_data.push(*val);
                    indices.push(col);
                }
            }
            indptr[row + 1] = csr_data.len();
        }

        Self::new(csr_data, indices, indptr, dims)
    }

    fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // Create empty CSR matrix with given dimensions
        let rows = dims[0];
        let indptr = vec![0; rows + 1];

        Self::new(vec![], vec![], indptr, dims)
    }

    fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
        T: num_traits::One,
    {
        // Create a dense matrix filled with ones and convert to sparse
        let size = dims.iter().product();
        let data = vec![T::one(); size];
        Self::from_vec(data, dims)
    }
}

impl<T: DataType> crate::StorageFromVec<T> for CscStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // For CSC storage, convert from dense representation
        let rows = dims[0];
        let cols = dims[1];

        if data.len() != rows * cols {
            return Err(crate::StorageError::ShapeMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }

        let mut csc_data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0; cols + 1];

        for col in 0..cols {
            for row in 0..rows {
                let flat_idx = row * cols + col;
                let val = &data[flat_idx];

                if !val.is_zero() {
                    csc_data.push(*val);
                    indices.push(row);
                }
            }
            indptr[col + 1] = csc_data.len();
        }

        Self::new(csc_data, indices, indptr, dims)
    }

    fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // Create empty CSC matrix with given dimensions
        let cols = dims[1];
        let indptr = vec![0; cols + 1];

        Self::new(vec![], vec![], indptr, dims)
    }

    fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
        T: num_traits::One,
    {
        // Create a dense matrix filled with ones and convert to sparse
        let size = dims.iter().product();
        let data = vec![T::one(); size];
        Self::from_vec(data, dims)
    }
}

impl<T: DataType> crate::StorageFromVec<T> for CooStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // For COO storage, convert from dense representation
        let rows = dims[0];
        let cols = dims[1];

        if data.len() != rows * cols {
            return Err(crate::StorageError::ShapeMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }

        let mut coo_data = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let flat_idx = row * cols + col;
                let val = &data[flat_idx];

                if !val.is_zero() {
                    coo_data.push(*val);
                    row_indices.push(row);
                    col_indices.push(col);
                }
            }
        }

        Self::new(coo_data, row_indices, col_indices, dims)
    }

    fn zeros(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
    {
        // Create empty COO matrix with given dimensions
        Self::new(vec![], vec![], vec![], dims)
    }

    fn ones(dims: &[usize]) -> crate::Result<Self>
    where
        Self: Sized,
        T: num_traits::One,
    {
        // Create a dense matrix filled with ones and convert to sparse
        let size = dims.iter().product();
        let data = vec![T::one(); size];
        Self::from_vec(data, dims)
    }
}

// StorageToDense implementations for sparse storage types

impl<T: DataType> crate::StorageToDense<T> for CsrStorage<T> {
    fn to_dense(&self) -> crate::Result<crate::DenseStorage<T>> {
        let mut dense_data = vec![T::zero(); self.shape().size()];
        let dims = self.shape().dims();

        for row in 0..dims[0] {
            let row_start = self.indptr[row];
            let row_end = self.indptr[row + 1];

            for i in row_start..row_end {
                let col = self.indices[i];
                let flat_idx = row * dims[1] + col;
                dense_data[flat_idx] = self.data[i];
            }
        }

        crate::DenseStorage::from_vec(dense_data, dims)
    }
}

impl<T: DataType> crate::StorageToDense<T> for CscStorage<T> {
    fn to_dense(&self) -> crate::Result<crate::DenseStorage<T>> {
        let mut dense_data = vec![T::zero(); self.shape().size()];
        let dims = self.shape().dims();

        for col in 0..dims[1] {
            let col_start = self.indptr[col];
            let col_end = self.indptr[col + 1];

            for i in col_start..col_end {
                let row = self.indices[i];
                let flat_idx = row * dims[1] + col;
                dense_data[flat_idx] = self.data[i];
            }
        }

        crate::DenseStorage::from_vec(dense_data, dims)
    }
}

impl<T: DataType> crate::StorageToDense<T> for CooStorage<T> {
    fn to_dense(&self) -> crate::Result<crate::DenseStorage<T>> {
        let mut dense_data = vec![T::zero(); self.shape().size()];
        let dims = self.shape().dims();

        for i in 0..self.nnz() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            let flat_idx = row * dims[1] + col;
            dense_data[flat_idx] = self.data[i];
        }

        crate::DenseStorage::from_vec(dense_data, dims)
    }
}

// Implement MatMulStorage for sparse types using existing sparse arithmetic
impl<T: DataType> crate::MatMulStorage<T> for CsrStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
{
    fn matmul_storage(&self, other: &Self) -> crate::Result<Self> {
        // Use the existing sparse matrix multiplication
        let result_coo = self.matmul_sparse(other, crate::SparseFormat::Csr)?;
        Ok(result_coo.to_csr())
    }
}

impl<T: DataType> crate::MatMulStorage<T> for CscStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
{
    fn matmul_storage(&self, other: &Self) -> crate::Result<Self> {
        // Convert to CSR, multiply, convert back to CSC
        let self_csr = self.to_csr();
        let other_csr = other.to_csr();
        let result_csr = self_csr.matmul_sparse(&other_csr, crate::SparseFormat::Csr)?;
        Ok(result_csr.to_csc())
    }
}

impl<T: DataType> crate::MatMulStorage<T> for CooStorage<T>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
{
    fn matmul_storage(&self, other: &Self) -> crate::Result<Self> {
        // Use COO × COO multiplication
        self.matmul_sparse(other, crate::SparseFormat::Coo)
    }
}

// Implement TransposeStorage for sparse types
impl<T: DataType> crate::TransposeStorage<T> for CsrStorage<T> {
    fn transpose_storage(&self, dim0: usize, dim1: usize) -> crate::Result<Self> {
        if dim0 != 0 || dim1 != 1 {
            // Only support 2D transpose for now
            return Err(crate::StorageError::ShapeMismatch {
                expected: 2,
                actual: dim0.max(dim1),
            });
        }

        // Use existing transpose implementation
        Ok(self.transpose())
    }
}

impl<T: DataType> crate::TransposeStorage<T> for CscStorage<T> {
    fn transpose_storage(&self, dim0: usize, dim1: usize) -> crate::Result<Self> {
        if dim0 != 0 || dim1 != 1 {
            // Only support 2D transpose for now
            return Err(crate::StorageError::ShapeMismatch {
                expected: 2,
                actual: dim0.max(dim1),
            });
        }

        // Convert to CSR, transpose, convert back to CSC
        let csr = self.to_csr();
        let transposed_csr = csr.transpose();
        Ok(transposed_csr.to_csc())
    }
}

impl<T: DataType> crate::TransposeStorage<T> for CooStorage<T> {
    fn transpose_storage(&self, dim0: usize, dim1: usize) -> crate::Result<Self> {
        if dim0 != 0 || dim1 != 1 {
            // Only support 2D transpose for now
            return Err(crate::StorageError::ShapeMismatch {
                expected: 2,
                actual: dim0.max(dim1),
            });
        }

        // Convert to CSR, transpose, convert back to COO
        let csr = self.to_csr();
        let transposed_csr = csr.transpose();
        Ok(transposed_csr.to_coo())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use dtype::float::Float32;

    #[test]
    fn test_csr_storage_creation() {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        // Non-zeros: (0,0)=1, (0,2)=2, (1,1)=3
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let indices = vec![0, 2, 1]; // column indices
        let indptr = vec![0, 2, 3]; // row pointers

        let storage = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 3]);
        assert_eq!(storage.nnz(), 3);
        assert!((storage.sparsity() - 3.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_csr_storage_invalid_indptr_length() {
        let data = vec![Float32::new(1.0)];
        let indices = vec![0];
        let indptr = vec![0, 1, 2]; // Should be length 3 for 2 rows

        let result = CsrStorage::new(data, indices, indptr, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_storage_invalid_column_index() {
        let data = vec![Float32::new(1.0)];
        let indices = vec![3]; // Column 3 doesn't exist in 2x2 matrix
        let indptr = vec![0, 1, 1];

        let result = CsrStorage::new(data, indices, indptr, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_storage_creation() {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        // Non-zeros: (0,0)=1, (0,2)=2, (1,1)=3
        let data = vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)];
        let indices = vec![0, 1, 0]; // row indices
        let indptr = vec![0, 1, 2, 3]; // column pointers

        let storage = CscStorage::new(data, indices, indptr, &[2, 3]).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 3]);
        assert_eq!(storage.nnz(), 3);
    }

    #[test]
    fn test_coo_storage_creation() {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        // Non-zeros: (0,0)=1, (0,2)=2, (1,1)=3
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 1];

        let storage = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 3]);
        assert_eq!(storage.nnz(), 3);
    }

    #[test]
    fn test_coo_storage_sort() {
        // Unsorted COO: (1,0), (0,2), (0,0) -> should sort to (0,0), (0,2), (1,0)
        let data = vec![Float32::new(3.0), Float32::new(2.0), Float32::new(1.0)];
        let row_indices = vec![1, 0, 0];
        let col_indices = vec![0, 2, 0];

        let mut storage = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();
        storage.sort();

        // After sorting: (0,0), (0,2), (1,0)
        assert_eq!(storage.row_indices, vec![0, 0, 1]);
        assert_eq!(storage.col_indices, vec![0, 2, 0]);
        assert_eq!(
            storage.data,
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)]
        );
    }

    #[test]
    fn test_sparse_storage_trait_impls() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let indices = vec![0, 1];
        let indptr = vec![0, 1, 2];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 2]).unwrap();

        // Test Storage trait methods
        assert_eq!(csr.as_slice().len(), 2);
        assert_eq!(csr.shape().dims(), &[2, 2]);
        assert_eq!(csr.strides(), &[] as &[usize]);
        assert!(!csr.is_contiguous());
        assert_eq!(csr.len(), 4); // Total elements, not nnz
    }

    #[test]
    fn test_csr_to_coo_conversion() {
        // Create CSR: [[1, 0, 2], [0, 3, 0]]
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let coo = csr.to_coo();

        // Verify COO format
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape().dims(), &[2, 3]);
        assert_eq!(coo.row_indices, vec![0, 0, 1]);
        assert_eq!(coo.col_indices, vec![0, 2, 1]);
        assert_eq!(
            coo.data,
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)]
        );
    }

    #[test]
    fn test_csc_to_coo_conversion() {
        // Create CSC: [[1, 0, 2], [0, 3, 0]]
        let data = vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)];
        let indices = vec![0, 1, 0];
        let indptr = vec![0, 1, 2, 3];
        let csc = CscStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let coo = csc.to_coo();

        // Verify COO format
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape().dims(), &[2, 3]);
        assert_eq!(coo.row_indices, vec![0, 1, 0]);
        assert_eq!(coo.col_indices, vec![0, 1, 2]);
        assert_eq!(
            coo.data,
            vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)]
        );
    }

    #[test]
    fn test_coo_to_csr_conversion() {
        // Create COO: (0,0)=1, (0,2)=2, (1,1)=3
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();

        let csr = coo.to_csr();

        // Verify CSR format
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.shape().dims(), &[2, 3]);
        assert_eq!(
            csr.data,
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)]
        );
        assert_eq!(csr.indices, vec![0, 2, 1]);
        assert_eq!(csr.indptr, vec![0, 2, 3]);
    }

    #[test]
    fn test_coo_to_csc_conversion() {
        // Create COO: (0,0)=1, (0,2)=2, (1,1)=3
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();

        let csc = coo.to_csc();

        // Verify CSC format
        assert_eq!(csc.nnz(), 3);
        assert_eq!(csc.shape().dims(), &[2, 3]);
        assert_eq!(
            csc.data,
            vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)]
        );
        assert_eq!(csc.indices, vec![0, 1, 0]);
        assert_eq!(csc.indptr, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_format_conversion_round_trip() {
        // Test CSR -> COO -> CSR
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let original_csr = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let coo = original_csr.to_coo();
        let round_trip_csr = coo.to_csr();

        assert_eq!(original_csr.data, round_trip_csr.data);
        assert_eq!(original_csr.indices, round_trip_csr.indices);
        assert_eq!(original_csr.indptr, round_trip_csr.indptr);
        assert_eq!(original_csr.shape.dims(), round_trip_csr.shape.dims());

        // Test CSC -> COO -> CSC
        let data = vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)];
        let indices = vec![0, 1, 0];
        let indptr = vec![0, 1, 2, 3];
        let original_csc_storage = CscStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let coo = original_csc_storage.to_coo();
        let round_trip_csc_storage = coo.to_csc();

        assert_eq!(original_csc_storage.data, round_trip_csc_storage.data);
        assert_eq!(original_csc_storage.indices, round_trip_csc_storage.indices);
        assert_eq!(original_csc_storage.indptr, round_trip_csc_storage.indptr);
        assert_eq!(
            original_csc_storage.shape.dims(),
            round_trip_csc_storage.shape.dims()
        );
    }

    #[test]
    fn test_csr_to_csc_conversion() {
        // Create CSR: [[1, 0, 2], [0, 3, 0]]
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let csc = csr.to_csc();

        // Verify CSC format
        assert_eq!(csc.nnz(), 3);
        assert_eq!(csc.shape().dims(), &[2, 3]);
        assert_eq!(
            csc.data,
            vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)]
        );
        assert_eq!(csc.indices, vec![0, 1, 0]);
        assert_eq!(csc.indptr, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_csc_to_csr_conversion() {
        // Create CSC: [[1, 0, 2], [0, 3, 0]]
        let data = vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)];
        let indices = vec![0, 1, 0];
        let indptr = vec![0, 1, 2, 3];
        let csc = CscStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let csr = csc.to_csr();

        // Verify CSR format
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.shape().dims(), &[2, 3]);
        assert_eq!(
            csr.data,
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)]
        );
        assert_eq!(csr.indices, vec![0, 2, 1]);
        assert_eq!(csr.indptr, vec![0, 2, 3]);
    }

    #[test]
    fn test_csr_transpose() {
        // Create 2×3 matrix: [[1, 0, 2], [0, 3, 0]]
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let transposed = csr.transpose();

        // Transposed should be 3×2: [[1, 0], [0, 3], [2, 0]]
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.nnz(), 3);

        // Convert to COO to check values
        let coo = transposed.to_coo();
        assert_eq!(coo.row_indices, vec![0, 1, 2]);
        assert_eq!(coo.col_indices, vec![0, 1, 0]);
        assert_eq!(
            coo.data,
            vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)]
        );
    }

    #[test]
    fn test_csc_transpose() {
        // Create 2×3 matrix: [[1, 0, 2], [0, 3, 0]]
        let data = vec![Float32::new(1.0), Float32::new(3.0), Float32::new(2.0)];
        let indices = vec![0, 1, 0];
        let indptr = vec![0, 1, 2, 3];
        let csc = CscStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let transposed = csc.transpose();

        // Transposed should be 3×2: [[1, 0], [0, 3], [2, 0]]
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.nnz(), 3);
    }

    #[test]
    fn test_coo_transpose() {
        // Create COO: (0,0)=1, (0,2)=2, (1,1)=3
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();

        let transposed = coo.transpose();

        // Transposed should be 3×2: (0,0)=1, (2,0)=2, (1,1)=3
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.nnz(), 3);
        assert_eq!(transposed.row_indices, vec![0, 2, 1]);
        assert_eq!(transposed.col_indices, vec![0, 0, 1]);
        assert_eq!(
            transposed.data,
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)]
        );
    }

    #[test]
    fn test_transpose_round_trip() {
        // Test that transpose(transpose(matrix)) == matrix
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let original = CsrStorage::new(data, indices, indptr, &[2, 3]).unwrap();

        let transposed = original.transpose();
        let round_trip = transposed.transpose();

        assert_eq!(original.data, round_trip.data);
        assert_eq!(original.indices, round_trip.indices);
        assert_eq!(original.indptr, round_trip.indptr);
        assert_eq!(original.shape.dims(), round_trip.shape.dims());
    }
}
