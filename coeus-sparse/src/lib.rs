//! # coeus-sparse
//!
//! Pure sparse tensor data structures for Coeus: [`CooTensor`] (coordinate-list,
//! N-D) and [`CsrTensor`] (compressed-sparse-row, 2-D). Following DIP, these
//! types hold only shape, indices, and values — they carry no arithmetic.
//! Conversions (dense ↔ COO ↔ CSR) and sparse kernels (SpMV/SpMM and their
//! backward passes) live in `coeus-ops`/`coeus-leto`, keeping the data
//! structures backend-agnostic and isolation-testable.
//!
//! Both constructors validate their structural invariants (index rank/count,
//! 2-D CSR shape, `row_offsets` length); see `tests/sparse_tests.rs`.

#![deny(missing_docs)]

use coeus_core::{ComputeBackend, MoiraiBackend, Scalar, Shape};
use coeus_tensor::Tensor;

/// N-Dimensional Sparse Tensor in Coordinate List (COO) format.
///
/// # Examples
///
/// Create a 2×3 COO tensor with 2 non-zero entries:
///
/// ```
/// use coeus_sparse::CooTensor;
/// use coeus_core::Shape;
/// use coeus_tensor::Tensor;
///
/// let indices = Tensor::<i64>::from_slice([2, 2], &[0, 1, 1, 2]); // (0,0) and (1,2)
/// let values = Tensor::<f32>::from_slice([2], &[5.0, 7.0]);
/// let coo = CooTensor::new(Shape::from(vec![2, 3]), indices, values);
/// assert_eq!(coo.nnz(), 2);
/// assert_eq!(coo.shape().as_ref(), &[2, 3]);
/// ```
#[derive(Clone)]
pub struct CooTensor<T: Scalar, B: ComputeBackend = MoiraiBackend> {
    shape: Shape,
    indices: Tensor<i64, B>, // Shape [rank, nnz]
    values: Tensor<T, B>,    // Shape [nnz]
}

impl<T: Scalar, B: ComputeBackend> CooTensor<T, B> {
    /// Create a new CooTensor with shape, coordinate indices, and non-zero values.
    ///
    /// # Panics
    /// If `indices` is not 2-D `[rank, nnz]`, or if dimensions are inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_sparse::CooTensor;
    /// use coeus_core::Shape;
    /// use coeus_tensor::Tensor;
    ///
    /// let indices = Tensor::<i64>::from_slice([2, 1], &[1, 0]); // entry at (1,0)
    /// let values = Tensor::<f32>::from_slice([1], &[3.0]);
    /// let coo = CooTensor::new(Shape::from(vec![2, 2]), indices, values);
    /// assert_eq!(coo.nnz(), 1);
    /// ```
    #[inline]
    pub fn new(shape: Shape, indices: Tensor<i64, B>, values: Tensor<T, B>) -> Self {
        let rank = shape.len();
        assert_eq!(
            indices.shape().len(),
            2,
            "Indices tensor must be 2D [rank, nnz]"
        );
        assert_eq!(
            indices.shape()[0],
            rank,
            "Indices row count must match tensor rank"
        );
        let nnz = values.numel();
        assert_eq!(
            indices.shape()[1],
            nnz,
            "Indices col count must match number of values"
        );
        Self {
            shape,
            indices,
            values,
        }
    }

    /// Access the shape of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_sparse::CooTensor;
    /// use coeus_core::Shape;
    /// use coeus_tensor::Tensor;
    ///
    /// let indices = Tensor::<i64>::from_slice([2, 1], &[0, 0]);
    /// let values = Tensor::<f32>::from_slice([1], &[1.0]);
    /// let coo = CooTensor::new(Shape::from(vec![3, 4]), indices, values);
    /// assert_eq!(coo.shape().as_ref(), &[3, 4]);
    /// ```
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Access the coordinate indices `[rank, nnz]`.
    #[inline]
    pub fn indices(&self) -> &Tensor<i64, B> {
        &self.indices
    }

    /// Access the non-zero values `[nnz]`.
    #[inline]
    pub fn values(&self) -> &Tensor<T, B> {
        &self.values
    }

    /// Return the number of non-zero elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_sparse::CooTensor;
    /// use coeus_core::Shape;
    /// use coeus_tensor::Tensor;
    ///
    /// let indices = Tensor::<i64>::from_slice([2, 3], &[0, 1, 0, 0, 1, 2]);
    /// let values = Tensor::<f32>::from_slice([3], &[1.0, 2.0, 3.0]);
    /// let coo = CooTensor::new(Shape::from(vec![2, 3]), indices, values);
    /// assert_eq!(coo.nnz(), 3);
    /// ```
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.numel()
    }
}

/// 2D Sparse Matrix in Compressed Sparse Row (CSR) format.
///
/// # Examples
///
/// Create a 3×3 CSR matrix with 3 non-zero entries:
///
/// ```
/// use coeus_sparse::CsrTensor;
/// use coeus_core::Shape;
/// use coeus_tensor::Tensor;
///
/// // Matrix: [[1,0,0],[0,0,2],[0,3,0]]
/// let values = Tensor::<f32>::from_slice([3], &[1.0, 2.0, 3.0]);
/// let col_indices = Tensor::<i64>::from_slice([3], &[0, 2, 1]);
/// let row_offsets = Tensor::<i64>::from_slice([4], &[0, 1, 2, 3]);
/// let csr = CsrTensor::new(Shape::from(vec![3, 3]), values, col_indices, row_offsets);
/// assert_eq!(csr.nnz(), 3);
/// assert_eq!(csr.shape().as_ref(), &[3, 3]);
/// ```
#[derive(Clone)]
pub struct CsrTensor<T: Scalar, B: ComputeBackend = MoiraiBackend> {
    shape: Shape,                // Must be exactly 2D: [rows, cols]
    values: Tensor<T, B>,        // Shape [nnz]
    col_indices: Tensor<i64, B>, // Shape [nnz]
    row_offsets: Tensor<i64, B>, // Shape [rows + 1]
}

impl<T: Scalar, B: ComputeBackend> CsrTensor<T, B> {
    /// Create a new CsrTensor.
    ///
    /// # Panics
    /// If `shape` is not 2-D, or if `col_indices` or `row_offsets` lengths
    /// are inconsistent with `values` count or `rows + 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_sparse::CsrTensor;
    /// use coeus_core::Shape;
    /// use coeus_tensor::Tensor;
    ///
    /// // 2×2 identity matrix
    /// let values = Tensor::<f32>::from_slice([2], &[1.0, 1.0]);
    /// let col_indices = Tensor::<i64>::from_slice([2], &[0, 1]);
    /// let row_offsets = Tensor::<i64>::from_slice([3], &[0, 1, 2]);
    /// let csr = CsrTensor::new(Shape::from(vec![2, 2]), values, col_indices, row_offsets);
    /// assert_eq!(csr.nnz(), 2);
    /// ```
    #[inline]
    pub fn new(
        shape: Shape,
        values: Tensor<T, B>,
        col_indices: Tensor<i64, B>,
        row_offsets: Tensor<i64, B>,
    ) -> Self {
        assert_eq!(shape.len(), 2, "CSR format is restricted to 2D matrices");
        let rows = shape[0];
        let nnz = values.numel();
        assert_eq!(
            col_indices.numel(),
            nnz,
            "col_indices length must match values count"
        );
        assert_eq!(
            row_offsets.numel(),
            rows + 1,
            "row_offsets length must equal rows + 1"
        );
        Self {
            shape,
            values,
            col_indices,
            row_offsets,
        }
    }

    /// Access the shape `[rows, cols]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_sparse::CsrTensor;
    /// use coeus_core::Shape;
    /// use coeus_tensor::Tensor;
    ///
    /// let values = Tensor::<f32>::from_slice([1], &[5.0]);
    /// let col_indices = Tensor::<i64>::from_slice([1], &[1]);
    /// let row_offsets = Tensor::<i64>::from_slice([3], &[0, 1, 1]);
    /// let csr = CsrTensor::new(Shape::from(vec![2, 3]), values, col_indices, row_offsets);
    /// assert_eq!(csr.shape().as_ref(), &[2, 3]);
    /// ```
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Access the non-zero values `[nnz]`.
    #[inline]
    pub fn values(&self) -> &Tensor<T, B> {
        &self.values
    }

    /// Access the column indices for each non-zero element `[nnz]`.
    #[inline]
    pub fn col_indices(&self) -> &Tensor<i64, B> {
        &self.col_indices
    }

    /// Access the row offsets marking start and end of rows `[rows + 1]`.
    #[inline]
    pub fn row_offsets(&self) -> &Tensor<i64, B> {
        &self.row_offsets
    }

    /// Return the number of non-zero elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_sparse::CsrTensor;
    /// use coeus_core::Shape;
    /// use coeus_tensor::Tensor;
    ///
    /// let values = Tensor::<f32>::from_slice([2], &[1.0, 2.0]);
    /// let col_indices = Tensor::<i64>::from_slice([2], &[0, 1]);
    /// let row_offsets = Tensor::<i64>::from_slice([3], &[0, 1, 2]);
    /// let csr = CsrTensor::new(Shape::from(vec![2, 2]), values, col_indices, row_offsets);
    /// assert_eq!(csr.nnz(), 2);
    /// ```
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.numel()
    }
}
