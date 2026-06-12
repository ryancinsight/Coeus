use coeus_core::{ComputeBackend, MoiraiBackend, Scalar, Shape};
use coeus_tensor::Tensor;

/// N-Dimensional Sparse Tensor in Coordinate List (COO) format.
#[derive(Clone)]
pub struct CooTensor<T: Scalar, B: ComputeBackend = MoiraiBackend> {
    shape: Shape,
    indices: Tensor<i64, B>, // Shape [rank, nnz]
    values: Tensor<T, B>,    // Shape [nnz]
}

impl<T: Scalar, B: ComputeBackend> CooTensor<T, B> {
    /// Create a new CooTensor with shape, coordinate indices, and non-zero values.
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
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.numel()
    }
}

/// 2D Sparse Matrix in Compressed Sparse Row (CSR) format.
#[derive(Clone)]
pub struct CsrTensor<T: Scalar, B: ComputeBackend = MoiraiBackend> {
    shape: Shape,                // Must be exactly 2D: [rows, cols]
    values: Tensor<T, B>,        // Shape [nnz]
    col_indices: Tensor<i64, B>, // Shape [nnz]
    row_offsets: Tensor<i64, B>, // Shape [rows + 1]
}

impl<T: Scalar, B: ComputeBackend> CsrTensor<T, B> {
    /// Create a new CsrTensor.
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
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.numel()
    }
}
