//! Coordinate-list (COO) sparse tensor.

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
