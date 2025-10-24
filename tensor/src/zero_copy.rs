//! Zero-copy optimizations using GATs (Generic Associated Types)
//!
//! This module provides advanced zero-copy abstractions using GATs for efficient
//! lifetime management and compile-time borrow checking. GATs allow us to express
//! complex lifetime relationships that are impossible with traditional generics.

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::{Backend, Tensor};
use coeus_dtype::DataType;
use coeus_storage::Storage;

/// Generic Associated Type for tensor views with lifetime polymorphism.
///
/// This trait enables zero-copy tensor operations with compile-time lifetime safety.
/// GATs allow the return type to reference the input lifetime, enabling views
/// that borrow from the original tensor without copying data.
///
/// # Examples
///
/// ```rust
/// use coeus_tensor::zero_copy::{TensorView, ViewGAT};
///
/// // Create a view that borrows from the tensor
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let view = tensor.view(); // Zero-copy view
/// assert_eq!(view.as_slice(), tensor.as_slice()); // Same underlying data
/// ```
pub trait ViewGAT<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// The view type with lifetime polymorphism
    type View<'a>: Deref<Target = Tensor<B, S, T>> + 'a
    where
        Self: 'a,
        B: 'a,
        S: 'a,
        T: 'a;

    /// The mutable view type with lifetime polymorphism
    type ViewMut<'a>: DerefMut<Target = Tensor<B, S, T>> + 'a
    where
        Self: 'a,
        B: 'a,
        S: 'a,
        T: 'a;

    /// Create an immutable view of the tensor
    fn view(&self) -> Self::View<'_>;

    /// Create a mutable view of the tensor
    fn view_mut(&mut self) -> Self::ViewMut<'_>;
}

/// Zero-copy tensor slice with GAT-based lifetime management.
///
/// This struct represents a view into a tensor that borrows the underlying data
/// without copying. The GAT ensures compile-time lifetime safety while enabling
/// zero-copy operations.
///
/// # Type Parameters
///
/// * `'tensor` - The lifetime of the original tensor being viewed
/// * `B` - The backend type
/// * `S` - The storage type
/// * `T` - The data type
#[derive(Debug)]
pub struct TensorSlice<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Reference to the original tensor
    tensor: &'tensor Tensor<B, S, T>,
    /// Start offset in the flattened tensor
    offset: usize,
    /// Length of the slice
    len: usize,
    /// Shape of the slice view
    shape: Vec<usize>,
    /// Strides for indexing the slice
    strides: Vec<usize>,
}

impl<'tensor, B, S, T> TensorSlice<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Create a new tensor slice from a tensor reference
    pub fn new(
        tensor: &'tensor Tensor<B, S, T>,
        offset: usize,
        len: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        Self {
            tensor,
            offset,
            len,
            shape,
            strides,
        }
    }

    /// Get the shape of the slice
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides for indexing
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the underlying tensor data as a slice (zero-copy)
    pub fn as_slice(&self) -> &[T] {
        let data = self.tensor.as_slice();
        &data[self.offset..self.offset + self.len]
    }
}

impl<'tensor, B, S, T> Deref for TensorSlice<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    type Target = Tensor<B, S, T>;

    fn deref(&self) -> &Self::Target {
        self.tensor
    }
}

/// Iterator over tensor elements with zero-copy semantics.
///
/// This iterator yields references to tensor elements without copying.
pub trait TensorIterator<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Create an iterator over tensor elements
    fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_>;

    /// Create an iterator over mutable tensor elements
    fn iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut T> + '_>;
}

/// Zero-copy tensor windowing for convolution operations.
///
/// This struct provides efficient sliding window views for convolution
/// operations without copying data, using GATs for lifetime safety.
pub struct ConvWindow<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Reference to the input tensor
    input: &'tensor Tensor<B, S, T>,
    /// Kernel dimensions (height, width)
    kernel_size: (usize, usize),
    /// Stride dimensions (height, width)
    stride: (usize, usize),
    /// Padding dimensions (height, width)
    padding: (usize, usize),
    /// Current window position
    position: (usize, usize),
    /// Phantom data for type safety
    _phantom: PhantomData<(B, S, T)>,
}

impl<'tensor, B, S, T> ConvWindow<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Create a new convolution window iterator
    pub fn new(
        input: &'tensor Tensor<B, S, T>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            input,
            kernel_size,
            stride,
            padding,
            position: (0, 0),
            _phantom: PhantomData,
        }
    }

    /// Get the current window as a zero-copy slice
    pub fn current_window(&self, batch: usize, channel: usize) -> Option<&[T]> {
        let input_shape = self.input.shape().dims();
        let (_, channels, height, width) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (oh, ow) = self.position;

        // Calculate input coordinates
        let ih_start = oh * sh;
        let iw_start = ow * sw;

        // Check bounds with padding
        if ih_start >= height + 2 * ph || iw_start >= width + 2 * pw {
            return None;
        }

        let ih_end = ih_start + kh;
        let iw_end = iw_start + kw;

        let actual_ih_start = ih_start.saturating_sub(ph);
        let actual_iw_start = iw_start.saturating_sub(pw);
        let actual_ih_end = (ih_end - ph).min(height);
        let actual_iw_end = (iw_end - pw).min(width);

        if actual_ih_start >= actual_ih_end || actual_iw_start >= actual_iw_end {
            return None;
        }

        // Calculate the flat offset for this batch/channel
        let batch_offset = batch * channels * height * width;
        let channel_offset = channel * height * width;

        let data = self.input.as_slice();
        let start_idx = batch_offset + channel_offset + actual_ih_start * width + actual_iw_start;

        // For now, return the entire channel slice (simplified implementation)
        // A full implementation would create a windowed view
        Some(&data[start_idx..start_idx + (actual_ih_end - actual_ih_start) * (actual_iw_end - actual_iw_start)])
    }

    /// Move to the next window position
    pub fn advance(&mut self) -> bool {
        let input_shape = self.input.shape().dims();
        let (_, _, height, width) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        let out_height = (height + 2 * ph - kh) / sh + 1;
        let out_width = (width + 2 * pw - kw) / sw + 1;

        let (mut oh, mut ow) = self.position;
        ow += 1;

        if ow >= out_width {
            ow = 0;
            oh += 1;
        }

        self.position = (oh, ow);
        oh < out_height
    }
}

/// Zero-copy tensor broadcasting using GATs.
///
/// This trait provides broadcasting operations that return views instead of copies,
/// using GATs to ensure proper lifetime management.
pub trait BroadcastGAT<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// The broadcast view type with lifetime polymorphism
    type BroadcastView<'a>: Deref<Target = Tensor<B, S, T>> + 'a
    where
        Self: 'a,
        B: 'a,
        S: 'a,
        T: 'a;

    /// Broadcast tensor to target shape without copying
    fn broadcast_to<'a>(&'a self, target_shape: &[usize]) -> Result<Self::BroadcastView<'a>, crate::TensorError>;
}

/// Iterator that yields zero-copy views of tensor chunks.
///
/// This iterator is useful for batch processing where each batch element
/// can be processed without copying data.
pub struct ChunkIterator<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Reference to the tensor being chunked
    tensor: &'tensor Tensor<B, S, T>,
    /// Dimension along which to chunk
    dim: usize,
    /// Size of each chunk
    chunk_size: usize,
    /// Current chunk index
    current_idx: usize,
    /// Total number of chunks
    total_chunks: usize,
}

impl<'tensor, B, S, T> ChunkIterator<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Create a new chunk iterator
    pub fn new(tensor: &'tensor Tensor<B, S, T>, dim: usize, chunk_size: usize) -> Self {
        let shape = tensor.shape().dims();
        let total_chunks = if dim < shape.len() {
            (shape[dim] + chunk_size - 1) / chunk_size
        } else {
            1
        };

        Self {
            tensor,
            dim,
            chunk_size,
            current_idx: 0,
            total_chunks,
        }
    }
}

impl<'tensor, B, S, T> Iterator for ChunkIterator<'tensor, B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    type Item = TensorSlice<'tensor, B, S, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.total_chunks {
            return None;
        }

        let shape = self.tensor.shape().dims();
        let start_idx = self.current_idx * self.chunk_size;
        let actual_chunk_size = (shape[self.dim] - start_idx).min(self.chunk_size);

        // Create a slice view of this chunk
        let mut chunk_shape = shape.to_vec();
        chunk_shape[self.dim] = actual_chunk_size;

        // Calculate offset and strides for the slice
        let mut offset = 0;
        let mut strides = vec![1; shape.len()];

        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }

        for i in 0..self.dim {
            offset += start_idx * strides[i];
        }

        let len = chunk_shape.iter().product();

        self.current_idx += 1;

        Some(TensorSlice::new(
            self.tensor,
            offset,
            len,
            chunk_shape,
            strides,
        ))
    }
}

/// Extension trait for tensors to provide zero-copy operations
pub trait ZeroCopyExt<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Create a chunk iterator for zero-copy batch processing
    fn chunks(&self, dim: usize, chunk_size: usize) -> ChunkIterator<'_, B, S, T>;

    /// Create a convolution window iterator
    fn conv_windows(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> ConvWindow<'_, B, S, T>;
}

impl<B, S, T> ZeroCopyExt<B, S, T> for Tensor<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    fn chunks(&self, dim: usize, chunk_size: usize) -> ChunkIterator<'_, B, S, T> {
        ChunkIterator::new(self, dim, chunk_size)
    }

    fn conv_windows(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> ConvWindow<'_, B, S, T> {
        ConvWindow::new(self, kernel_size, stride, padding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, DenseStorage};
    use crate::float::Float32;

    #[test]
    fn test_chunk_iterator() {
        let data = vec![
            Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
            Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),
        ];
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            data, &[2, 3]
        ).unwrap();

        let mut chunks: Vec<_> = tensor.chunks(1, 2).collect();
        assert_eq!(chunks.len(), 2);

        // First chunk should have shape [2, 2]
        assert_eq!(chunks[0].shape(), &[2, 2]);
        // Second chunk should have shape [2, 1]
        assert_eq!(chunks[1].shape(), &[2, 1]);
    }

    #[test]
    fn test_conv_window() {
        let data = vec![Float32::new(1.0); 64]; // 1x1x8x8 tensor
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            data, &[1, 1, 8, 8]
        ).unwrap();

        let mut window = tensor.conv_windows((3, 3), (1, 1), (0, 0));
        // Should have 6x6 = 36 windows
        let mut count = 0;
        while window.advance() {
            count += 1;
        }
        assert_eq!(count, 36);
    }
}
