//! CLIP encoder traits.

use backend::Backend;
use storage::DenseStorage;
use dtype::DataType;
use tensor::Tensor;
use crate::error::Result;

/// Trait for CLIP encoders that can encode text and images.
pub trait ClipEncoder<B, T>
where
    B: Backend<Data = T> + Clone,
    T: DataType + 'static,
{
    /// Encode text into an embedding.
    fn encode_text(&self, texts: &[&str]) -> Result<Tensor<B, DenseStorage<T>, T>>;

    /// Encode image into an embedding.
    fn encode_image(&self, image_data: &[f32], batch_size: usize) -> Result<Tensor<B, DenseStorage<T>, T>>;
}