//! Core transformation traits and basic transforms
//!
//! This module provides the fundamental Transform trait and basic transforms
//! that form the foundation of the transformation pipeline.

use crate::Result;
use coeus_tensor::Tensor;
use rand::Rng;

/// Trait for data transformations
pub trait Transform<T: coeus_dtype::Dtype> {
    /// Apply the transformation to the input tensor
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
}

/// Normalize transform
///
/// Normalizes tensor values to have zero mean and unit variance
/// Compatible with PyTorch's `transforms.Normalize`
#[derive(Clone)]
pub struct Normalize<T: coeus_dtype::Dtype + num_traits::Float> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T: coeus_dtype::Dtype + num_traits::Float> Normalize<T> {
    /// Create a new normalization transform
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Self {
        assert_eq!(
            mean.len(),
            std.len(),
            "Mean and std must have the same length"
        );
        Self { mean, std }
    }

    /// Create normalization for single-channel data
    pub fn from_single(mean: T, std: T) -> Self {
        Self::new(vec![mean], vec![std])
    }
}

impl<T: coeus_dtype::Dtype + num_traits::Float> Transform<T> for Normalize<T> {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Handle empty mean/std case - return input unchanged
        if self.mean.is_empty() || self.std.is_empty() {
            return Ok(input.clone());
        }

        // Implement proper normalization: (input - mean) / std
        let mut result = input.clone();

        // Apply normalization per channel if dimensions match
        if input.shape().len() >= 3 && self.mean.len() == input.shape()[1] {
            // Assume NCHW format (batch, channels, height, width)
            let channels = input.shape()[1];
            let channel_size: usize = input.shape().iter().skip(2).product();

            for c in 0..channels {
                let start_idx = c * channel_size;
                let end_idx = (c + 1) * channel_size;

                // Normalize each channel: (channel_data - mean[c]) / std[c]
                for i in start_idx..end_idx {
                    if let Some(value) = result.data_mut().get_mut(i) {
                        let normalized = (*value - self.mean[c]) / self.std[c];
                        *value = normalized;
                    }
                }
            }
        } else {
            // Simple per-element normalization for non-image tensors
            for (i, value) in result.data_mut().iter_mut().enumerate() {
                let mean_idx = i % self.mean.len();
                let std_idx = i % self.std.len();
                *value = (*value - self.mean[mean_idx]) / self.std[std_idx];
            }
        }

        Ok(result)
    }
}

/// Random crop transform
///
/// Randomly crops the input tensor to the specified size
/// Compatible with PyTorch's `transforms.RandomCrop`
#[derive(Clone)]
pub struct RandomCrop {
    size: Vec<usize>,
    #[allow(dead_code)]
    padding: Option<usize>,
}

impl RandomCrop {
    /// Create a new random crop transform
    pub fn new(size: Vec<usize>) -> Self {
        Self {
            size,
            padding: None,
        }
    }

    /// Create a random crop with padding
    pub fn with_padding(size: Vec<usize>, padding: usize) -> Self {
        Self {
            size,
            padding: Some(padding),
        }
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for RandomCrop {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            return Err(Box::<dyn std::error::Error + Send + Sync>::from(
                "RandomCrop requires at least 2D tensors",
            ));
        }

        let height = input_shape[input_shape.len() - 2];
        let width = input_shape[input_shape.len() - 1];

        // Validate crop size
        if self.size.len() != 2 {
            return Err(Box::<dyn std::error::Error + Send + Sync>::from(
                "RandomCrop size must be [height, width]",
            ));
        }

        let crop_height = self.size[0];
        let crop_width = self.size[1];

        if crop_height > height || crop_width > width {
            return Err(Box::<dyn std::error::Error + Send + Sync>::from(format!(
                "Crop size [{}, {}] exceeds input size [{}, {}]",
                crop_height, crop_width, height, width
            )));
        }

        // Generate random crop position
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let max_h_offset = height - crop_height;
        let max_w_offset = width - crop_width;

        let h_offset = if max_h_offset > 0 {
            rng.gen_range(0..=max_h_offset)
        } else {
            0
        };
        let w_offset = if max_w_offset > 0 {
            rng.gen_range(0..=max_w_offset)
        } else {
            0
        };

        // Extract crop region using tensor indexing
        // For now, implement basic 2D cropping
        if input_shape.len() == 2 {
            // Simple 2D case
            let mut cropped_data = Vec::with_capacity(crop_height * crop_width);

            for h in h_offset..h_offset + crop_height {
                for w in w_offset..w_offset + crop_width {
                    let idx = h * width + w;
                    cropped_data.push(input.data()[idx]);
                }
            }

            let cropped_shape = vec![crop_height, crop_width];
            Ok(Tensor::from_vec(cropped_data, cropped_shape))
        } else {
            // Handle multi-dimensional tensors (e.g., batch x channels x height x width)
            // Apply cropping to the last two dimensions (spatial dimensions)
            let ndim = input_shape.len();
            if ndim < 3 {
                return Err(Box::<dyn std::error::Error + Send + Sync>::from(
                    "Multi-dimensional cropping requires at least 3D tensors",
                ));
            }

            // For tensors with shape [..., H, W], crop the last two dimensions
            let batch_dims = &input_shape[..ndim - 2];
            let height = input_shape[ndim - 2];
            let width = input_shape[ndim - 1];

            // Validate crop size
            if crop_height > height || crop_width > width {
                return Err(Box::<dyn std::error::Error + Send + Sync>::from(format!(
                    "Crop size [{}, {}] exceeds spatial dimensions [{}, {}]",
                    crop_height, crop_width, height, width
                )));
            }

            // Calculate total batch size and output size
            let batch_size: usize = batch_dims.iter().product();
            let output_size = batch_size * crop_height * crop_width;

            // Generate random crop position (same for all items in batch)
            let max_h_offset = height - crop_height;
            let max_w_offset = width - crop_width;
            let h_offset = if max_h_offset > 0 {
                rng.gen_range(0..=max_h_offset)
            } else {
                0
            };
            let w_offset = if max_w_offset > 0 {
                rng.gen_range(0..=max_w_offset)
            } else {
                0
            };

            // Create output tensor
            let mut output_shape = batch_dims.to_vec();
            output_shape.push(crop_height);
            output_shape.push(crop_width);

            let mut output_data = Vec::with_capacity(output_size);

            // Perform cropping for each batch element
            for batch_idx in 0..batch_size {
                // Calculate the offset for this batch element in the input data
                let batch_offset = batch_idx * height * width;

                // Extract the cropped region
                for h in h_offset..h_offset + crop_height {
                    for w in w_offset..w_offset + crop_width {
                        let input_idx = batch_offset + h * width + w;
                        output_data.push(input.data()[input_idx]);
                    }
                }
            }

            Ok(Tensor::from_vec(output_data, output_shape))
        }
    }
}

/// Random horizontal flip transform
///
/// Randomly flips the input tensor horizontally
/// Compatible with PyTorch's `transforms.RandomHorizontalFlip`
#[derive(Clone)]
pub struct RandomHorizontalFlip {
    p: f64,
}

impl RandomHorizontalFlip {
    /// Create a new random horizontal flip transform
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability must be between 0 and 1"
        );
        Self { p }
    }

    /// Create with default probability of 0.5
    pub fn default_p() -> Self {
        Self::new(0.5)
    }

    /// Internal function to perform horizontal flip
    fn flip_horizontal<T: coeus_dtype::Dtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
        let input_shape = input.shape();
        let ndim = input_shape.len();

        if ndim < 2 {
            return Err(Box::<dyn std::error::Error + Send + Sync>::from(
                "RandomHorizontalFlip requires at least 2D tensors",
            ));
        }

        let height = input_shape[ndim - 2];
        let width = input_shape[ndim - 1];

        if ndim == 2 {
            // Simple 2D case: [height, width]
            let mut flipped_data = vec![T::zero(); input.numel()];

            for h in 0..height {
                for w in 0..width {
                    let src_idx = h * width + w;
                    let dst_idx = h * width + (width - 1 - w);
                    flipped_data[dst_idx] = input.data()[src_idx];
                }
            }

            Ok(Tensor::from_vec(flipped_data, input_shape.to_vec()))
        } else {
            // Multi-dimensional case: [..., height, width]
            let batch_dims: Vec<usize> = input_shape[..ndim - 2].to_vec();
            let batch_size: usize = batch_dims.iter().product();
            let spatial_size = height * width;

            let mut flipped_data = vec![T::zero(); input.numel()];

            for batch_idx in 0..batch_size {
                let batch_offset = batch_idx * spatial_size;

                for h in 0..height {
                    for w in 0..width {
                        let src_idx = batch_offset + h * width + w;
                        let dst_idx = batch_offset + h * width + (width - 1 - w);
                        flipped_data[dst_idx] = input.data()[src_idx];
                    }
                }
            }

            Ok(Tensor::from_vec(flipped_data, input_shape.to_vec()))
        }
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for RandomHorizontalFlip {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Generate random number to decide whether to flip
        let mut rng = rand::thread_rng();
        let rand_val: f64 = rng.gen();

        if rand_val < self.p {
            // Perform horizontal flip
            Self::flip_horizontal(input)
        } else {
            Ok(input.clone())
        }
    }
}

/// Compose multiple transforms
///
/// Applies a sequence of transforms in order
/// Compatible with PyTorch's `transforms.Compose`
pub struct Compose<T: coeus_dtype::Dtype> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: coeus_dtype::Dtype> Compose<T> {
    /// Create a new compose transform
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for Compose<T> {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let mut result = input.clone();

        for transform in &self.transforms {
            result = transform.transform(&result)?;
        }

        Ok(result)
    }
}

/// ToTensor transform
///
/// Converts various data types to tensors
/// Compatible with PyTorch's `transforms.ToTensor`
#[derive(Clone)]
pub struct ToTensor;

impl ToTensor {
    /// Create a new ToTensor transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for ToTensor {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // ToTensor typically converts various data types to tensors
        // Since input is already a tensor, return a clone
        // In a full implementation, this could handle conversion from other data structures
        Ok(input.clone())
    }
}

/// Lambda transform
///
/// Applies a custom function to the input tensor
/// Compatible with PyTorch's `transforms.Lambda`
#[derive(Clone)]
pub struct Lambda<F, T: coeus_dtype::Dtype> {
    func: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T: coeus_dtype::Dtype> Lambda<F, T>
where
    F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
{
    /// Create a new lambda transform
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, T: coeus_dtype::Dtype> Transform<T> for Lambda<F, T>
where
    F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
{
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        (self.func)(input)
    }
}

/// Identity transform
///
/// Identity transform that returns input unchanged
/// Useful for testing or as a no-op in transform pipelines
#[derive(Clone)]
pub struct Identity;

impl Identity {
    /// Create a new identity transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for Identity {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        Ok(input.clone())
    }
}
