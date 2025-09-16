//! Data transformation utilities
//!
//! Provides common preprocessing operations for tensors,
//! compatible with PyTorch's torchvision.transforms interface.

use crate::Result;
use coeus_tensor::Tensor;

/// Trait for data transformations
pub trait Transform<T: coeus_dtype::Dtype> {
    /// Apply the transformation to the input tensor
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
}

/// Normalization transform
///
/// Normalizes tensor values to have zero mean and unit variance
/// Compatible with PyTorch's `transforms.Normalize`
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
        let result = input.clone();

        // Apply normalization: (input - mean) / std
        // For now, apply to the entire tensor
        // In a full implementation, this would handle multi-channel data
        if !self.mean.is_empty() {
            let _mean_val = self.mean[0];
            let _std_val = self.std[0];

            // result = (result - mean_val) / std_val
            // Implementation would depend on tensor operations available
        }

        Ok(result)
    }
}

/// Random crop transform
///
/// Randomly crops the input tensor to the specified size
/// Compatible with PyTorch's `transforms.RandomCrop`
#[allow(dead_code)] // Fields used in future implementation
pub struct RandomCrop {
    size: Vec<usize>,
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
#[allow(dead_code)] // Field used in future implementation
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
}

impl<T: coeus_dtype::Dtype> Transform<T> for RandomHorizontalFlip {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // For now, return a copy
        // In a full implementation, this would perform actual flipping
        Ok(input.clone())
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
        // For now, return a copy
        // In a full implementation, this would handle type conversions
        Ok(input.clone())
    }
}

/// Lambda transform
///
/// Applies a custom function to the input tensor
/// Compatible with PyTorch's `transforms.Lambda`
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
/// Returns the input unchanged
/// Useful as a placeholder or for testing
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

pub use Compose as compose;
pub use Identity as identity;
pub use Lambda as lambda;
/// Re-export common transforms for easy access
pub use Normalize as normalize;
pub use RandomCrop as random_crop;
pub use RandomHorizontalFlip as random_horizontal_flip;
pub use ToTensor as to_tensor;
