//! Compose transform
//!
//! Chains multiple transformations together into a single pipeline.
//! Transformations are applied in order, allowing complex preprocessing workflows.

use std::any::{Any, TypeId};

use super::TransformError;

/// Trait for transformations that can be composed
///
/// This trait allows transformations to be stored in heterogeneous collections
/// and applied in sequence. It's a more flexible alternative to the generic Transform trait.
pub trait ComposableTransform {
    /// Apply the transformation using dynamic typing
    ///
    /// # Arguments
    /// * `input` - Input data as a trait object
    ///
    /// # Returns
    /// The transformed data as a trait object
    fn apply_dynamic(&self, input: Box<dyn Any>) -> Result<Box<dyn Any>, TransformError>;

    /// Get a description of the transform for debugging
    fn describe(&self) -> String {
        "ComposableTransform".to_string()
    }
}

/// Transform that composes multiple transformations into a pipeline
///
/// Applies transformations in sequence using dynamic typing.
/// This allows chaining transforms with different input/output types.
#[derive(Default)]
pub struct Compose {
    /// The sequence of transformations to apply
    transforms: Vec<Box<dyn ComposableTransform>>,
}

impl Compose {
    /// Create a new Compose transform
    ///
    /// # Arguments
    /// * `transforms` - Vector of boxed composable transformations
    pub fn new(transforms: Vec<Box<dyn ComposableTransform>>) -> Self {
        Self { transforms }
    }

    /// Add a transform to the end of the pipeline
    ///
    /// # Arguments
    /// * `transform` - The transform to add
    pub fn add_transform(&mut self, transform: Box<dyn ComposableTransform>) {
        self.transforms.push(transform);
    }

    /// Get the number of transforms in the pipeline
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl ComposableTransform for Compose {
    fn apply_dynamic(&self, input: Box<dyn Any>) -> Result<Box<dyn Any>, TransformError> {
        if self.transforms.is_empty() {
            return Err(TransformError::TransformError {
                message: "Cannot apply empty transform pipeline".to_string(),
            });
        }

        let mut current = input;

        for (i, transform) in self.transforms.iter().enumerate() {
            current = match transform.apply_dynamic(current) {
                Ok(output) => output,
                Err(e) => {
                    return Err(TransformError::TransformError {
                        message: format!(
                            "Transform {} ({}) failed: {}",
                            i,
                            transform.describe(),
                            e
                        ),
                    });
                }
            };
        }

        Ok(current)
    }

    fn describe(&self) -> String {
        format!("Compose({} transforms)", self.transforms.len())
    }
}

// Implement ComposableTransform for concrete transforms
impl ComposableTransform for super::ToTensor {
    fn apply_dynamic(&self, input: Box<dyn Any>) -> Result<Box<dyn Any>, TransformError> {
        // Check the type and downcast appropriately
        let type_id = input.as_ref().type_id();

        if type_id == TypeId::of::<Vec<f32>>() {
            let vec_f32 = input.downcast::<Vec<f32>>().unwrap();
            let result = self.apply_f32(*vec_f32)?;
            Ok(Box::new(result))
        } else if type_id == TypeId::of::<Vec<f64>>() {
            let vec_f64 = input.downcast::<Vec<f64>>().unwrap();
            let result = self.apply_f64(*vec_f64)?;
            Ok(Box::new(result))
        } else if type_id == TypeId::of::<Vec<u8>>() {
            let vec_u8 = input.downcast::<Vec<u8>>().unwrap();
            let result = self.apply_u8(*vec_u8)?;
            Ok(Box::new(result))
        } else if type_id == TypeId::of::<Vec<Vec<f32>>>() {
            let vec_vec_f32 = input.downcast::<Vec<Vec<f32>>>().unwrap();
            let result = self.apply_f32_2d(*vec_vec_f32)?;
            Ok(Box::new(result))
        } else {
            Err(TransformError::InvalidInput {
                message: "Unsupported input type for ToTensor".to_string(),
            })
        }
    }

    fn describe(&self) -> String {
        "ToTensor".to_string()
    }
}

impl ComposableTransform for super::Normalize {
    fn apply_dynamic(&self, input: Box<dyn Any>) -> Result<Box<dyn Any>, TransformError> {
        // Check the type and downcast appropriately
        let type_id = input.as_ref().type_id();

        if type_id
            == TypeId::of::<
                coeus_tensor::Tensor<
                    coeus_backend::CpuBackend,
                    coeus_storage::DenseStorage<coeus_dtype::float::Float32>,
                    coeus_dtype::float::Float32,
                >,
            >()
        {
            let tensor = input
                .downcast::<coeus_tensor::Tensor<
                    coeus_backend::CpuBackend,
                    coeus_storage::DenseStorage<coeus_dtype::float::Float32>,
                    coeus_dtype::float::Float32,
                >>()
                .unwrap();
            let result = self.apply_tensor(&tensor)?;
            Ok(Box::new(result))
        } else {
            Err(TransformError::InvalidInput {
                message: "Normalize requires tensor input".to_string(),
            })
        }
    }

    fn describe(&self) -> String {
        format!("Normalize(mean={:?}, std={:?})", self.mean(), self.std())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::{Normalize, ToTensor};
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    #[test]
    fn test_compose_empty() {
        let compose = Compose::new(vec![]);
        let input = Box::new(vec![1.0, 2.0, 3.0]);

        let result = compose.apply_dynamic(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            TransformError::TransformError { message } => {
                assert!(message.contains("empty transform pipeline"));
            }
            _ => panic!("Expected TransformError"),
        }
    }

    #[test]
    fn test_compose_single_transform() {
        let transform = ToTensor::new();
        let compose = Compose::new(vec![Box::new(transform)]);
        let input = Box::new(vec![1.0, 2.0, 3.0]);

        let result = compose.apply_dynamic(input).unwrap();

        // Should be a tensor
        let tensor = result
            .downcast::<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>()
            .unwrap();
        assert_eq!(tensor.shape().dims(), &[3]);
        assert_eq!(tensor.as_slice()[0].get(), 1.0);
        assert_eq!(tensor.as_slice()[1].get(), 2.0);
        assert_eq!(tensor.as_slice()[2].get(), 3.0);
    }

    #[test]
    fn test_compose_multiple_transforms() {
        let to_tensor = ToTensor::new();
        let normalize = Normalize::single_channel(2.0, 1.0);
        let compose = Compose::new(vec![Box::new(to_tensor), Box::new(normalize)]);
        let input = Box::new(vec![1.0, 2.0, 3.0]);

        let result = compose.apply_dynamic(input).unwrap();

        // Should be a normalized tensor: (1-2)/1 = -1, (2-2)/1 = 0, (3-2)/1 = 1
        let tensor = result
            .downcast::<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>()
            .unwrap();
        assert_eq!(tensor.shape().dims(), &[3]);
        assert!((tensor.as_slice()[0].get() - (-1.0)).abs() < 1e-6);
        assert!((tensor.as_slice()[1].get() - 0.0).abs() < 1e-6);
        assert!((tensor.as_slice()[2].get() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compose_add_transform() {
        let mut compose = Compose::default();
        assert_eq!(compose.len(), 0);
        assert!(compose.is_empty());

        compose.add_transform(Box::new(ToTensor::new()));
        assert_eq!(compose.len(), 1);
        assert!(!compose.is_empty());

        compose.add_transform(Box::new(Normalize::single_channel(0.0, 1.0)));
        assert_eq!(compose.len(), 2);
    }

    #[test]
    fn test_compose_describe() {
        let mut compose = Compose::default();
        assert_eq!(compose.describe(), "Compose(0 transforms)");

        compose.add_transform(Box::new(ToTensor::new()));
        assert_eq!(compose.describe(), "Compose(1 transforms)");
    }

    #[test]
    fn test_compose_invalid_input() {
        let normalize = Normalize::single_channel(0.0, 1.0);
        let compose = Compose::new(vec![Box::new(normalize)]);
        let input = Box::new(vec![1.0, 2.0, 3.0]); // Wrong input type for normalize

        let result = compose.apply_dynamic(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            TransformError::TransformError { message } => {
                assert!(message.contains("Normalize requires tensor input"));
            }
            _ => panic!("Expected TransformError"),
        }
    }

    #[test]
    fn test_compose_transform_failure() {
        let normalize = Normalize::new(vec![0.5, 1.0], vec![0.5, 0.5]); // 2 channels
        let compose = Compose::new(vec![Box::new(normalize)]);

        // Create 1-channel tensor (wrong shape)
        let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)],
            &[1],
        )
        .unwrap();
        let input = Box::new(tensor);

        let result = compose.apply_dynamic(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            TransformError::TransformError { message } => {
                assert!(message.contains("failed"));
            }
            _ => panic!("Expected TransformError"),
        }
    }
}
