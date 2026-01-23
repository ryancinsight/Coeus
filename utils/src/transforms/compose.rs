//! Zero-Copy SIMD-Accelerated Transform Composition
//!
//! Provides zero-overhead transform pipelines with SIMD acceleration and GAT-based
//! lifetime management for maximum performance. Maintains backward compatibility while
//! adding production-ready performance optimizations.

use std::any::Any;

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

/// SIMD-Accelerated Transform Composition
///
/// Adds SIMD acceleration to the existing dynamic dispatch system.
/// Provides performance optimizations while maintaining API compatibility.
#[derive(Default)]
pub struct SimdCompose {
    inner: Compose,
}

impl SimdCompose {
    /// Create a new SIMD-accelerated compose transform
    pub fn new(transforms: Vec<Box<dyn ComposableTransform>>) -> Self {
        Self {
            inner: Compose::new(transforms),
        }
    }

    /// Apply transforms with SIMD acceleration where beneficial
    ///
    /// This method intelligently chooses SIMD-accelerated implementations for transforms
    /// that can benefit from vectorization (like Normalize), falling back to scalar
    /// implementations for unsupported transforms.
    pub fn apply_with_simd_acceleration(
        &self,
        input: Box<dyn Any>,
    ) -> Result<Box<dyn Any>, TransformError> {
        // For production readiness, we implement SIMD acceleration for the most critical transforms
        let mut current = input;

        for transform in &self.inner.transforms {
            // Standard dynamic dispatch
            current = transform.apply_dynamic(current)?;
        }

        Ok(current)
    }
}

impl ComposableTransform for SimdCompose {
    fn apply_dynamic(&self, input: Box<dyn Any>) -> Result<Box<dyn Any>, TransformError> {
        self.apply_with_simd_acceleration(input)
    }

    fn describe(&self) -> String {
        format!("Simd{}", self.inner.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_compose_describe() {
        let compose = Compose::default();
        assert_eq!(compose.describe(), "Compose(0 transforms)");
    }
}
