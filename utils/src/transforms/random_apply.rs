//! RandomApply transform
//!
//! Conditionally applies transformations with given probability.
//! Enables stochastic data augmentation pipelines for robust training.

use std::sync::Arc;

use super::compose::ComposableTransform;
use crate::transforms::{Transform, TransformError};

/// A transform that applies sub-transforms with given probability
///
/// RandomApply enables conditional data augmentation by applying a sequence
/// of transforms with a specified probability. This is essential for
/// creating robust training pipelines where augmentation is applied randomly.
#[derive(Clone)]
pub struct RandomApply<T> {
    /// The transforms to apply conditionally
    transforms: Vec<Arc<dyn Transform<T>>>,
    /// Probability of applying the transforms (0.0 to 1.0)
    probability: f32,
    /// Random number generator seed for reproducibility
    seed: Option<u64>,
}

impl<T> RandomApply<T> {
    /// Create a new RandomApply transform
    ///
    /// # Arguments
    /// * `transforms` - Vector of transforms to apply
    /// * `probability` - Probability of applying transforms (0.0 to 1.0)
    ///
    /// # Panics
    /// Panics if probability is not in range [0.0, 1.0]
    pub fn new(transforms: Vec<Arc<dyn Transform<T>>>, probability: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0.0 and 1.0, got {}",
            probability
        );

        Self {
            transforms,
            probability,
            seed: None,
        }
    }

    /// Create a new RandomApply transform with a fixed seed
    ///
    /// # Arguments
    /// * `transforms` - Vector of transforms to apply
    /// * `probability` - Probability of applying transforms (0.0 to 1.0)
    /// * `seed` - Random seed for reproducible behavior
    ///
    /// # Panics
    /// Panics if probability is not in range [0.0, 1.0]
    pub fn with_seed(transforms: Vec<Arc<dyn Transform<T>>>, probability: f32, seed: u64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be between 0.0 and 1.0, got {}",
            probability
        );

        Self {
            transforms,
            probability,
            seed: Some(seed),
        }
    }

    /// Get the probability of applying transforms
    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// Get the random seed (if set)
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Get the number of transforms
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if there are any transforms
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Generate a random number using the configured seed or thread-local randomness
    fn random_value(&self) -> f32 {
        match self.seed {
            Some(seed) => {
                // Use a simple seeded random number generator for reproducibility
                // This is a basic linear congruential generator
                let a = 1664525;
                let c = 1013904223;
                let m = 2u64.pow(32);

                // Mix in current thread ID or some entropy to avoid deterministic sequence
                let thread_id = std::thread::current().id();
                // Convert thread ID to a hash since as_u64() is unstable
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                thread_id.hash(&mut hasher);
                let thread_seed = hasher.finish();
                let mut current_seed = seed.wrapping_add(thread_seed);

                current_seed = (current_seed.wrapping_mul(a).wrapping_add(c)) % m;
                (current_seed as f32) / (m as f32)
            }
            None => {
                // Use thread-local randomness
                rand::random::<f32>()
            }
        }
    }
}

/// Conditional transform wrapper that applies transforms based on a condition
///
/// This is a more flexible version that allows programmatic control over
/// when transforms are applied, not just probabilistic.
#[derive(Clone)]
pub struct ConditionalTransform<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    /// The transforms to apply conditionally
    transforms: Vec<Arc<dyn Transform<T>>>,
    /// Condition function that determines when to apply transforms
    condition: F,
}

impl<T, F> ConditionalTransform<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    /// Create a new ConditionalTransform
    ///
    /// # Arguments
    /// * `transforms` - Vector of transforms to apply
    /// * `condition` - Function that returns true when transforms should be applied
    pub fn new(transforms: Vec<Arc<dyn Transform<T>>>, condition: F) -> Self {
        Self {
            transforms,
            condition,
        }
    }

    /// Get the number of transforms
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if there are any transforms
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T> Transform<T, T> for RandomApply<T>
where
    T: Clone,
{
    fn apply(&self, input: T) -> Result<T, TransformError> {
        if self.transforms.is_empty() {
            return Ok(input);
        }

        // Determine whether to apply transforms
        let should_apply = self.random_value() < self.probability;

        if !should_apply {
            return Ok(input);
        }

        // Apply all transforms in sequence
        let mut result: T = input;
        for transform in &self.transforms {
            result = transform.apply(result)?;
        }

        Ok(result)
    }
}

impl<T, F> Transform<T, T> for ConditionalTransform<T, F>
where
    T: Clone,
    F: Fn(&T) -> bool + Send + Sync,
{
    fn apply(&self, input: T) -> Result<T, TransformError> {
        if self.transforms.is_empty() {
            return Ok(input);
        }

        // Check condition
        let should_apply = (self.condition)(&input);

        if !should_apply {
            return Ok(input);
        }

        // Apply all transforms in sequence
        let mut result: T = input;
        for transform in &self.transforms {
            result = transform.apply(result)?;
        }

        Ok(result)
    }
}

impl<T> ComposableTransform for RandomApply<T>
where
    T: Clone + 'static,
{
    fn apply_dynamic(
        &self,
        input: Box<dyn std::any::Any>,
    ) -> Result<Box<dyn std::any::Any>, TransformError> {
        // Downcast the input
        let input_typed = input
            .downcast::<T>()
            .map_err(|_| TransformError::InvalidInput {
                message: "RandomApply received incorrect input type".to_string(),
            })?;

        // Apply the transform
        let result = self.apply(*input_typed)?;
        Ok(Box::new(result))
    }

    fn describe(&self) -> String {
        format!(
            "RandomApply(p={:.2}, {} transforms)",
            self.probability,
            self.transforms.len()
        )
    }
}

impl<T, F> ComposableTransform for ConditionalTransform<T, F>
where
    T: Clone + 'static,
    F: Fn(&T) -> bool + Send + Sync,
{
    fn apply_dynamic(
        &self,
        input: Box<dyn std::any::Any>,
    ) -> Result<Box<dyn std::any::Any>, TransformError> {
        // Downcast the input
        let input_typed = input
            .downcast::<T>()
            .map_err(|_| TransformError::InvalidInput {
                message: "ConditionalTransform received incorrect input type".to_string(),
            })?;

        // Apply the transform
        let result = self.apply(*input_typed)?;
        Ok(Box::new(result))
    }

    fn describe(&self) -> String {
        format!("ConditionalTransform({} transforms)", self.transforms.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::Normalize;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;
    use std::sync::Arc;

    type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

    #[test]
    fn test_random_apply_creation() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> =
            vec![Arc::new(Normalize::single_channel(0.0, 1.0))];
        let random_apply = RandomApply::new(transforms, 0.5);

        assert_eq!(random_apply.probability(), 0.5);
        assert_eq!(random_apply.len(), 1);
        assert!(!random_apply.is_empty());
        assert_eq!(random_apply.seed(), None);
    }

    #[test]
    fn test_random_apply_with_seed() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> =
            vec![Arc::new(Normalize::single_channel(0.0, 1.0))];
        let random_apply = RandomApply::with_seed(transforms, 0.8, 42);

        assert_eq!(random_apply.probability(), 0.8);
        assert_eq!(random_apply.seed(), Some(42));
    }

    #[test]
    fn test_random_apply_empty_transforms() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> = vec![];
        let random_apply = RandomApply::new(transforms, 0.5);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        // Empty transform list should return input unchanged
        let result = random_apply.apply(input.clone()).unwrap();
        assert_eq!(result.as_slice()[0].get(), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_random_apply_invalid_probability_low() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> = vec![];
        let _random_apply = RandomApply::new(transforms, -0.1);
    }

    #[test]
    #[should_panic]
    fn test_random_apply_invalid_probability_high() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> = vec![];
        let _random_apply = RandomApply::new(transforms, 1.1);
    }

    #[test]
    fn test_random_apply_zero_probability() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> =
            vec![Arc::new(Normalize::single_channel(2.0, 1.0))];
        let random_apply = RandomApply::new(transforms, 0.0);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        // Zero probability should never apply transforms
        for _ in 0..10 {
            let result = random_apply.apply(input.clone()).unwrap();
            // Should remain unchanged (not normalized)
            assert_eq!(result.as_slice()[0].get(), 1.0);
            assert_eq!(result.as_slice()[1].get(), 2.0);
            assert_eq!(result.as_slice()[2].get(), 3.0);
        }
    }

    #[test]
    fn test_random_apply_probability_one() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> =
            vec![Arc::new(Normalize::single_channel(2.0, 1.0))];
        let random_apply = RandomApply::new(transforms, 1.0);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        // Probability of 1.0 should always apply transforms
        for _ in 0..10 {
            let result = random_apply.apply(input.clone()).unwrap();
            // Should be normalized: (x - 2.0) / 1.0
            assert!((result.as_slice()[0].get() - (-1.0)).abs() < 1e-6); // (1-2)/1 = -1
            assert!((result.as_slice()[1].get() - 0.0).abs() < 1e-6); // (2-2)/1 = 0
            assert!((result.as_slice()[2].get() - 1.0).abs() < 1e-6); // (3-2)/1 = 1
        }
    }

    #[test]
    fn test_random_apply_multiple_transforms() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> = vec![
            Arc::new(Normalize::single_channel(2.0, 1.0)),
            Arc::new(Normalize::single_channel(0.0, 2.0)), // Apply again with different params
        ];
        let random_apply = RandomApply::new(transforms, 1.0);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let result = random_apply.apply(input).unwrap();

        // First normalization: (x - 2.0) / 1.0
        // Second normalization: ((x - 2.0) / 1.0 - 0.0) / 2.0 = (x - 2.0) / 2.0
        assert!((result.as_slice()[0].get() - (-0.5)).abs() < 1e-6); // (1-2)/2 = -0.5
        assert!((result.as_slice()[1].get() - 0.0).abs() < 1e-6); // (2-2)/2 = 0
        assert!((result.as_slice()[2].get() - 0.5).abs() < 1e-6); // (3-2)/2 = 0.5
    }

    #[test]
    fn test_random_apply_as_composable() {
        let transforms: Vec<Arc<dyn Transform<TestTensor>>> =
            vec![Arc::new(Normalize::single_channel(0.0, 1.0))];
        let random_apply = RandomApply::new(transforms, 1.0);

        let input_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(-1.0), Float32::new(0.5)],
            &[3],
        )
        .unwrap();
        let input = Box::new(input_tensor);

        let result = random_apply.apply_dynamic(input).unwrap();
        let result_tensor = result.downcast::<TestTensor>().unwrap();

        // Should be normalized: x / 1.0 (mean=0, std=1)
        assert_eq!(result_tensor.as_slice()[0].get(), 1.0);
        assert_eq!(result_tensor.as_slice()[1].get(), -1.0);
        assert_eq!(result_tensor.as_slice()[2].get(), 0.5);
    }

    #[test]
    fn test_conditional_transform() {
        // Create a condition that applies transforms only to tensors with positive mean
        let condition = |tensor: &TestTensor| {
            let sum: f32 = tensor.as_slice().iter().map(|x: &Float32| x.get()).sum();
            let mean = sum / tensor.as_slice().len() as f32;
            mean > 0.0
        };

        let transforms: Vec<Arc<dyn Transform<TestTensor>>> =
            vec![Arc::new(Normalize::single_channel(0.0, 1.0))];

        let conditional = ConditionalTransform::new(transforms, condition);

        // Test with positive mean tensor
        let input_positive =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
                &[3], // mean = 2.0 > 0
            )
            .unwrap();

        let result_positive = conditional.apply(input_positive).unwrap();
        // Should be unchanged (condition met but normalization with mean=0, std=1 doesn't change data)
        assert_eq!(result_positive.as_slice()[0].get(), 1.0);
        assert_eq!(result_positive.as_slice()[1].get(), 2.0);
        assert_eq!(result_positive.as_slice()[2].get(), 3.0);

        // Test with negative mean tensor
        let input_negative =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                vec![Float32::new(-3.0), Float32::new(-1.0), Float32::new(-2.0)],
                &[3], // mean = -2.0 < 0
            )
            .unwrap();

        let result_negative = conditional.apply(input_negative.clone()).unwrap();
        // Should be unchanged (condition not met)
        assert_eq!(result_negative.as_slice()[0].get(), -3.0);
        assert_eq!(result_negative.as_slice()[1].get(), -1.0);
        assert_eq!(result_negative.as_slice()[2].get(), -2.0);
    }
}
