//! General-purpose data transformations
//!
//! This module provides general transforms that can be applied to any data type,
//! compatible with PyTorch's transforms interface.

use crate::{Result, Transform};
use coeus_tensor::Tensor;
use rand::prelude::SliceRandom;
use rand::Rng;

/// Random apply transform
///
/// Applies a transform with given probability
/// Compatible with PyTorch's `transforms.RandomApply`
pub struct RandomApply<T: coeus_dtype::Dtype> {
    transforms: Vec<Box<dyn Transform<T>>>,
    p: f64,
}

impl<T: coeus_dtype::Dtype> RandomApply<T> {
    /// Create a new random apply transform
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability must be between 0 and 1"
        );
        Self { transforms, p }
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for RandomApply<T> {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Generate random number to decide whether to apply transforms
        let mut rng = rand::thread_rng();
        let rand_val: f64 = rng.gen();

        if rand_val < self.p {
            // Apply all transforms in sequence
            let mut result = input.clone();
            for transform in &self.transforms {
                result = transform.transform(&result)?;
            }
            Ok(result)
        } else {
            Ok(input.clone())
        }
    }
}

/// Random choice transform
///
/// Randomly selects one transform from a list and applies it
/// Compatible with PyTorch's `transforms.RandomChoice`
pub struct RandomChoice<T: coeus_dtype::Dtype> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: coeus_dtype::Dtype> RandomChoice<T> {
    /// Create a new random choice transform
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for RandomChoice<T> {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.transforms.is_empty() {
            return Ok(input.clone());
        }

        // Randomly select one transform
        let mut rng = rand::thread_rng();
        let choice_idx = rng.gen_range(0..self.transforms.len());

        self.transforms[choice_idx].transform(input)
    }
}

/// Random order transform
///
/// Applies transforms in random order
/// Compatible with PyTorch's `transforms.RandomOrder`
pub struct RandomOrder<T: coeus_dtype::Dtype> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: coeus_dtype::Dtype> RandomOrder<T> {
    /// Create a new random order transform
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }
}

impl<T: coeus_dtype::Dtype> Transform<T> for RandomOrder<T> {
    fn transform(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.transforms.is_empty() {
            return Ok(input.clone());
        }

        // Create a random permutation of transform indices
        let mut indices: Vec<usize> = (0..self.transforms.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        // Apply transforms in random order
        let mut result = input.clone();
        for &idx in &indices {
            result = self.transforms[idx].transform(&result)?;
        }

        Ok(result)
    }
}
