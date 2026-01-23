//! CPU mean reduction primitive
//!
//! Provides optimized mean reduction for CPU execution.

use crate::DataType;

/// Mean reduction primitive for CPU backend
///
/// Computes the mean of all elements in the input slice
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Mean of all elements, or zero if input is empty
pub fn mean_primitive<T: DataType>(input: &[T]) -> T
where
    T: core::ops::Add<Output = T> + core::ops::Div<Output = T> + Default + Copy,
{
    if input.is_empty() {
        return T::default();
    }

    let mut sum = T::default();
    
    // TODO: Future SIMD optimization point
    for &val in input.iter() {
        sum = sum + val;
    }
    
    // Convert length to T for division
    let len_f64 = input.len() as f64;
    let len_t = T::from(len_f64).unwrap_or(T::one());
    
    sum / len_t
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_mean_primitive() {
        let input = [
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];

        let result = mean_primitive(&input);
        assert_eq!(result, Float32::new(2.5));
    }

    #[test]
    fn test_mean_primitive_empty() {
        let input: [Float32; 0] = [];
        let result = mean_primitive(&input);
        assert_eq!(result, Float32::new(0.0));
    }
}