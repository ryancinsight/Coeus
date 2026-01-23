//! CPU max/min reduction primitives
//!
//! Provides optimized max and min reduction for CPU execution.

use crate::DataType;

/// Max reduction primitive for CPU backend
///
/// Finds the maximum element in the input slice
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Maximum element, or error if input is empty
pub fn max_primitive<T: DataType>(input: &[T]) -> crate::Result<T>
where
    T: PartialOrd + Copy,
{
    if input.is_empty() {
        return Err(crate::BackendError::InvalidInput(
            "Cannot find max of empty slice".to_string(),
        ));
    }

    let mut max_val = input[0];
    
    // TODO: Future SIMD optimization point
    for &val in input.iter().skip(1) {
        if val > max_val {
            max_val = val;
        }
    }
    
    Ok(max_val)
}

/// Min reduction primitive for CPU backend
///
/// Finds the minimum element in the input slice
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Minimum element, or error if input is empty
pub fn min_primitive<T: DataType>(input: &[T]) -> crate::Result<T>
where
    T: PartialOrd + Copy,
{
    if input.is_empty() {
        return Err(crate::BackendError::InvalidInput(
            "Cannot find min of empty slice".to_string(),
        ));
    }

    let mut min_val = input[0];
    
    // TODO: Future SIMD optimization point
    for &val in input.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
    }
    
    Ok(min_val)
}

/// Argmax primitive for CPU backend
///
/// Finds the index of the maximum element in the input slice
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Index of maximum element, or error if input is empty
pub fn argmax_primitive<T: DataType>(input: &[T]) -> crate::Result<usize>
where
    T: PartialOrd + Copy,
{
    if input.is_empty() {
        return Err(crate::BackendError::InvalidInput(
            "Cannot find argmax of empty slice".to_string(),
        ));
    }

    let mut max_idx = 0;
    let mut max_val = input[0];
    
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    Ok(max_idx)
}

/// Argmin primitive for CPU backend
///
/// Finds the index of the minimum element in the input slice
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Index of minimum element, or error if input is empty
pub fn argmin_primitive<T: DataType>(input: &[T]) -> crate::Result<usize>
where
    T: PartialOrd + Copy,
{
    if input.is_empty() {
        return Err(crate::BackendError::InvalidInput(
            "Cannot find argmin of empty slice".to_string(),
        ));
    }

    let mut min_idx = 0;
    let mut min_val = input[0];
    
    for (i, &val) in input.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    
    Ok(min_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_max_primitive() {
        let input = [
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(2.0),
            Float32::new(3.0),
        ];

        let result = max_primitive(&input).unwrap();
        assert_eq!(result, Float32::new(4.0));
    }

    #[test]
    fn test_min_primitive() {
        let input = [
            Float32::new(3.0),
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(2.0),
        ];

        let result = min_primitive(&input).unwrap();
        assert_eq!(result, Float32::new(1.0));
    }

    #[test]
    fn test_argmax_primitive() {
        let input = [
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(2.0),
            Float32::new(3.0),
        ];

        let result = argmax_primitive(&input).unwrap();
        assert_eq!(result, 1); // Index of 4.0
    }

    #[test]
    fn test_argmin_primitive() {
        let input = [
            Float32::new(3.0),
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(2.0),
        ];

        let result = argmin_primitive(&input).unwrap();
        assert_eq!(result, 1); // Index of 1.0
    }

    #[test]
    fn test_max_primitive_empty() {
        let input: [Float32; 0] = [];
        let result = max_primitive(&input);
        assert!(result.is_err());
    }
}