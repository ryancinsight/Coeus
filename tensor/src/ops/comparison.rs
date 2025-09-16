//! Comparison operations for tensors
//!
//! This module provides element-wise comparison operations between tensors,
//! returning boolean tensors that can be used for masking and conditional operations.
//!
//! ## Supported Operations
//!
//! - **Equality**: `tensor.eq(&other)` or `tensor == other`
//! - **Inequality**: `tensor.ne(&other)` or `tensor != other`
//! - **Less than**: `tensor.lt(&other)`
//! - **Less than or equal**: `tensor.le(&other)`
//! - **Greater than**: `tensor.gt(&other)`
//! - **Greater than or equal**: `tensor.ge(&other)`
//!
//! ## Boolean Operations
//!
//! - **Logical AND**: `tensor.logical_and(&other)`
//! - **Logical OR**: `tensor.logical_or(&other)`
//! - **Logical NOT**: `tensor.logical_not()`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use coeus_tensor::Tensor;
//!
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
//! let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);
//!
//! let mask = a.lt(&b); // [true, false, false]
//! let equal = a.eq(&b); // [false, true, false]
//!
//! // Use boolean tensors for masking
//! let masked = a.where_cond(&mask, &Tensor::zeros(vec![3]));
//! ```
//!
//! ## Broadcasting
//!
//! Comparison operations support broadcasting following NumPy/PyTorch conventions.
//!
//! ## References
//!
//! - [PyTorch Comparison Operations](https://pytorch.org/docs/stable/torch.html#comparison-ops)
//! - [NumPy Comparison Operations](https://numpy.org/doc/stable/reference/routines.logic.html)

use crate::{Dtype, Result, Tensor, TensorError};

/// Element-wise equality comparison
///
/// Returns a boolean vector where each element is true if the corresponding
/// elements in the input tensors are equal.
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Boolean vector with the same length as the input tensors
///
/// # Example
/// ```rust,ignore
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
/// let b = Tensor::from_vec(vec![1.0, 3.0, 3.0], vec![3]);
/// let result = eq(&a, &b); // [true, false, true]
/// ```
pub fn eq<T: Dtype + PartialEq>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<bool> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x == y)
        .collect();

    Ok(result_data)
}

/// Element-wise inequality comparison
///
/// Returns a boolean tensor where each element is true if the corresponding
/// elements in the input tensors are not equal.
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Boolean tensor with the same shape as the broadcasted inputs
pub fn ne<T: Dtype + PartialEq>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<bool> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x != y)
        .collect();

    Ok(result_data)
}

/// Element-wise less than comparison
///
/// Returns a boolean tensor where each element is true if the corresponding
/// element in the first tensor is less than the corresponding element in the second tensor.
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Boolean tensor with the same shape as the broadcasted inputs
pub fn lt<T: Dtype + PartialOrd>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<bool> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x < y)
        .collect();

    Ok(result_data)
}

/// Element-wise less than or equal comparison
///
/// Returns a boolean tensor where each element is true if the corresponding
/// element in the first tensor is less than or equal to the corresponding element in the second tensor.
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Boolean tensor with the same shape as the broadcasted inputs
pub fn le<T: Dtype + PartialOrd>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<bool> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x <= y)
        .collect();

    Ok(result_data)
}

/// Element-wise greater than comparison
///
/// Returns a boolean tensor where each element is true if the corresponding
/// element in the first tensor is greater than the corresponding element in the second tensor.
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Boolean tensor with the same shape as the broadcasted inputs
pub fn gt<T: Dtype + PartialOrd>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<bool> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x > y)
        .collect();

    Ok(result_data)
}

/// Element-wise greater than or equal comparison
///
/// Returns a boolean tensor where each element is true if the corresponding
/// element in the first tensor is greater than or equal to the corresponding element in the second tensor.
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Boolean tensor with the same shape as the broadcasted inputs
pub fn ge<T: Dtype + PartialOrd>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Vec<bool>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<bool> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x >= y)
        .collect();

    Ok(result_data)
}

/// Logical AND operation on boolean vectors
///
/// Returns a boolean vector where each element is the logical AND of the
/// corresponding elements in the input vectors.
///
/// # Arguments
/// * `a` - First boolean vector
/// * `b` - Second boolean vector
///
/// # Returns
/// Boolean vector with the same length as the inputs
pub fn logical_and(a: &[bool], b: &[bool]) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    let result_data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| *x && *y).collect();

    Ok(result_data)
}

/// Logical OR operation on boolean vectors
///
/// Returns a boolean vector where each element is the logical OR of the
/// corresponding elements in the input vectors.
///
/// # Arguments
/// * `a` - First boolean vector
/// * `b` - Second boolean vector
///
/// # Returns
/// Boolean vector with the same length as the inputs
pub fn logical_or(a: &[bool], b: &[bool]) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    let result_data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| *x || *y).collect();

    Ok(result_data)
}

/// Logical XOR operation on boolean tensors
///
/// Returns a boolean tensor where each element is the logical XOR of the
/// corresponding elements in the input tensors.
///
/// # Arguments
/// * `a` - First boolean tensor
/// * `b` - Second boolean tensor
///
/// # Returns
/// Boolean tensor with the same shape as the broadcasted inputs
pub fn logical_xor(a: &[bool], b: &[bool]) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    let result_data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| *x ^ *y).collect();

    Ok(result_data)
}

/// Logical NOT operation on boolean vector
///
/// Returns a boolean vector where each element is the logical NOT of the
/// corresponding element in the input vector.
///
/// # Arguments
/// * `tensor` - Boolean vector
///
/// # Returns
/// Boolean vector with the same length as the input
pub fn logical_not(tensor: &[bool]) -> Vec<bool> {
    tensor.iter().map(|x| !*x).collect()
}

/// Conditional selection based on boolean mask
///
/// Returns a tensor where elements are selected from `on_true` where the mask is true,
/// and from `on_false` where the mask is false.
///
/// # Arguments
/// * `mask` - Boolean vector used for selection
/// * `on_true` - Tensor to select from when mask is true
/// * `on_false` - Tensor to select from when mask is false
///
/// # Returns
/// Tensor with the same shape as the inputs, containing selected values
pub fn where_cond<T: Dtype + Clone>(
    mask: &[bool],
    on_true: &Tensor<T>,
    on_false: &Tensor<T>,
) -> Result<Tensor<T>> {
    if mask.len() != on_true.numel() || mask.len() != on_false.numel() {
        return Err(TensorError::ShapeMismatch {
            expected: vec![mask.len()],
            actual: vec![on_true.numel()],
        });
    }

    let result_data: Vec<T> = mask
        .iter()
        .zip(on_true.data().iter())
        .zip(on_false.data().iter())
        .map(|((m, t), f)| if *m { *t } else { *f })
        .collect();

    Ok(Tensor::from_vec(result_data, on_true.shape().to_vec()))
}

/// Test if any element is true
///
/// Returns true if any element in the boolean vector is true.
///
/// # Arguments
/// * `tensor` - Boolean vector
///
/// # Returns
/// True if any element is true, false otherwise
pub fn any(tensor: &[bool]) -> bool {
    tensor.iter().any(|x| *x)
}

/// Test if all elements are true
///
/// Returns true if all elements in the boolean vector are true.
///
/// # Arguments
/// * `tensor` - Boolean vector
///
/// # Returns
/// True if all elements are true, false otherwise
pub fn all(tensor: &[bool]) -> bool {
    tensor.iter().all(|x| *x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_eq() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![1.0, 3.0, 3.0], vec![3]);

        let result = eq(&a, &b).unwrap();
        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn test_lt() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]);

        let result = lt(&a, &b).unwrap();
        assert_eq!(result, vec![true, false, false]);
    }

    #[test]
    fn test_logical_and() {
        let a = &[true, true, false, false];
        let b = &[true, false, true, false];

        let result = logical_and(a, b).unwrap();
        assert_eq!(result, vec![true, false, false, false]);
    }

    #[test]
    fn test_logical_xor() {
        let a = &[true, true, false, false];
        let b = &[true, false, true, false];

        let result = logical_xor(a, b).unwrap();
        assert_eq!(result, vec![false, true, true, false]);
    }

    #[test]
    fn test_logical_not() {
        let tensor = &[true, false, true];
        let result = logical_not(tensor);
        assert_eq!(result, vec![false, true, false]);
    }

    #[test]
    fn test_where_cond() {
        let mask = &[true, false, true];
        let on_true = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let on_false = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);

        let result = where_cond(mask, &on_true, &on_false).unwrap();
        assert_eq!(result.data(), &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_any_all() {
        let all_true = &[true, true, true];
        let has_false = &[true, false, true];
        let all_false = &[false, false, false];

        assert!(all(all_true));
        assert!(!all(has_false));
        assert!(!all(all_false));

        assert!(any(all_true));
        assert!(any(has_false));
        assert!(!any(all_false));
    }
}
