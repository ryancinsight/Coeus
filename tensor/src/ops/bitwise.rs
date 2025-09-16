//! Bitwise operations for tensors
//!
//! This module provides element-wise bitwise operations on integer tensors,
//! following PyTorch conventions for bitwise operations.
//!
//! ## Supported Operations
//!
//! - **Bitwise AND**: `tensor.bitwise_and(&other)` or `tensor & other`
//! - **Bitwise OR**: `tensor.bitwise_or(&other)` or `tensor | other`
//! - **Bitwise XOR**: `tensor.bitwise_xor(&other)` or `tensor ^ other`
//! - **Bitwise NOT**: `tensor.bitwise_not()` or `!tensor`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use coeus_tensor::Tensor;
//!
//! let a = Tensor::from_vec(vec![1i32, 3, 5, 7], vec![4]);  // Binary: 001, 011, 101, 111
//! let b = Tensor::from_vec(vec![2i32, 3, 6, 7], vec![4]);  // Binary: 010, 011, 110, 111
//!
//! let and_result = a.bitwise_and(&b); // [0, 3, 4, 7]
//! let or_result = a.bitwise_or(&b);   // [3, 3, 7, 7]
//! let xor_result = a.bitwise_xor(&b); // [3, 0, 3, 0]
//! let not_result = a.bitwise_not();   // [-2, -4, -6, -8]
//! ```
//!
//! ## Type Requirements
//!
//! Bitwise operations require integer tensor types. Floating-point types are not supported.
//!
//! ## References
//!
//! - [PyTorch Bitwise Operations](https://pytorch.org/docs/stable/torch.html#bitwise-ops)
//! - [Rust Integer Types](https://doc.rust-lang.org/std/primitive/index.html)

use crate::{Dtype, Result, Tensor, TensorError};

/// Bitwise AND operation on integer tensors
///
/// Performs element-wise bitwise AND between corresponding elements of the input tensors.
///
/// # Arguments
/// * `a` - First integer tensor
/// * `b` - Second integer tensor
///
/// # Returns
/// Integer tensor with the same shape as the inputs
///
/// # Example
/// ```rust,ignore
/// let a = Tensor::from_vec(vec![5i32, 3], vec![2]);  // Binary: 101, 011
/// let b = Tensor::from_vec(vec![3i32, 6], vec![2]);  // Binary: 011, 110
/// let result = bitwise_and(&a, &b);                  // [1, 2] (001, 010)
/// ```
pub fn bitwise_and<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Dtype + std::ops::BitAnd<Output = T> + Copy,
{
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| *x & *y)
        .collect();

    Ok(Tensor::from_vec(result_data, a.shape().to_vec()))
}

/// Bitwise OR operation on integer tensors
///
/// Performs element-wise bitwise OR between corresponding elements of the input tensors.
///
/// # Arguments
/// * `a` - First integer tensor
/// * `b` - Second integer tensor
///
/// # Returns
/// Integer tensor with the same shape as the inputs
pub fn bitwise_or<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Dtype + std::ops::BitOr<Output = T> + Copy,
{
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| *x | *y)
        .collect();

    Ok(Tensor::from_vec(result_data, a.shape().to_vec()))
}

/// Bitwise XOR operation on integer tensors
///
/// Performs element-wise bitwise XOR between corresponding elements of the input tensors.
///
/// # Arguments
/// * `a` - First integer tensor
/// * `b` - Second integer tensor
///
/// # Returns
/// Integer tensor with the same shape as the inputs
pub fn bitwise_xor<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Dtype + std::ops::BitXor<Output = T> + Copy,
{
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }

    let result_data: Vec<T> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| *x ^ *y)
        .collect();

    Ok(Tensor::from_vec(result_data, a.shape().to_vec()))
}

/// Bitwise NOT operation on integer tensor
///
/// Performs element-wise bitwise NOT (one's complement) on each element of the input tensor.
///
/// # Arguments
/// * `tensor` - Integer tensor
///
/// # Returns
/// Integer tensor with the same shape as the input
///
/// # Note
/// For signed integers, this performs two's complement negation minus one.
/// Use with caution as it may produce unexpected results for signed types.
pub fn bitwise_not<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: Dtype + std::ops::Not<Output = T> + Copy,
{
    let result_data: Vec<T> = tensor.data().iter().map(|x| !*x).collect();
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_bitwise_and() {
        let a = Tensor::from_vec(vec![5i32, 3, 9, 15], vec![4]); // Binary: 0101, 0011, 1001, 1111
        let b = Tensor::from_vec(vec![3i32, 6, 10, 15], vec![4]); // Binary: 0011, 0110, 1010, 1111

        let result = bitwise_and(&a, &b).unwrap();
        assert_eq!(result.data(), &[1, 2, 8, 15]); // Binary: 0001, 0010, 1000, 1111
        assert_eq!(result.shape(), &[4]);
    }

    #[test]
    fn test_bitwise_or() {
        let a = Tensor::from_vec(vec![5i32, 3, 9, 1], vec![4]); // Binary: 0101, 0011, 1001, 0001
        let b = Tensor::from_vec(vec![3i32, 6, 10, 2], vec![4]); // Binary: 0011, 0110, 1010, 0010

        let result = bitwise_or(&a, &b).unwrap();
        assert_eq!(result.data(), &[7, 7, 11, 3]); // Binary: 0111, 0111, 1011, 0011
    }

    #[test]
    fn test_bitwise_xor() {
        let a = Tensor::from_vec(vec![5i32, 3, 9, 15], vec![4]); // Binary: 0101, 0011, 1001, 1111
        let b = Tensor::from_vec(vec![3i32, 6, 10, 15], vec![4]); // Binary: 0011, 0110, 1010, 1111

        let result = bitwise_xor(&a, &b).unwrap();
        assert_eq!(result.data(), &[6, 5, 3, 0]); // Binary: 0110, 0101, 0011, 0000
    }

    #[test]
    fn test_bitwise_not() {
        let tensor = Tensor::from_vec(vec![1i32, -1, 0, 15], vec![4]);

        let result = bitwise_not(&tensor);
        // Note: Two's complement behavior for signed integers
        assert_eq!(result.data(), &[-2, 0, -1, -16]);
    }

    #[test]
    fn test_bitwise_shape_mismatch() {
        let a = Tensor::from_vec(vec![1i32, 2], vec![2]);
        let b = Tensor::from_vec(vec![3i32, 4, 5], vec![3]);

        assert!(bitwise_and(&a, &b).is_err());
        assert!(bitwise_or(&a, &b).is_err());
        assert!(bitwise_xor(&a, &b).is_err());
    }

    #[test]
    fn test_bitwise_u8() {
        let a = Tensor::from_vec(vec![0x0Fu8, 0xF0u8], vec![2]);
        let b = Tensor::from_vec(vec![0xFFu8, 0x0Fu8], vec![2]);

        let and_result = bitwise_and(&a, &b).unwrap();
        assert_eq!(and_result.data(), &[0x0F, 0x00]);

        let or_result = bitwise_or(&a, &b).unwrap();
        assert_eq!(or_result.data(), &[0xFF, 0xFF]);

        let xor_result = bitwise_xor(&a, &b).unwrap();
        assert_eq!(xor_result.data(), &[0xF0, 0xFF]);
    }
}
