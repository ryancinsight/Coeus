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
use coeus_backend::{Backend, CpuBackend};
use std::ops::{BitAnd, BitOr, BitXor, Not};

// Dispatch impl for Tensor
impl<T: Dtype + Clone + BitAnd<Output = T>, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn bitwise_and(&self, rhs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: rhs.shape().to_vec(),
            });
        }
        let out_data = self.data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(&a, &b)| a.bitand(b))
            .collect();
        let out_shape = self.shape().to_vec();
        let backend = self.backend().clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }
}

impl<T: Dtype + Clone + BitOr<Output = T>, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn bitwise_or(&self, rhs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: rhs.shape().to_vec(),
            });
        }
        let out_data = self.data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(&a, &b)| a.bitor(b))
            .collect();
        let out_shape = self.shape().to_vec();
        let backend = self.backend().clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }

}

impl<T: Dtype + Clone + BitXor<Output = T>, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn bitwise_xor(&self, rhs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: rhs.shape().to_vec(),
            });
        }
        let out_data = self.data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(&a, &b)| a.bitxor(b))
            .collect();
        let out_shape = self.shape().to_vec();
        let backend = self.backend().clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }

}

impl<T: Dtype + Clone + Not<Output = T>, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn bitwise_not(&self) -> Result<Tensor<T, B>> {
        let out_data = self.data().iter().map(|&a| a.not()).collect();
        let out_shape = self.shape().to_vec();
        let backend = self.backend().clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_bitwise_and() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![5i32, 3, 9, 15], vec![4]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![3i32, 6, 10, 15], vec![4]).unwrap();

        let result = a.bitwise_and(&b).unwrap();
        assert_eq!(result.data(), &[1, 2, 8, 15]);
        assert_eq!(result.shape(), &[4]);
    }

    #[test]
    fn test_bitwise_or() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![5i32, 3, 9, 1], vec![4]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![3i32, 6, 10, 2], vec![4]).unwrap();

        let result = a.bitwise_or(&b).unwrap();
        assert_eq!(result.data(), &[7, 7, 11, 3]);
    }

    #[test]
    fn test_bitwise_xor() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![5i32, 3, 9, 15], vec![4]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![3i32, 6, 10, 15], vec![4]).unwrap();

        let result = a.bitwise_xor(&b).unwrap();
        assert_eq!(result.data(), &[6, 5, 3, 0]);
    }

    #[test]
    fn test_bitwise_not() {
        let tensor = Tensor::from_vec(CpuBackend::default(), vec![1i32, -1, 0, 15], vec![4]).unwrap();

        let result = tensor.bitwise_not().unwrap();
        assert_eq!(result.data(), &[-2, 0, -1, -16]);
    }

    #[test]
    fn test_bitwise_shape_mismatch() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![1i32, 2], vec![2]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![3i32, 4, 5], vec![3]).unwrap();

        assert!(a.bitwise_and(&b).is_err());
        assert!(a.bitwise_or(&b).is_err());
        assert!(a.bitwise_xor(&b).is_err());
    }

    #[test]
    fn test_bitwise_u8() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![0x0Fu8, 0xF0u8], vec![2]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![0xFFu8, 0x0Fu8], vec![2]).unwrap();

        let and_result = a.bitwise_and(&b).unwrap();
        assert_eq!(and_result.data(), &[0x0F, 0x00]);

        let or_result = a.bitwise_or(&b).unwrap();
        assert_eq!(or_result.data(), &[0xFF, 0xFF]);

        let xor_result = a.bitwise_xor(&b).unwrap();
        assert_eq!(xor_result.data(), &[0xF0, 0xFF]);
    }
}
