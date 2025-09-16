//! Matrix operations

use crate::{FloatDtype, Result, Tensor, TensorError};

/// Matrix multiplication
pub fn matmul<T: FloatDtype>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(TensorError::InvalidOperation {
            message: "Matrix multiplication requires 2D tensors".to_string(),
        });
    }

    if a.shape()[1] != b.shape()[0] {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a.shape()[0], b.shape()[1]],
            actual: vec![a.shape()[0], a.shape()[1]],
        });
    }

    let m = a.shape()[0];
    let n = b.shape()[1];
    let k = a.shape()[1];

    let mut result_data = vec![T::zero(); m * n];

    // Naive matrix multiplication (can be optimized later)
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                let a_idx = i * k + p;
                let b_idx = p * n + j;
                sum = sum + a.data()[a_idx] * b.data()[b_idx];
            }
            result_data[i * n + j] = sum;
        }
    }

    let mut result = Tensor::from_vec(result_data, vec![m, n]);

    // Handle gradient computation
    if a.requires_grad() || b.requires_grad() {
        result.set_requires_grad(true);
        // Note: Graph integration is handled by tensor methods, not free functions
    }

    Ok(result)
}

/// Broadcasting utilities
pub struct Broadcast;

impl Broadcast {
    /// Check if two shapes are broadcastable
    pub fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Get the broadcasted shape
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);
        let mut result = vec![0; max_len];

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            result[max_len - 1 - i] = dim1.max(dim2);
        }

        result
    }
}
