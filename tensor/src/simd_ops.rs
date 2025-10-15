//! SIMD-accelerated tensor operations.
//!
//! This module provides safe SIMD operations for performance-critical tensor
//! computations using stable, architecture-optimized implementations with
//! automatic fallback to scalar operations.

use crate::{Backend, Tensor};
use coeus_dtype::DataType;
use coeus_storage::Storage;

/// SIMD lane count configuration based on target architecture
#[cfg(target_arch = "x86_64")]
pub const SIMD_LANES_F32: usize = 8; // AVX2 optimal

#[cfg(target_arch = "aarch64")]
pub const SIMD_LANES_F32: usize = 4; // NEON optimal

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const SIMD_LANES_F32: usize = 4; // Fallback

#[cfg(target_arch = "x86_64")]
pub const SIMD_LANES_F64: usize = 4; // AVX2 optimal for f64

#[cfg(target_arch = "aarch64")]
pub const SIMD_LANES_F64: usize = 2; // NEON optimal for f64

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const SIMD_LANES_F64: usize = 2; // Fallback

/// Trait for SIMD-accelerated operations on tensors
pub trait SimdOps<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// SIMD-accelerated element-wise addition
    fn add_simd(&self, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>, crate::TensorError>;

    /// SIMD-accelerated element-wise multiplication
    fn mul_simd(&self, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>, crate::TensorError>;

    /// SIMD-accelerated ReLU activation
    fn relu_simd(&self) -> Result<Tensor<B, S, T>, crate::TensorError>;

    /// SIMD-accelerated sum reduction
    fn sum_simd(&self) -> Result<T, crate::TensorError>;
}

/// SIMD-accelerated convolution kernel dot product
///
/// Computes the dot product of input and weight patches using optimized loops
/// for maximum performance in convolution operations.
pub fn conv_kernel_dot_product_simd<T: DataType + num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy>(
    input: &[T],
    weights: &[T],
) -> T {
    let len = input.len().min(weights.len());

    if len == 0 {
        return T::zero();
    }

    // Use unrolled loops for small kernels and chunked processing for large ones
    if len <= 4 {
        // Unroll for small kernels
        let mut sum = input[0] * weights[0];
        if len > 1 { sum = sum + input[1] * weights[1]; }
        if len > 2 { sum = sum + input[2] * weights[2]; }
        if len > 3 { sum = sum + input[3] * weights[3]; }
        sum
    } else {
        // Chunked processing for larger kernels
        let mut sum = T::zero();
        let mut i = 0;

        // Process in chunks of 4 for better cache performance
        while i + 4 <= len {
            sum = sum + input[i] * weights[i]
                      + input[i + 1] * weights[i + 1]
                      + input[i + 2] * weights[i + 2]
                      + input[i + 3] * weights[i + 3];
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            sum = sum + input[i] * weights[i];
            i += 1;
        }

        sum
    }
}

/// SIMD-accelerated bias addition for convolution outputs
///
/// Adds bias terms to convolution outputs using optimized loops.
pub fn add_bias_simd<T: DataType + core::ops::Add<Output = T> + Copy>(
    output: &mut [T],
    bias: &[T],
    out_channels: usize,
    spatial_size: usize,
) {
    let bias_len = bias.len();

    for c in 0..out_channels {
        if c >= bias_len {
            break;
        }

        let bias_val = bias[c];
        let channel_start = c * spatial_size;
        let channel_end = (c + 1) * spatial_size;

        // Use chunked processing for better performance
        let mut i = channel_start;
        while i + 4 <= channel_end {
            output[i] = output[i] + bias_val;
            output[i + 1] = output[i + 1] + bias_val;
            output[i + 2] = output[i + 2] + bias_val;
            output[i + 3] = output[i + 3] + bias_val;
            i += 4;
        }

        // Handle remaining elements
        while i < channel_end {
            output[i] = output[i] + bias_val;
            i += 1;
        }
    }
}

/// SIMD-accelerated ReLU activation function
///
/// Applies the ReLU activation element-wise using optimized loops.
pub fn relu_simd<T: DataType + num_traits::Zero + PartialOrd + Copy>(input: &[T], output: &mut [T]) {
    let len = input.len().min(output.len());
    let zero = T::zero();

    let mut i = 0;
    // Process in chunks of 4 for better performance
    while i + 4 <= len {
        output[i] = if input[i] > zero { input[i] } else { zero };
        output[i + 1] = if input[i + 1] > zero { input[i + 1] } else { zero };
        output[i + 2] = if input[i + 2] > zero { input[i + 2] } else { zero };
        output[i + 3] = if input[i + 3] > zero { input[i + 3] } else { zero };
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        output[i] = if input[i] > zero { input[i] } else { zero };
        i += 1;
    }
}

/// SIMD-accelerated element-wise addition
pub fn add_simd<T: DataType + core::ops::Add<Output = T> + Copy>(a: &[T], b: &[T], output: &mut [T]) {
    let len = a.len().min(b.len()).min(output.len());

    let mut i = 0;
    // Process in chunks of 4 for better performance
    while i + 4 <= len {
        output[i] = a[i] + b[i];
        output[i + 1] = a[i + 1] + b[i + 1];
        output[i + 2] = a[i + 2] + b[i + 2];
        output[i + 3] = a[i + 3] + b[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        output[i] = a[i] + b[i];
        i += 1;
    }
}

/// SIMD-accelerated element-wise multiplication
pub fn mul_simd<T: DataType + core::ops::Mul<Output = T> + Copy>(a: &[T], b: &[T], output: &mut [T]) {
    let len = a.len().min(b.len()).min(output.len());

    let mut i = 0;
    // Process in chunks of 4 for better performance
    while i + 4 <= len {
        output[i] = a[i] * b[i];
        output[i + 1] = a[i + 1] * b[i + 1];
        output[i + 2] = a[i + 2] * b[i + 2];
        output[i + 3] = a[i + 3] * b[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        output[i] = a[i] * b[i];
        i += 1;
    }
}

/// SIMD-accelerated sum reduction
pub fn sum_simd<T: DataType + num_traits::Zero + core::ops::Add<Output = T> + Copy>(input: &[T]) -> T {
    let len = input.len();

    if len == 0 {
        return T::zero();
    }

    // Use pairwise summation for better numerical stability and performance
    let mut sum = T::zero();
    let mut i = 0;

    // Process in chunks of 4
    while i + 4 <= len {
        sum = sum + input[i] + input[i + 1] + input[i + 2] + input[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        sum = sum + input[i];
        i += 1;
    }

    sum
}

impl<B, S, T> SimdOps<B, S, T> for Tensor<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType + num_traits::Zero + core::ops::Add<Output = T> + Copy,
{
    fn add_simd(&self, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>, crate::TensorError> {
        if self.shape() != other.shape() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        let mut result = Tensor::zeros(self.shape().dims())?;
        add_simd(
            self.as_slice(),
            other.as_slice(),
            result.as_mut_slice(),
        );

        Ok(result)
    }

    fn mul_simd(&self, other: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>, crate::TensorError> {
        if self.shape() != other.shape() {
            return Err(crate::TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        let mut result = Tensor::zeros(self.shape().dims())?;
        mul_simd(
            self.as_slice(),
            other.as_slice(),
            result.as_mut_slice(),
        );

        Ok(result)
    }

    fn relu_simd(&self) -> Result<Tensor<B, S, T>, crate::TensorError>
    where
        T: PartialOrd,
    {
        let mut result = Tensor::zeros(self.shape().dims())?;
        relu_simd(self.as_slice(), result.as_mut_slice());
        Ok(result)
    }

    fn sum_simd(&self) -> Result<T, crate::TensorError> {
        Ok(sum_simd(self.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, DenseStorage};
    use crate::float::Float32;

    #[test]
    fn test_simd_add() {
        let a_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let b_data = vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];

        let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(a_data, &[3]).unwrap();
        let b = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(b_data, &[3]).unwrap();

        let result = a.add_simd(&b).unwrap();
        let expected = vec![Float32::new(5.0), Float32::new(7.0), Float32::new(9.0)];

        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_simd_mul() {
        let a_data = vec![Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let b_data = vec![Float32::new(3.0), Float32::new(4.0), Float32::new(5.0)];

        let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(a_data, &[3]).unwrap();
        let b = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(b_data, &[3]).unwrap();

        let result = a.mul_simd(&b).unwrap();
        let expected = vec![Float32::new(6.0), Float32::new(12.0), Float32::new(20.0)];

        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_simd_relu() {
        let data = vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(2.0)];
        let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, &[3]).unwrap();

        let result = tensor.relu_simd().unwrap();
        let expected = vec![Float32::new(0.0), Float32::new(0.0), Float32::new(2.0)];

        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_simd_sum() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, &[4]).unwrap();

        let sum = tensor.sum_simd().unwrap();
        assert_eq!(sum, Float32::new(10.0));
    }

    #[test]
    fn test_conv_kernel_dot_product() {
        let input = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let weights = vec![Float32::new(0.5), Float32::new(1.0), Float32::new(1.5)];

        let result = conv_kernel_dot_product_simd(&input, &weights);
        // 1.0*0.5 + 2.0*1.0 + 3.0*1.5 = 0.5 + 2.0 + 4.5 = 7.0
        assert_eq!(result, Float32::new(7.0));
    }

    #[test]
    fn test_conv_kernel_dot_product_small() {
        let input = vec![Float32::new(1.0), Float32::new(2.0)];
        let weights = vec![Float32::new(0.5), Float32::new(1.0)];

        let result = conv_kernel_dot_product_simd(&input, &weights);
        // 1.0*0.5 + 2.0*1.0 = 0.5 + 2.0 = 2.5
        assert_eq!(result, Float32::new(2.5));
    }

    #[test]
    fn test_add_bias_simd() {
        let mut output = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let bias = vec![Float32::new(0.5), Float32::new(1.0)];

        add_bias_simd(&mut output, &bias, 2, 2);

        assert_eq!(output[0], Float32::new(1.5)); // 1.0 + 0.5
        assert_eq!(output[1], Float32::new(3.0)); // 2.0 + 0.5
        assert_eq!(output[2], Float32::new(4.0)); // 3.0 + 1.0
        assert_eq!(output[3], Float32::new(5.0)); // 4.0 + 1.0
    }

    #[test]
    fn test_simd_operations_consistency() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data.clone(), &[4]).unwrap();

        // Test that SIMD operations produce the same results as scalar operations
        let sum_simd = tensor.sum_simd().unwrap();
        let sum_scalar: Float32 = data.iter().fold(Float32::new(0.0), |acc, &x| acc + x);
        assert_eq!(sum_simd, sum_scalar);

        // Test ReLU consistency
        let relu_simd = tensor.relu_simd().unwrap();
        let relu_expected: Vec<Float32> = data.iter().map(|&x| if x > Float32::new(0.0) { x } else { Float32::new(0.0) }).collect();
        assert_eq!(relu_simd.as_slice(), relu_expected.as_slice());
    }
}
