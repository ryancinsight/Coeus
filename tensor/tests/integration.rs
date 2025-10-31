//! Integration tests for tensor creation and basic operations

use backend::CpuBackend;
use dtype::float::{Float32, Float64};
use dtype::int::{Int32, Int64};
use storage::DenseStorage;
use tensor::Tensor;
use num_traits::{One, Zero};

// GPU backend is not implemented - commented out
// #[cfg(feature = "gpu")]
// use backend::GpuBackend;

type CpuTensorF32 = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
type CpuTensorF64 = Tensor<CpuBackend<Float64>, DenseStorage<Float64>, Float64>;
type CpuTensorI32 = Tensor<CpuBackend<Int32>, DenseStorage<Int32>, Int32>;
type CpuTensorI64 = Tensor<CpuBackend<Int64>, DenseStorage<Int64>, Int64>;

#[test]
fn test_tensor_from_vec_float32() {
    let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let tensor = CpuTensorF32::from_vec(data, &[3]).unwrap();

    assert_eq!(tensor.len(), 3);
    assert_eq!(tensor.shape().dims(), &[3]);
    assert!(!tensor.is_empty());
}

#[test]
fn test_tensor_from_vec_float64() {
    let data = vec![Float64::new(1.0), Float64::new(2.0)];
    let tensor = CpuTensorF64::from_vec(data, &[2]).unwrap();

    assert_eq!(tensor.len(), 2);
    assert_eq!(CpuTensorF64::dtype(), coeus_dtype::Dtype::Float64);
}

#[test]
fn test_tensor_from_vec_int32() {
    let data = vec![Int32::new(1), Int32::new(2), Int32::new(3), Int32::new(4)];
    let tensor = CpuTensorI32::from_vec(data, &[2, 2]).unwrap();

    assert_eq!(tensor.shape().dims(), &[2, 2]);
    assert_eq!(tensor.len(), 4);
}

#[test]
fn test_tensor_from_slice() {
    let data = [Int64::new(1), Int64::new(2), Int64::new(3)];
    let tensor = CpuTensorI64::from_slice(&data, &[3]).unwrap();

    assert_eq!(tensor.as_slice().len(), 3);
}

#[test]
fn test_tensor_zeros_1d() {
    let tensor = CpuTensorF32::zeros(&[5]).unwrap();

    assert_eq!(tensor.len(), 5);
    assert!(tensor.as_slice().iter().all(|&x| x.is_zero()));
}

#[test]
fn test_tensor_zeros_2d() {
    let tensor = CpuTensorF64::zeros(&[3, 4]).unwrap();

    assert_eq!(tensor.shape().dims(), &[3, 4]);
    assert_eq!(tensor.len(), 12);
    assert!(tensor.as_slice().iter().all(|&x| x.is_zero()));
}

#[test]
fn test_tensor_zeros_3d() {
    let tensor = CpuTensorF32::zeros(&[2, 3, 4]).unwrap();

    assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
    assert_eq!(tensor.len(), 24);
}

#[test]
fn test_tensor_ones_1d() {
    let tensor = CpuTensorI32::ones(&[3]).unwrap();

    assert_eq!(tensor.len(), 3);
    assert!(tensor.as_slice().iter().all(|&x| x.is_one()));
}

#[test]
fn test_tensor_ones_2d() {
    let tensor = CpuTensorF32::ones(&[2, 2]).unwrap();

    assert_eq!(tensor.len(), 4);
    assert!(tensor.as_slice().iter().all(|&x| x.is_one()));
}

#[test]
fn test_tensor_scalar() {
    let data = vec![Float32::new(42.0)];
    let tensor = CpuTensorF32::from_vec(data, &[]).unwrap();

    // Scalar has shape [], but size 1
    assert_eq!(tensor.shape().ndim(), 0);
    assert_eq!(tensor.len(), 1);
}

#[test]
fn test_tensor_device_name() {
    let tensor = CpuTensorF32::zeros(&[3]).unwrap();
    assert_eq!(tensor.device_name(), "cpu");
}

#[test]
fn test_tensor_mut_slice() {
    let mut tensor = CpuTensorI32::zeros(&[3]).unwrap();

    tensor.as_mut_slice()[0] = Int32::new(42);
    tensor.as_mut_slice()[1] = Int32::new(99);

    assert_eq!(tensor.as_slice()[0], Int32::new(42));
    assert_eq!(tensor.as_slice()[1], Int32::new(99));
}

#[test]
fn test_tensor_shape_mismatch() {
    let data = vec![Float32::new(1.0), Float32::new(2.0)];
    let result = CpuTensorF32::from_vec(data, &[3]);

    assert!(result.is_err());
}

#[test]
fn test_tensor_empty() {
    let data = vec![];
    let tensor = CpuTensorF32::from_vec(data, &[0]).unwrap();

    assert_eq!(tensor.len(), 0);
    assert!(tensor.is_empty());
}

#[test]
fn test_tensor_large_shape() {
    let tensor = CpuTensorF64::zeros(&[10, 10, 10]).unwrap();

    assert_eq!(tensor.len(), 1000);
    assert_eq!(tensor.shape().ndim(), 3);
}

// ============================================================================
// Arithmetic Operations Tests
// ============================================================================

#[test]
fn test_add_1d_float() {
    let a = CpuTensorF32::from_slice(
        &[Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )
    .unwrap();
    let b = CpuTensorF32::from_slice(
        &[Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
        &[3],
    )
    .unwrap();

    let c = &a + &b;

    assert_eq!(c.as_slice()[0], Float32::new(5.0));
    assert_eq!(c.as_slice()[1], Float32::new(7.0));
    assert_eq!(c.as_slice()[2], Float32::new(9.0));
    assert_eq!(c.shape().dims(), &[3]);
}

#[test]
fn test_add_2d_float() {
    let a = CpuTensorF64::from_slice(
        &[
            Float64::new(1.0),
            Float64::new(2.0),
            Float64::new(3.0),
            Float64::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let b = CpuTensorF64::from_slice(
        &[
            Float64::new(5.0),
            Float64::new(6.0),
            Float64::new(7.0),
            Float64::new(8.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let c = &a + &b;

    assert_eq!(c.len(), 4);
    assert_eq!(c.as_slice()[0], Float64::new(6.0));
    assert_eq!(c.as_slice()[3], Float64::new(12.0));
}

#[test]
fn test_add_negative_values() {
    let a = CpuTensorF32::from_slice(&[Float32::new(-1.0), Float32::new(-2.0)], &[2]).unwrap();
    let b = CpuTensorF32::from_slice(&[Float32::new(3.0), Float32::new(4.0)], &[2]).unwrap();

    let c = &a + &b;

    assert_eq!(c.as_slice()[0], Float32::new(2.0));
    assert_eq!(c.as_slice()[1], Float32::new(2.0));
}

#[test]
fn test_add_shape_mismatch() {
    let a = CpuTensorF32::zeros(&[3]).unwrap();
    let b = CpuTensorF32::zeros(&[4]).unwrap();

    // std::ops Add doesn't panic - it returns a safe fallback
    let result = &a + &b;
    // The result should be the left operand (safe fallback behavior)
    assert_eq!(result.shape().dims(), a.shape().dims());
    assert_eq!(result.as_slice(), a.as_slice());
}

#[test]
fn test_sub_1d_float() {
    let a = CpuTensorF32::from_slice(&[Float32::new(10.0), Float32::new(20.0)], &[2]).unwrap();
    let b = CpuTensorF32::from_slice(&[Float32::new(3.0), Float32::new(5.0)], &[2]).unwrap();

    let c = &a - &b;

    assert_eq!(c.as_slice()[0], Float32::new(7.0));
    assert_eq!(c.as_slice()[1], Float32::new(15.0));
}

#[test]
fn test_sub_negative_result() {
    let a = CpuTensorF64::from_slice(&[Float64::new(5.0), Float64::new(3.0)], &[2]).unwrap();
    let b = CpuTensorF64::from_slice(&[Float64::new(10.0), Float64::new(8.0)], &[2]).unwrap();

    let c = &a - &b;

    assert_eq!(c.as_slice()[0], Float64::new(-5.0));
    assert_eq!(c.as_slice()[1], Float64::new(-5.0));
}

#[test]
fn test_sub_shape_mismatch() {
    let a = CpuTensorF32::zeros(&[2, 3]).unwrap();
    let b = CpuTensorF32::zeros(&[3, 2]).unwrap();

    // std::ops Sub doesn't panic - it returns a safe fallback
    let result = &a - &b;
    // The result should be the left operand (safe fallback behavior)
    assert_eq!(result.shape().dims(), a.shape().dims());
    assert_eq!(result.as_slice(), a.as_slice());
}

#[test]
fn test_mul_1d_int() {
    let a = CpuTensorI32::from_slice(&[Int32::new(2), Int32::new(3), Int32::new(4)], &[3]).unwrap();
    let b = CpuTensorI32::from_slice(&[Int32::new(5), Int32::new(6), Int32::new(7)], &[3]).unwrap();

    let c = &a * &b;

    assert_eq!(c.as_slice()[0], Int32::new(10));
    assert_eq!(c.as_slice()[1], Int32::new(18));
    assert_eq!(c.as_slice()[2], Int32::new(28));
}

#[test]
fn test_mul_2d_float() {
    let a = CpuTensorF32::from_slice(
        &[
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let b = CpuTensorF32::from_slice(
        &[
            Float32::new(0.5),
            Float32::new(2.0),
            Float32::new(1.5),
            Float32::new(0.2),
        ],
        &[2, 2],
    )
    .unwrap();

    let c = &a * &b;

    assert_eq!(c.as_slice()[0], Float32::new(1.0));
    assert_eq!(c.as_slice()[1], Float32::new(6.0));
    assert_eq!(c.as_slice()[2], Float32::new(6.0));
    assert_eq!(c.as_slice()[3], Float32::new(1.0));
}

#[test]
fn test_mul_negative_values() {
    let a = CpuTensorI64::from_slice(&[Int64::new(-2), Int64::new(3)], &[2]).unwrap();
    let b = CpuTensorI64::from_slice(&[Int64::new(4), Int64::new(-5)], &[2]).unwrap();

    let c = &a * &b;

    assert_eq!(c.as_slice()[0], Int64::new(-8));
    assert_eq!(c.as_slice()[1], Int64::new(-15));
}

#[test]
fn test_div_1d_float() {
    let a = CpuTensorF64::from_slice(
        &[Float64::new(10.0), Float64::new(20.0), Float64::new(30.0)],
        &[3],
    )
    .unwrap();
    let b = CpuTensorF64::from_slice(
        &[Float64::new(2.0), Float64::new(4.0), Float64::new(5.0)],
        &[3],
    )
    .unwrap();

    let c = &a / &b;

    assert_eq!(c.as_slice()[0], Float64::new(5.0));
    assert_eq!(c.as_slice()[1], Float64::new(5.0));
    assert_eq!(c.as_slice()[2], Float64::new(6.0));
}

#[test]
fn test_div_2d_int() {
    let a = CpuTensorI32::from_slice(
        &[
            Int32::new(20),
            Int32::new(30),
            Int32::new(40),
            Int32::new(50),
        ],
        &[2, 2],
    )
    .unwrap();
    let b = CpuTensorI32::from_slice(
        &[Int32::new(4), Int32::new(5), Int32::new(8), Int32::new(10)],
        &[2, 2],
    )
    .unwrap();

    let c = &a / &b;

    assert_eq!(c.as_slice()[0], Int32::new(5));
    assert_eq!(c.as_slice()[1], Int32::new(6));
    assert_eq!(c.as_slice()[2], Int32::new(5));
    assert_eq!(c.as_slice()[3], Int32::new(5));
}

#[test]
fn test_div_negative_values() {
    let a = CpuTensorF32::from_slice(&[Float32::new(-10.0), Float32::new(20.0)], &[2]).unwrap();
    let b = CpuTensorF32::from_slice(&[Float32::new(2.0), Float32::new(-4.0)], &[2]).unwrap();

    let c = &a / &b;

    assert_eq!(c.as_slice()[0], Float32::new(-5.0));
    assert_eq!(c.as_slice()[1], Float32::new(-5.0));
}

#[test]
fn test_chained_operations() {
    let a = CpuTensorF32::from_slice(&[Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    let b = CpuTensorF32::from_slice(&[Float32::new(3.0), Float32::new(4.0)], &[2]).unwrap();
    let c = CpuTensorF32::from_slice(&[Float32::new(2.0), Float32::new(2.0)], &[2]).unwrap();

    // (a + b) * c
    let temp = &a + &b;
    let result = &temp * &c;

    assert_eq!(result.as_slice()[0], Float32::new(8.0)); // (1+3)*2 = 8
    assert_eq!(result.as_slice()[1], Float32::new(12.0)); // (2+4)*2 = 12
}

#[test]
fn test_operations_preserve_shape() {
    let a = CpuTensorF64::zeros(&[2, 3, 4]).unwrap();
    let b = CpuTensorF64::ones(&[2, 3, 4]).unwrap();

    let c = &a + &b;

    assert_eq!(c.shape().dims(), &[2, 3, 4]);
    assert_eq!(c.len(), 24);
}

// ============================================================================
// Shape Manipulation Operations Tests
// ============================================================================

#[test]
fn test_reshape_1d_to_2d() {
    let tensor = CpuTensorF32::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[6],
    )
    .unwrap();

    let reshaped = tensor.reshape(&[2, 3]).unwrap();

    assert_eq!(reshaped.shape().dims(), &[2, 3]);
    assert_eq!(reshaped.len(), 6);
    assert_eq!(reshaped.as_slice()[0], Float32::new(1.0));
    assert_eq!(reshaped.as_slice()[5], Float32::new(6.0));
}

#[test]
fn test_reshape_2d_to_1d() {
    let tensor = CpuTensorF32::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 3],
    )
    .unwrap();

    let reshaped = tensor.reshape(&[6]).unwrap();

    assert_eq!(reshaped.shape().dims(), &[6]);
    assert_eq!(reshaped.len(), 6);
    assert_eq!(reshaped.as_slice()[0], Float32::new(1.0));
    assert_eq!(reshaped.as_slice()[5], Float32::new(6.0));
}

#[test]
fn test_reshape_3d_to_2d() {
    let tensor = CpuTensorF32::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ],
        &[2, 2, 2],
    )
    .unwrap();

    let reshaped = tensor.reshape(&[4, 2]).unwrap();

    assert_eq!(reshaped.shape().dims(), &[4, 2]);
    assert_eq!(reshaped.len(), 8);
    assert_eq!(reshaped.as_slice()[0], Float32::new(1.0));
    assert_eq!(reshaped.as_slice()[7], Float32::new(8.0));
}

#[test]
fn test_reshape_dimension_inference() {
    let tensor = CpuTensorF32::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 3],
    )
    .unwrap();

    // Use -1 to infer the second dimension: 6 elements / 2 = 3
    let reshaped = tensor.reshape(&[2, -1]).unwrap();

    assert_eq!(reshaped.shape().dims(), &[2, 3]);
    assert_eq!(reshaped.len(), 6);
}

#[test]
fn test_reshape_dimension_inference_complex() {
    let tensor = CpuTensorF32::from_vec(
        (0..24).map(|x| Float32::new(x as f32)).collect(),
        &[2, 3, 4],
    )
    .unwrap();

    // Use -1 to infer: 24 elements / (2*4) = 3
    let reshaped = tensor.reshape(&[2, -1, 4]).unwrap();

    assert_eq!(reshaped.shape().dims(), &[2, 3, 4]);
    assert_eq!(reshaped.len(), 24);
}

#[test]
fn test_reshape_invalid_total_size() {
    let tensor = CpuTensorF32::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )
    .unwrap();

    // Try to reshape to 4 elements (but we only have 3)
    let result = tensor.reshape(&[2, 2]);

    assert!(result.is_err());
}

#[test]
fn test_reshape_multiple_inference_dimensions() {
    let tensor = CpuTensorF32::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )
    .unwrap();

    // Multiple -1 dimensions should fail
    let result = tensor.reshape(&[-1, -1]);

    assert!(result.is_err());
}

#[test]
fn test_reshape_invalid_dimension() {
    let tensor = CpuTensorF32::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

    // Zero dimension should fail
    let result = tensor.reshape(&[2, 0]);

    assert!(result.is_err());
}

#[test]
fn test_reshape_inference_not_divisible() {
    let tensor = CpuTensorF32::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )
    .unwrap();

    // 3 elements cannot be reshaped to [2, -1] (3/2 not integer)
    let result = tensor.reshape(&[2, -1]);

    assert!(result.is_err());
}

#[test]
fn test_transpose_2d_matrix() {
    let tensor = CpuTensorF32::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 3],
    )
    .unwrap();

    let transposed = tensor.transpose(0, 1).unwrap();

    assert_eq!(transposed.shape().dims(), &[3, 2]);
    assert_eq!(transposed.len(), 6);

    // Check transposed values: [1,4,2,5,3,6]
    assert_eq!(transposed.as_slice()[0], Float32::new(1.0)); // [0,0] -> [0,0]
    assert_eq!(transposed.as_slice()[1], Float32::new(4.0)); // [1,0] -> [0,1]
    assert_eq!(transposed.as_slice()[2], Float32::new(2.0)); // [0,1] -> [1,0]
    assert_eq!(transposed.as_slice()[3], Float32::new(5.0)); // [1,1] -> [1,1]
    assert_eq!(transposed.as_slice()[4], Float32::new(3.0)); // [0,2] -> [2,0]
    assert_eq!(transposed.as_slice()[5], Float32::new(6.0)); // [1,2] -> [2,1]
}

#[test]
fn test_transpose_square_matrix() {
    let tensor = CpuTensorF32::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let transposed = tensor.transpose(0, 1).unwrap();

    assert_eq!(transposed.shape().dims(), &[2, 2]);
    assert_eq!(transposed.len(), 4);

    // Check transposed values: [1,3,2,4]
    assert_eq!(transposed.as_slice()[0], Float32::new(1.0));
    assert_eq!(transposed.as_slice()[1], Float32::new(3.0));
    assert_eq!(transposed.as_slice()[2], Float32::new(2.0));
    assert_eq!(transposed.as_slice()[3], Float32::new(4.0));
}

#[test]
fn test_transpose_same_dimension() {
    let tensor = CpuTensorF32::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )
    .unwrap();

    let transposed = tensor.transpose(0, 0).unwrap();

    // Transposing same dimension should be identity
    assert_eq!(transposed.shape().dims(), &[3]);
    assert_eq!(transposed.as_slice(), tensor.as_slice());
}

#[test]
fn test_transpose_invalid_dimension() {
    let tensor = CpuTensorF32::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

    // 1D tensor only has dimension 0, so dim 1 is invalid
    let result = tensor.transpose(0, 1);
    assert!(result.is_err());
}

#[test]
fn test_transpose_preserves_dtype() {
    let tensor = CpuTensorI32::from_vec(
        vec![Int32::new(1), Int32::new(2), Int32::new(3), Int32::new(4)],
        &[2, 2],
    )
    .unwrap();

    let transposed = tensor.transpose(0, 1).unwrap();

    assert_eq!(transposed.shape().dims(), &[2, 2]);
    assert_eq!(CpuTensorI32::dtype(), coeus_dtype::Dtype::Int32);
}

#[test]
fn test_reshape_and_transpose_chain() {
    let tensor =
        CpuTensorF32::from_vec((0..12).map(|x| Float32::new(x as f32)).collect(), &[3, 4]).unwrap();

    // Reshape [3,4] -> [6,2]
    let reshaped = tensor.reshape(&[6, 2]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[6, 2]);

    // Transpose [6,2] -> [2,6]
    let transposed = reshaped.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape().dims(), &[2, 6]);

    // Final shape should be [2,6] with 12 elements
    assert_eq!(transposed.len(), 12);
}

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use backend::Backend;
    use pollster::FutureExt;

    // GPU backend is not implemented - commented out
    // type GpuTensorF32 = Tensor<GpuBackend, DenseStorage<Float32>, Float32>;

    #[test]
    #[ignore] // GPU backend is incomplete - skip until implemented
    fn test_gpu_tensor_creation() {
        // Skip test if no GPU available or in CI environment
        if std::env::var("CI").is_ok() {
            return;
        }

        // GPU backend is not yet implemented - this test is disabled
        // TODO: Re-enable when GPU backend provides actual functionality
        panic!("GPU backend not implemented");
    }
}
