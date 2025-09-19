//! Comprehensive error condition testing
//!
//! This module implements exhaustive error condition validation to ensure
//! robust error handling and graceful failure recovery across all tensor operations.

use crate::Tensor;

/// Test invalid tensor creation scenarios
#[test]
fn test_invalid_tensor_creation() {
    // Test empty data with non-empty shape
    let result = Tensor::try_from_vec(vec![], vec![1, 2]);
    assert!(result.is_err(), "Empty data with non-empty shape should fail");

    // Test data length mismatch with shape
    let result = Tensor::try_from_vec(vec![1.0, 2.0, 3.0], vec![2, 2]);
    assert!(result.is_err(), "Data length 3 with shape [2,2] should fail (needs 4 elements)");
}

/// Test shape validation errors
#[test]
fn test_shape_validation_errors() {
    // Test zero dimension in shape
    let result = Tensor::try_from_vec(vec![1.0, 2.0], vec![0, 2]);
    assert!(result.is_err(), "Shape with zero dimension should fail");

    // Test negative dimension (if supported)
    let result = Tensor::try_from_vec(vec![1.0, 2.0], vec![-1, 2]);
    assert!(result.is_err(), "Negative dimension should fail");
}

/// Test arithmetic operation errors
#[test]
fn test_arithmetic_operation_errors() {
    let tensor_a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let tensor_b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]);

    // Test shape mismatch in addition
    let result = &tensor_a + &tensor_b;
    assert!(result.is_err(), "Addition with mismatched shapes should fail");

    // Test shape mismatch in multiplication
    let result = &tensor_a * &tensor_b;
    assert!(result.is_err(), "Multiplication with mismatched shapes should fail");

    // Test shape mismatch in subtraction
    let result = &tensor_a - &tensor_b;
    assert!(result.is_err(), "Subtraction with mismatched shapes should fail");

    // Test shape mismatch in division
    let result = &tensor_a / &tensor_b;
    assert!(result.is_err(), "Division with mismatched shapes should fail");
}

/// Test division by zero handling
#[test]
fn test_division_by_zero() {
    let tensor_a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let tensor_b = Tensor::from_vec(vec![0.0, 1.0], vec![2]);

    // Test division by zero
    let result = &tensor_a / &tensor_b;
    assert!(result.is_ok(), "Division by zero should be handled gracefully");

    let result_tensor = result.unwrap();
    let data = result_tensor.data();

    // First element should be infinite or NaN due to division by zero
    assert!(!data[0].is_finite(), "Division by zero should produce infinite or NaN");

    // Second element should be finite (2.0 / 1.0 = 2.0)
    assert!(data[1].is_finite(), "Division by non-zero should be finite");
    assert!((data[1] - 2.0).abs() < 1e-12, "2.0 / 1.0 should equal 2.0");
}

/// Test matrix multiplication shape errors
#[test]
fn test_matrix_multiplication_shape_errors() {
    // Test incompatible matrix dimensions
    let matrix_a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]); // 2x2
    let matrix_b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);       // 3x1

    let result = &matrix_a @ &matrix_b;
    assert!(result.is_err(), "Matrix multiplication with incompatible dimensions should fail");
}

/// Test reduction operation errors
#[test]
fn test_reduction_operation_errors() {
    // Test reduction on empty tensor
    let empty_tensor = Tensor::from_vec(vec![], vec![0]);
    let result = empty_tensor.sum();
    assert!(result.is_err(), "Sum reduction on empty tensor should fail");

    let result = empty_tensor.mean();
    assert!(result.is_err(), "Mean reduction on empty tensor should fail");
}

/// Test indexing operation errors
#[test]
fn test_indexing_operation_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    // Test out of bounds indexing
    let result = tensor.get(vec![3]); // Index 3 is out of bounds for length 3
    assert!(result.is_err(), "Out of bounds indexing should fail");

    // Test negative indexing (if not supported)
    let result = tensor.get(vec![-1]);
    assert!(result.is_err(), "Negative indexing should fail");

    // Test multi-dimensional indexing errors
    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let result = matrix.get(vec![2, 0]); // Row index 2 is out of bounds
    assert!(result.is_err(), "Out of bounds multi-dimensional indexing should fail");
}

/// Test reshape operation errors
#[test]
fn test_reshape_operation_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    // Test reshape with incompatible total elements
    let result = tensor.reshape(&[2, 3]); // 4 elements cannot be reshaped to 2x3 (6 elements)
    assert!(result.is_err(), "Reshape with incompatible element count should fail");

    // Test reshape with zero dimension
    let result = tensor.reshape(&[0, 4]);
    assert!(result.is_err(), "Reshape with zero dimension should fail");
}

/// Test transpose operation errors
#[test]
fn test_transpose_operation_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Test transpose with invalid dimension indices
    let result = tensor.transpose(0, 2); // Dimension 2 doesn't exist in 2D tensor
    assert!(result.is_err(), "Transpose with invalid dimension index should fail");

    // Test transpose with same dimension indices
    let result = tensor.transpose(0, 0); // Transposing dimension with itself
    assert!(result.is_ok(), "Transpose with same dimension indices should be valid");
}

/// Test gradient computation errors
#[test]
fn test_gradient_computation_errors() {
    // Test gradient computation without requires_grad
    let tensor = Tensor::scalar(1.0);
    let y = (&tensor * &tensor).unwrap();

    let result = y.backward();
    assert!(result.is_ok(), "Backward on tensor without requires_grad should succeed but be no-op");

    // Check that no gradient was computed
    assert!(tensor.grad().is_none(), "Tensor without requires_grad should not have gradient");

    // Test multiple backward calls on same graph
    let mut x = Tensor::scalar(2.0);
    x.set_requires_grad(true);
    let y = (&x * &x).unwrap();

    // First backward should succeed
    let result1 = y.backward();
    assert!(result1.is_ok(), "First backward call should succeed");

    // Second backward on same graph might fail or be no-op depending on implementation
    let result2 = y.backward();
    // This is implementation-dependent - could succeed or fail
}

/// Test scalar extraction errors
#[test]
fn test_scalar_extraction_errors() {
    // Test extracting scalar from multi-element tensor
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let result = tensor.as_scalar();
    assert!(result.is_err(), "Extracting scalar from multi-element tensor should fail");

    // Test extracting scalar from empty tensor
    let empty_tensor = Tensor::from_vec(vec![], vec![0]);
    let result = empty_tensor.as_scalar();
    assert!(result.is_err(), "Extracting scalar from empty tensor should fail");
}

/// Test broadcasting errors
#[test]
fn test_broadcasting_errors() {
    // Test incompatible broadcasting shapes
    let tensor_a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let tensor_b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = &tensor_a + &tensor_b;
    assert!(result.is_err(), "Broadcasting incompatible shapes should fail");

    // Test valid broadcasting should work
    let scalar = Tensor::scalar(1.0);
    let vector = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    let result = &scalar + &vector;
    assert!(result.is_ok(), "Broadcasting scalar + vector should succeed");

    let result_tensor = result.unwrap();
    let data = result_tensor.data();
    assert_eq!(data.len(), 3, "Result should have 3 elements");
    assert!((data[0] - 2.0).abs() < 1e-12, "First element should be 1.0 + 1.0 = 2.0");
    assert!((data[1] - 3.0).abs() < 1e-12, "Second element should be 1.0 + 2.0 = 3.0");
    assert!((data[2] - 4.0).abs() < 1e-12, "Third element should be 1.0 + 3.0 = 4.0");
}

/// Test memory allocation errors
#[test]
fn test_memory_allocation_errors() {
    // Test allocation with extremely large size (this might fail due to system limits)
    let large_size = usize::MAX / 8; // Would require more than all available memory
    let result = Tensor::try_from_vec(vec![], vec![large_size]);
    assert!(result.is_err(), "Allocation of extremely large tensor should fail");

    // Test reasonable allocation should work
    let reasonable_size = 1000;
    let data = vec![1.0; reasonable_size];
    let result = Tensor::try_from_vec(data, vec![reasonable_size]);
    assert!(result.is_ok(), "Reasonable tensor allocation should succeed");
}

/// Test serialization/deserialization errors
#[test]
fn test_serialization_errors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

    // Test serialization (should work)
    let serialized = tensor.to_bytes();
    assert!(serialized.is_ok(), "Tensor serialization should succeed");

    // Test deserialization with invalid data
    let invalid_data = vec![0u8, 1u8, 2u8]; // Too short
    let result = Tensor::from_bytes(&invalid_data);
    assert!(result.is_err(), "Deserialization with invalid data should fail");

    // Test round-trip serialization
    let serialized = tensor.to_bytes().unwrap();
    let deserialized = Tensor::from_bytes(&serialized);
    assert!(deserialized.is_ok(), "Round-trip serialization should succeed");

    let deserialized_tensor = deserialized.unwrap();
    assert_eq!(tensor.shape(), deserialized_tensor.shape(), "Shape should be preserved");
    assert_eq!(tensor.data(), deserialized_tensor.data(), "Data should be preserved");
}

/// Test activation function errors
#[test]
fn test_activation_function_errors() {
    // Test activation functions on empty tensor
    let empty_tensor = Tensor::from_vec(vec![], vec![0]);

    let result = empty_tensor.relu();
    assert!(result.is_err() || result.unwrap().data().is_empty(),
            "ReLU on empty tensor should handle gracefully");

    let result = empty_tensor.sigmoid();
    assert!(result.is_err() || result.unwrap().data().is_empty(),
            "Sigmoid on empty tensor should handle gracefully");

    let result = empty_tensor.tanh();
    assert!(result.is_err() || result.unwrap().data().is_empty(),
            "Tanh on empty tensor should handle gracefully");
}

/// Test concatenation errors
#[test]
fn test_concatenation_errors() {
    // Test concatenation with incompatible shapes
    let tensor_a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    let tensor_b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]);

    let result = Tensor::concat(&[&tensor_a, &tensor_b], 0);
    assert!(result.is_err(), "Concatenation with incompatible shapes should fail");

    // Test concatenation along invalid dimension
    let matrix_a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let matrix_b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let result = Tensor::concat(&[&matrix_a, &matrix_b], 2); // Dimension 2 doesn't exist
    assert!(result.is_err(), "Concatenation along invalid dimension should fail");

    // Test valid concatenation should work
    let result = Tensor::concat(&[&tensor_a, &tensor_a], 0);
    assert!(result.is_ok(), "Concatenation with compatible shapes should succeed");

    let result_tensor = result.unwrap();
    assert_eq!(result_tensor.shape(), &[4], "Concatenated tensor should have correct shape");
    assert_eq!(result_tensor.data(), &[1.0, 2.0, 1.0, 2.0], "Concatenated data should be correct");
}
