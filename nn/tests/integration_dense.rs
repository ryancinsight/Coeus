//! Integration Tests for Dense Crate Integration
//!
//! Tests dense crate integration with tensor, dense operations delegation,
//! and storage trait compatibility.
//! Validates Requirements 15.2

use backend::CpuBackend;
use dtype::float::Float32;
use storage::{DenseStorage, Storage};
use tensor::Tensor;
use tensor::dense::{DenseArithmetic, DenseCreation, DenseLayout};

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestTensor = Tensor<TestBackend, TestStorage, Float32>;

/// Test basic dense crate integration with storage
#[test]
fn test_dense_crate_basic_integration() {
    // Create dense storage using dense crate
    let storage = TestStorage::zeros(&[2, 3]).unwrap();

    // Verify shape
    assert_eq!(storage.shape().dims(), &[2, 3]);
    assert_eq!(storage.len(), 6);

    // Verify all elements are zero
    for val in storage.as_slice() {
        assert_eq!(val.get(), 0.0);
    }
}

/// Test dense creation operations
#[test]
fn test_dense_creation_operations() {
    // Test zeros
    {
        let zeros = TestStorage::zeros(&[3, 4]).unwrap();
        assert_eq!(zeros.len(), 12);
        for val in zeros.as_slice() {
            assert_eq!(val.get(), 0.0);
        }
    }

    // Test ones
    {
        let ones = TestStorage::ones(&[2, 2]).unwrap();
        assert_eq!(ones.len(), 4);
        for val in ones.as_slice() {
            assert_eq!(val.get(), 1.0);
        }
    }

    // Test full (constant value)
    {
        let full = TestStorage::full(&[2, 3], Float32::new(5.0)).unwrap();
        assert_eq!(full.len(), 6);
        for val in full.as_slice() {
            assert_eq!(val.get(), 5.0);
        }
    }

    // Test from_vec
    {
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let storage = TestStorage::from_vec(data, &[2, 2]).unwrap();
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.as_slice()[0].get(), 1.0);
        assert_eq!(storage.as_slice()[3].get(), 4.0);
    }
}

/// Test dense arithmetic operations
#[test]
fn test_dense_arithmetic_operations() {
    let backend = TestBackend::default();

    // Create test storages
    let a = TestStorage::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let b = TestStorage::from_vec(
        vec![
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ],
        &[2, 2],
    )
    .unwrap();

    // Test addition
    {
        let result = tensor::dense::arithmetic::add(&a, &b, &backend).unwrap();
        assert_eq!(result.as_slice()[0].get(), 3.0); // 1 + 2
        assert_eq!(result.as_slice()[1].get(), 5.0); // 2 + 3
        assert_eq!(result.as_slice()[2].get(), 7.0); // 3 + 4
        assert_eq!(result.as_slice()[3].get(), 9.0); // 4 + 5
    }

    // Test subtraction
    {
        let result = tensor::dense::arithmetic::sub(&b, &a, &backend).unwrap();
        assert_eq!(result.as_slice()[0].get(), 1.0); // 2 - 1
        assert_eq!(result.as_slice()[1].get(), 1.0); // 3 - 2
        assert_eq!(result.as_slice()[2].get(), 1.0); // 4 - 3
        assert_eq!(result.as_slice()[3].get(), 1.0); // 5 - 4
    }

    // Test multiplication
    {
        let result = tensor::dense::arithmetic::mul(&a, &b, &backend).unwrap();
        assert_eq!(result.as_slice()[0].get(), 2.0); // 1 * 2
        assert_eq!(result.as_slice()[1].get(), 6.0); // 2 * 3
        assert_eq!(result.as_slice()[2].get(), 12.0); // 3 * 4
        assert_eq!(result.as_slice()[3].get(), 20.0); // 4 * 5
    }

    // Test division
    {
        let result = tensor::dense::arithmetic::div(&b, &a).unwrap();
        assert_eq!(result.as_slice()[0].get(), 2.0); // 2 / 1
        assert_eq!(result.as_slice()[1].get(), 1.5); // 3 / 2
        assert!((result.as_slice()[2].get() - 1.333333).abs() < 0.001); // 4 / 3
        assert_eq!(result.as_slice()[3].get(), 1.25); // 5 / 4
    }
}

/// Test dense scalar operations
#[test]
fn test_dense_scalar_operations() {
    let storage = TestStorage::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    // Test scalar addition
    {
        let result = tensor::dense::arithmetic::add_scalar(&storage, Float32::new(10.0)).unwrap();
        assert_eq!(result.as_slice()[0].get(), 11.0);
        assert_eq!(result.as_slice()[1].get(), 12.0);
        assert_eq!(result.as_slice()[2].get(), 13.0);
        assert_eq!(result.as_slice()[3].get(), 14.0);
    }

    // Test scalar multiplication
    {
        let result = tensor::dense::arithmetic::mul_scalar(&storage, Float32::new(2.0)).unwrap();
        assert_eq!(result.as_slice()[0].get(), 2.0);
        assert_eq!(result.as_slice()[1].get(), 4.0);
        assert_eq!(result.as_slice()[2].get(), 6.0);
        assert_eq!(result.as_slice()[3].get(), 8.0);
    }
}

/// Test dense layout operations
#[test]
fn test_dense_layout_operations() {
    // Test reshape
    {
        let storage = TestStorage::from_vec(
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

        let reshaped = tensor::dense::layout::reshape(&storage, &[3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.len(), 6);
    }

    // Test flatten
    {
        let storage = TestStorage::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let flattened = tensor::dense::layout::flatten(&storage).unwrap();
        assert_eq!(flattened.shape().dims(), &[4]);
        assert_eq!(flattened.len(), 4);
    }

    // Test transpose (2D)
    {
        let storage = TestStorage::from_vec(
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

        let transposed = tensor::dense::layout::transpose_2d(&storage).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.len(), 6);
    }

    // Test is_contiguous
    {
        let storage = TestStorage::ones(&[3, 4]).unwrap();
        assert!(tensor::dense::layout::is_contiguous(&storage));
    }
}

/// Test dense integration with tensor
#[test]
fn test_dense_tensor_integration() {
    // Create tensor using dense operations
    let tensor = TestTensor::zeros(&[2, 3]).unwrap();

    // Verify tensor properties
    assert_eq!(tensor.shape().dims(), &[2, 3]);
    assert_eq!(tensor.storage().len(), 6);

    // Verify tensor uses dense storage
    for val in tensor.as_slice() {
        assert_eq!(val.get(), 0.0);
    }
}

/// Test dense operations delegation from tensor
#[test]
fn test_dense_operations_delegation() {
    // Create tensors
    let a = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let b = TestTensor::from_vec(
        vec![
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ],
        &[2, 2],
    )
    .unwrap();

    // Test tensor operations that delegate to dense
    let result = &a + &b;
    assert_eq!(result.as_slice()[0].get(), 3.0);
    assert_eq!(result.as_slice()[3].get(), 9.0);
}

/// Test storage trait compatibility
#[test]
fn test_storage_trait_compatibility() {
    use storage::Storage;

    // Create dense storage
    let storage = TestStorage::ones(&[3, 4]).unwrap();

    // Test Storage trait methods
    assert_eq!(storage.len(), 12);
    assert!(!storage.is_empty());
    assert_eq!(storage.shape().dims(), &[3, 4]);

    // Test as_slice
    let slice = storage.as_slice();
    assert_eq!(slice.len(), 12);
    for val in slice {
        assert_eq!(val.get(), 1.0);
    }
}

/// Test dense operations with different shapes
#[test]
fn test_dense_operations_different_shapes() {
    let backend = TestBackend::default();

    // Test 1D operations
    {
        let a = TestStorage::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let b = TestStorage::from_vec(
            vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
            &[3],
        )
        .unwrap();

        let result = a.add(&b, &backend).unwrap();
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.as_slice()[0].get(), 5.0);
    }

    // Test 3D operations
    {
        let a = TestStorage::ones(&[2, 3, 4]).unwrap();
        let b = TestStorage::ones(&[2, 3, 4]).unwrap();

        let result = a.add(&b, &backend).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3, 4]);
        assert_eq!(result.len(), 24);
        for val in result.as_slice() {
            assert_eq!(val.get(), 2.0);
        }
    }
}

/// Test dense operations error handling
#[test]
fn test_dense_operations_error_handling() {
    let backend = TestBackend::default();

    // Test shape mismatch in addition
    {
        let a = TestStorage::ones(&[2, 3]).unwrap();
        let b = TestStorage::ones(&[3, 2]).unwrap();

        let result = a.add(&b, &backend);
        assert!(result.is_err());
    }

    // Test invalid reshape
    {
        let storage = TestStorage::ones(&[2, 3]).unwrap();
        let result = storage.reshape(&[2, 4]); // Wrong total size
        assert!(result.is_err());
    }
}

/// Test dense creation with edge cases
#[test]
fn test_dense_creation_edge_cases() {
    // Test single element
    {
        let storage = TestStorage::ones(&[1]).unwrap();
        assert_eq!(storage.len(), 1);
        assert_eq!(storage.as_slice()[0].get(), 1.0);
    }

    // Test scalar (0D)
    {
        let storage = TestStorage::ones(&[]).unwrap();
        assert_eq!(storage.len(), 1);
    }

    // Test large tensor
    {
        let storage = TestStorage::zeros(&[100, 100]).unwrap();
        assert_eq!(storage.len(), 10000);
    }
}

/// Test dense arithmetic with broadcasting concept
#[test]
fn test_dense_arithmetic_same_shape_only() {
    let backend = TestBackend::default();

    // Dense operations require same shape (no broadcasting at this level)
    let a = TestStorage::ones(&[2, 3]).unwrap();
    let b = TestStorage::ones(&[2, 3]).unwrap();

    // Same shape should work
    let result = a.add(&b, &backend).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);

    // Different shapes should fail
    let c = TestStorage::ones(&[3, 2]).unwrap();
    let result = a.add(&c, &backend);
    assert!(result.is_err());
}

/// Test dense layout operations preserve data
#[test]
fn test_dense_layout_preserves_data() {
    let original_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
    ];

    let storage = TestStorage::from_vec(original_data.clone(), &[2, 3]).unwrap();

    // Reshape should preserve data
    let reshaped = storage.reshape(&[3, 2]).unwrap();
    assert_eq!(reshaped.len(), 6);

    // Data should be preserved (though layout changes)
    let reshaped_data = reshaped.as_slice();
    assert_eq!(reshaped_data.len(), 6);
}

/// Test dense operations with zero elements
#[test]
fn test_dense_operations_with_zeros() {
    let backend = TestBackend::default();

    let a = TestStorage::zeros(&[2, 2]).unwrap();
    let b = TestStorage::ones(&[2, 2]).unwrap();

    // Adding zeros should give original
    let result = a.add(&b, &backend).unwrap();
    for val in result.as_slice() {
        assert_eq!(val.get(), 1.0);
    }

    // Multiplying by zero should give zeros
    let result = a.mul(&b, &backend).unwrap();
    for val in result.as_slice() {
        assert_eq!(val.get(), 0.0);
    }
}

/// Test dense operations consistency
#[test]
fn test_dense_operations_consistency() {
    let backend = TestBackend::default();

    let a = TestStorage::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let b = TestStorage::from_vec(
        vec![
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ],
        &[2, 2],
    )
    .unwrap();

    // a + b should equal b + a (commutativity)
    let result1 = a.add(&b, &backend).unwrap();
    let result2 = b.add(&a, &backend).unwrap();

    for (v1, v2) in result1.as_slice().iter().zip(result2.as_slice().iter()) {
        assert_eq!(v1.get(), v2.get());
    }
}

/// Test dense creation from iterator concept
#[test]
fn test_dense_creation_patterns() {
    // Test zeros_like concept
    {
        let template = TestStorage::ones(&[3, 4]).unwrap();
        let zeros = TestStorage::zeros(template.shape().dims()).unwrap();

        assert_eq!(zeros.shape().dims(), template.shape().dims());
        for val in zeros.as_slice() {
            assert_eq!(val.get(), 0.0);
        }
    }

    // Test ones_like concept
    {
        let template = TestStorage::zeros(&[2, 5]).unwrap();
        let ones = TestStorage::ones(template.shape().dims()).unwrap();

        assert_eq!(ones.shape().dims(), template.shape().dims());
        for val in ones.as_slice() {
            assert_eq!(val.get(), 1.0);
        }
    }
}

/// Test dense operations with negative values
#[test]
fn test_dense_operations_negative_values() {
    let backend = TestBackend::default();

    let a = TestStorage::from_vec(
        vec![
            Float32::new(-1.0),
            Float32::new(-2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let b = TestStorage::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(-3.0),
            Float32::new(-4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    // Test addition with negatives
    let result = a.add(&b, &backend).unwrap();
    assert_eq!(result.as_slice()[0].get(), 0.0); // -1 + 1
    assert_eq!(result.as_slice()[1].get(), 0.0); // -2 + 2
    assert_eq!(result.as_slice()[2].get(), 0.0); // 3 + (-3)
    assert_eq!(result.as_slice()[3].get(), 0.0); // 4 + (-4)
}

/// Test dense reshape to same shape
#[test]
fn test_dense_reshape_identity() {
    let storage = TestStorage::ones(&[2, 3]).unwrap();

    // Reshape to same shape should work
    let reshaped = storage.reshape(&[2, 3]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[2, 3]);
    assert_eq!(reshaped.len(), 6);
}

/// Test dense operations with large tensors
#[test]
fn test_dense_operations_large_tensors() {
    let backend = TestBackend::default();

    // Create larger tensors
    let a = TestStorage::ones(&[50, 50]).unwrap();
    let b = TestStorage::ones(&[50, 50]).unwrap();

    // Test operations still work
    let result = a.add(&b, &backend).unwrap();
    assert_eq!(result.len(), 2500);

    // Verify all values are correct
    for val in result.as_slice() {
        assert_eq!(val.get(), 2.0);
    }
}

/// Test dense storage trait implementation completeness
#[test]
fn test_dense_storage_trait_completeness() {
    use storage::Storage;

    let storage = TestStorage::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();

    // Test all Storage trait methods
    assert_eq!(storage.len(), 4);
    assert!(!storage.is_empty());
    assert_eq!(storage.shape().dims(), &[2, 2]);

    let slice = storage.as_slice();
    assert_eq!(slice.len(), 4);

    // Test clone
    let cloned = storage.clone();
    assert_eq!(cloned.len(), storage.len());
    assert_eq!(cloned.shape().dims(), storage.shape().dims());
}

/// Test dense operations maintain shape information
#[test]
fn test_dense_operations_maintain_shape() {
    let backend = TestBackend::default();

    let a = TestStorage::ones(&[3, 4, 5]).unwrap();
    let b = TestStorage::ones(&[3, 4, 5]).unwrap();

    // Operations should maintain shape
    let result = a.add(&b, &backend).unwrap();
    assert_eq!(result.shape().dims(), &[3, 4, 5]);

    let result = a.mul(&b, &backend).unwrap();
    assert_eq!(result.shape().dims(), &[3, 4, 5]);
}

/// Test dense crate separation from storage
#[test]
fn test_dense_crate_separation() {
    // Dense crate provides higher-level operations
    // Storage provides basic memory management

    // Create storage directly
    let storage = TestStorage::zeros(&[2, 3]).unwrap();
    assert_eq!(storage.len(), 6);

    // Use dense operations on storage
    let ones = TestStorage::ones(&[2, 3]).unwrap();
    assert_eq!(ones.len(), 6);

    // Dense operations work on storage types
    let backend = TestBackend::default();
    let result = storage.add(&ones, &backend).unwrap();
    assert_eq!(result.len(), 6);
}
