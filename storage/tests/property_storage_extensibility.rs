//! Property-based tests for storage trait extensibility
//!
//! **Feature: coeus-architecture-enhancement, Property 9: Storage Format Extensibility**
//! **Validates: Requirements 4.5**
//!
//! This test verifies that adding a new storage format doesn't break existing code
//! by creating a mock storage type and verifying all operations compile and work correctly.

use dtype::float::Float32;
use dtype::DataType;
use num_traits::{One, Zero};
use proptest::prelude::*;
use storage::{DenseStorage, Result, Shape, Storage, StorageFromVec, StorageToDense};

/// Mock custom storage format to test extensibility
///
/// This represents a hypothetical new storage format that someone might add
/// to the framework. The test verifies that such additions work seamlessly.
#[derive(Debug, Clone, PartialEq)]
struct MockCustomStorage<T: DataType> {
    data: Vec<T>,
    shape: Shape,
    strides: Vec<usize>,
    // Custom metadata that a real storage format might have
    custom_metadata: String,
}

impl<T: DataType> MockCustomStorage<T> {
    fn new(data: Vec<T>, dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.size() {
            return Err(storage::StorageError::ShapeMismatch {
                expected: shape.size(),
                actual: data.len(),
            });
        }
        let strides = shape.row_major_strides();
        Ok(Self {
            data,
            shape,
            strides,
            custom_metadata: "mock_storage_v1".to_string(),
        })
    }
}

// Implement Storage trait for our mock storage
impl<T: DataType> Storage<T> for MockCustomStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn as_storage_ref(&self) -> &Self {
        self
    }

    fn full(dims: &[usize], value: T) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let data = vec![value; shape.size()];
        let strides = shape.row_major_strides();
        Ok(Self {
            data,
            shape,
            strides,
            custom_metadata: "mock_storage_v1".to_string(),
        })
    }
}

// Implement StorageFromVec trait for our mock storage
impl<T: DataType> StorageFromVec<T> for MockCustomStorage<T> {
    fn from_vec(data: Vec<T>, dims: &[usize]) -> Result<Self> {
        Self::new(data, dims)
    }

    fn zeros(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        let shape = Shape::new(dims)?;
        let data = vec![T::zero(); shape.size()];
        let strides = shape.row_major_strides();
        Ok(Self {
            data,
            shape,
            strides,
            custom_metadata: "mock_storage_v1".to_string(),
        })
    }

    fn ones(dims: &[usize]) -> Result<Self>
    where
        T: num_traits::One,
    {
        let shape = Shape::new(dims)?;
        let data = vec![T::one(); shape.size()];
        let strides = shape.row_major_strides();
        Ok(Self {
            data,
            shape,
            strides,
            custom_metadata: "mock_storage_v1".to_string(),
        })
    }
}

// Implement StorageToDense trait for our mock storage
impl<T: DataType> StorageToDense<T> for MockCustomStorage<T> {
    fn to_dense(&self) -> Result<DenseStorage<T>> {
        DenseStorage::from_vec(self.data.clone(), self.shape.dims())
    }
}

/// Generic function that works with any storage type
///
/// This demonstrates that operations can be written generically and will work
/// with any storage format that implements the required traits.
fn generic_storage_operation<S, T>(storage: &S) -> Result<S>
where
    S: Storage<T> + StorageFromVec<T>,
    T: DataType + num_traits::Zero,
{
    // Create a new storage with the same shape
    S::zeros(storage.shape().dims())
}

/// Generic function that creates storage from data
fn generic_create_storage<S, T>(data: Vec<T>, dims: &[usize]) -> Result<S>
where
    S: StorageFromVec<T>,
    T: DataType,
{
    S::from_vec(data, dims)
}

/// Generic function that converts to dense
fn generic_to_dense<S, T>(storage: &S) -> Result<DenseStorage<T>>
where
    S: StorageToDense<T>,
    T: DataType,
{
    storage.to_dense()
}

// Property-based tests

proptest! {
    /// Property 9: Storage Format Extensibility
    ///
    /// For any new storage format that implements the required traits,
    /// all existing operations should work without modification.
    ///
    /// This test verifies that:
    /// 1. The mock storage type can be created from vectors
    /// 2. Generic operations work with the mock storage
    /// 3. The storage can be converted to dense format
    /// 4. All trait methods are callable and produce valid results
    #[test]
    fn prop_storage_extensibility_from_vec(
        data in prop::collection::vec(any::<f32>(), 1..100),
        dims in prop::collection::vec(1usize..10, 1..4)
    ) {
        // Calculate total size from dimensions
        let total_size: usize = dims.iter().product();

        // Skip if data size doesn't match dimensions
        if data.len() != total_size {
            return Ok(());
        }

        // Convert f32 to Float32
        let float_data: Vec<Float32> = data.iter().map(|&x| Float32::new(x)).collect();

        // Test 1: Create mock storage from vector
        let mock_storage = MockCustomStorage::from_vec(float_data.clone(), &dims);
        prop_assert!(mock_storage.is_ok(), "Mock storage creation should succeed");

        let mock_storage = mock_storage.unwrap();

        // Test 2: Verify Storage trait methods work
        prop_assert_eq!(mock_storage.len(), total_size);
        prop_assert_eq!(mock_storage.shape().dims(), dims.as_slice());
        prop_assert_eq!(mock_storage.as_slice().len(), total_size);
        prop_assert!(mock_storage.is_contiguous());

        // Test 3: Generic operation works with mock storage
        let zeros_storage = generic_storage_operation(&mock_storage);
        prop_assert!(zeros_storage.is_ok(), "Generic operation should work with mock storage");

        // Test 4: Conversion to dense works
        let dense = generic_to_dense(&mock_storage);
        prop_assert!(dense.is_ok(), "Conversion to dense should work");

        let dense = dense.unwrap();
        prop_assert_eq!(dense.len(), total_size);
        prop_assert_eq!(dense.shape().dims(), dims.as_slice());

        // Test 5: Data is preserved through conversion
        for (i, &original) in float_data.iter().enumerate() {
            prop_assert_eq!(dense.as_slice()[i], original, "Data should be preserved");
        }
    }

    /// Property: Storage creation methods work for new formats
    ///
    /// Verifies that zeros() and ones() methods work correctly for new storage formats.
    #[test]
    fn prop_storage_creation_methods(
        dims in prop::collection::vec(1usize..10, 1..4)
    ) {
        let total_size: usize = dims.iter().product();

        // Test zeros creation
        let zeros = MockCustomStorage::<Float32>::zeros(&dims);
        prop_assert!(zeros.is_ok(), "Zeros creation should succeed");

        let zeros = zeros.unwrap();
        prop_assert_eq!(zeros.len(), total_size);
        prop_assert!(zeros.as_slice().iter().all(|x| x.is_zero()), "All elements should be zero");

        // Test ones creation
        let ones = MockCustomStorage::<Float32>::ones(&dims);
        prop_assert!(ones.is_ok(), "Ones creation should succeed");

        let ones = ones.unwrap();
        prop_assert_eq!(ones.len(), total_size);
        prop_assert!(ones.as_slice().iter().all(|x| x.is_one()), "All elements should be one");

        // Test full creation
        let value = Float32::new(42.0);
        let full = MockCustomStorage::<Float32>::full(&dims, value);
        prop_assert!(full.is_ok(), "Full creation should succeed");

        let full = full.unwrap();
        prop_assert_eq!(full.len(), total_size);
        prop_assert!(full.as_slice().iter().all(|&x| x == value), "All elements should equal value");
    }

    /// Property: Generic operations compile and work with any storage type
    ///
    /// This test verifies that operations written generically over storage traits
    /// work correctly with both existing (DenseStorage) and new (MockCustomStorage) formats.
    #[test]
    fn prop_generic_operations_work(
        data in prop::collection::vec(any::<f32>(), 1..50),
        dims in prop::collection::vec(1usize..10, 1..3)
    ) {
        let total_size: usize = dims.iter().product();
        if data.len() != total_size {
            return Ok(());
        }

        let float_data: Vec<Float32> = data.iter().map(|&x| Float32::new(x)).collect();

        // Test with DenseStorage
        let dense = generic_create_storage::<DenseStorage<Float32>, Float32>(
            float_data.clone(),
            &dims
        );
        prop_assert!(dense.is_ok(), "Generic create should work with DenseStorage");

        // Test with MockCustomStorage
        let mock = generic_create_storage::<MockCustomStorage<Float32>, Float32>(
            float_data.clone(),
            &dims
        );
        prop_assert!(mock.is_ok(), "Generic create should work with MockCustomStorage");

        // Both should produce equivalent results
        let dense = dense.unwrap();
        let mock = mock.unwrap();

        prop_assert_eq!(dense.len(), mock.len());
        prop_assert_eq!(dense.shape().dims(), mock.shape().dims());

        for i in 0..total_size {
            prop_assert_eq!(dense.as_slice()[i], mock.as_slice()[i]);
        }
    }

    /// Property: Storage trait bounds enable compile-time verification
    ///
    /// This test verifies that the trait system prevents invalid operations
    /// at compile time while allowing valid ones.
    #[test]
    fn prop_trait_bounds_work(
        dims in prop::collection::vec(1usize..10, 1..3)
    ) {
        // This function requires both Storage and StorageFromVec
        fn requires_both_traits<S, T>(dims: &[usize]) -> Result<S>
        where
            S: Storage<T> + StorageFromVec<T>,
            T: DataType + num_traits::Zero,
        {
            S::zeros(dims)
        }

        // Should work with DenseStorage
        let dense = requires_both_traits::<DenseStorage<Float32>, Float32>(&dims);
        prop_assert!(dense.is_ok());

        // Should work with MockCustomStorage
        let mock = requires_both_traits::<MockCustomStorage<Float32>, Float32>(&dims);
        prop_assert!(mock.is_ok());

        // Both should produce valid storage
        let dense = dense.unwrap();
        let mock = mock.unwrap();

        prop_assert_eq!(dense.shape().dims(), dims.as_slice());
        prop_assert_eq!(mock.shape().dims(), dims.as_slice());
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_mock_storage_basic_operations() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = MockCustomStorage::from_vec(data.clone(), &[3]).unwrap();

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.shape().dims(), &[3]);
        assert_eq!(storage.as_slice(), data.as_slice());
        assert!(storage.is_contiguous());
    }

    #[test]
    fn test_mock_storage_zeros() {
        let storage = MockCustomStorage::<Float32>::zeros(&[2, 3]).unwrap();
        assert_eq!(storage.len(), 6);
        assert!(storage.as_slice().iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_mock_storage_ones() {
        let storage = MockCustomStorage::<Float32>::ones(&[2, 2]).unwrap();
        assert_eq!(storage.len(), 4);
        assert!(storage.as_slice().iter().all(|x| x.is_one()));
    }

    #[test]
    fn test_mock_storage_to_dense() {
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let storage = MockCustomStorage::from_vec(data.clone(), &[2, 2]).unwrap();
        let dense = storage.to_dense().unwrap();

        assert_eq!(dense.len(), 4);
        assert_eq!(dense.shape().dims(), &[2, 2]);
        assert_eq!(dense.as_slice(), data.as_slice());
    }

    #[test]
    fn test_generic_operations_with_mock_storage() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = MockCustomStorage::from_vec(data, &[3]).unwrap();

        // Test generic operation
        let zeros = generic_storage_operation(&storage).unwrap();
        assert_eq!(zeros.len(), 3);
        assert!(zeros.as_slice().iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_generic_create_with_both_storage_types() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];

        // Create with DenseStorage
        let dense =
            generic_create_storage::<DenseStorage<Float32>, Float32>(data.clone(), &[2]).unwrap();

        // Create with MockCustomStorage
        let mock =
            generic_create_storage::<MockCustomStorage<Float32>, Float32>(data.clone(), &[2])
                .unwrap();

        // Both should have same data
        assert_eq!(dense.as_slice(), mock.as_slice());
    }
}
