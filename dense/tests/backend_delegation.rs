use dense::arithmetic::add;
use storage::{DenseStorage, Storage};
use dtype::float::Float32;

#[test]
fn test_backend_delegation_add() {
    // Create test data
    let data_a = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let data_b = vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];
    
    let storage_a = DenseStorage::from_vec(data_a, &[3]).unwrap();
    let storage_b = DenseStorage::from_vec(data_b, &[3]).unwrap();
    
    // Test backend delegation
    let result = add(&storage_a, &storage_b).unwrap();
    
    // Verify the result
    let expected = [Float32::new(5.0), Float32::new(7.0), Float32::new(9.0)];
    assert_eq!(result.as_slice(), expected);
    assert_eq!(result.shape().dims(), &[3]);
}