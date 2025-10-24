//! Basic Linear Layer Forward Pass Test
//!
//! Simple test to verify Linear forward pass works without autograd dependencies.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{Linear, Module};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

#[test]
fn test_linear_forward_basic() {
    // Create a simple Linear layer: 3 -> 2
    let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, 2).unwrap();

    // Create input: [batch_size=1, features=3]
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[1, 3],
    )
    .unwrap();

    // Test forward pass
    let result = linear.forward(&input);

    // Should succeed (not return an error)
    if let Err(e) = &result {
        println!("Linear forward failed with error: {:?}", e);
        panic!("Linear forward should not fail");
    }

    let output = result.unwrap();

    // Check output shape: [batch_size=1, output_features=2]
    assert_eq!(output.shape().dims(), &[1, 2]);

    println!(
        "✅ Linear forward pass works! Input shape: {:?}, Output shape: {:?}",
        input.shape().dims(),
        output.shape().dims()
    );
}
