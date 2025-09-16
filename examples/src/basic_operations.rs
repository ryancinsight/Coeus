//! Basic tensor operations examples

use coeus_tensor::Tensor;

/// Example of basic tensor operations
pub fn basic_tensor_ops() {
    // Create tensors
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

    // Basic operations
    let _c = &a + &b;
    let _d = &a * &b;

    println!("Basic operations completed");
}
