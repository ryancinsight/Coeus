//! Automatic differentiation examples

use coeus_tensor::Tensor;

/// Example of automatic differentiation
pub fn autograd_example() {
    // Create tensors with gradient tracking
    let mut a = Tensor::from_vec(vec![2.0], vec![1]);
    a.set_requires_grad(true);

    let mut b = Tensor::from_vec(vec![3.0], vec![1]);
    b.set_requires_grad(true);

    // Compute some operations
    let c = (&a * &b).unwrap();
    let _d = (&c + &a).unwrap();

    // This would compute gradients if the autograd system was fully implemented
    // d.backward();

    println!("Autograd example completed");
}
