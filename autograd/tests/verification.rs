
use autograd::tensor_ops;
use tensor::{Tensor, CpuBackend, DenseStorage};
use dtype::float::Float32;

#[test]
fn test_autograd_simple_mul_mean() {
    // a = [2.0, 3.0]
    let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0), Float32::new(3.0)], 
        &[2]
    ).unwrap().requires_grad_(true);
    
    // b = [4.0, 5.0]
    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0)], 
        &[2]
    ).unwrap().requires_grad_(true);
    
    // c = a * b = [8.0, 15.0]
    let c = tensor_ops::mul(&a, &b).unwrap();
    
    // d = mean(c) = (8 + 15) / 2 = 11.5
    let d = tensor_ops::mean(&c, None, false).unwrap();
    
    // backward
    autograd::backward(&d, None, false, false).unwrap();
    
    // Check gradients
    // d(mean)/dc = 0.5
    // dc/da = b
    // d(mean)/da = d(mean)/dc * dc/da = 0.5 * b
    // grad_a = [0.5 * 4, 0.5 * 5] = [2.0, 2.5]
    
    let grad_a = a.grad().unwrap();
    let grad_a_data = grad_a.as_slice();
    assert!((grad_a_data[0].0 - 2.0).abs() < 1e-6);
    assert!((grad_a_data[1].0 - 2.5).abs() < 1e-6);
    
    // grad_b = 0.5 * a = [1.0, 1.5]
    let grad_b = b.grad().unwrap();
    let grad_b_data = grad_b.as_slice();
    assert!((grad_b_data[0].0 - 1.0).abs() < 1e-6);
    assert!((grad_b_data[1].0 - 1.5).abs() < 1e-6);
}

#[test]
fn test_autograd_reuse_variable() {
    // x = [2.0]
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)], 
        &[1]
    ).unwrap().requires_grad_(true);
    
    // y = x * x + x
    let x2 = tensor_ops::mul(&x, &x).unwrap();
    let y = tensor_ops::add(&x2, &x).unwrap();
    
    // backward
    autograd::backward(&y, None, false, false).unwrap();
    
    // dy/dx = 2x + 1 = 2(2) + 1 = 5
    let grad_x = x.grad().unwrap();
    let grad_x_data = grad_x.as_slice();
    assert!((grad_x_data[0].0 - 5.0).abs() < 1e-6);
}
