//! Tutorial Code Validation
//!
//! This example validates all code snippets from docs/tutorial.md
//! to ensure they compile and execute correctly.
//!
//! Sprint 7.7: Usability Validation
//!
//! Run with: cargo run --example tutorial_validation

use autograd::ops::backward_with_grad;
use dtype::float::Float32;
use nn::{Linear, Module, Sequential};
use optim::{Adam, SGD};
use storage::DenseStorage;
use tensor::CpuBackend;
use tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Tutorial Code Validation");
    println!("============================\n");

    test_creating_tensors()?;
    test_arithmetic_operations()?;
    test_shape_manipulation()?;
    test_matrix_operations()?;
    test_variable_wrapping()?;
    test_gradient_computation()?;
    test_higher_order_operations()?;
    test_sequential_composition()?;
    test_optimizer_setup()?;

    println!("\n✅ All tutorial code snippets validated successfully!");
    println!("   Total tests: 9/9 passing");

    Ok(())
}

fn test_creating_tensors() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Testing: Creating Tensors");

    // Create tensor from vector
    let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )?;

    // Create tensors with fill values
    let zeros = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3])?;
    let ones = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3])?;

    // Validate shapes
    assert_eq!(a.shape().dims(), &[3]);
    assert_eq!(zeros.shape().dims(), &[2, 3]);
    assert_eq!(ones.shape().dims(), &[2, 3]);

    // Validate values
    assert_eq!(a.as_slice()[0].get(), 1.0);
    assert_eq!(zeros.as_slice()[0].get(), 0.0);
    assert_eq!(ones.as_slice()[0].get(), 1.0);

    println!("   ✅ Creating tensors: PASS");
    Ok(())
}

fn test_arithmetic_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Testing: Arithmetic Operations");

    let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )?;

    // Element-wise operations
    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
        &[3],
    )?;

    // Addition (supports borrowing for zero-copy)
    let c = &a + &b;

    // Validate addition
    assert_eq!(c.as_slice()[0].get(), 5.0);
    assert_eq!(c.as_slice()[1].get(), 7.0);
    assert_eq!(c.as_slice()[2].get(), 9.0);

    // Scalar operations
    let scalar = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?;
    let d = &c * &scalar;

    // Validate scalar multiplication
    assert_eq!(d.as_slice()[0].get(), 10.0);
    assert_eq!(d.as_slice()[1].get(), 14.0);
    assert_eq!(d.as_slice()[2].get(), 18.0);

    // Broadcasting
    let scalar_broadcast = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(10.0)],
        &[1],
    )?;
    let broadcasted = &scalar_broadcast + &a;

    // Validate broadcasting
    assert_eq!(broadcasted.as_slice()[0].get(), 11.0);
    assert_eq!(broadcasted.as_slice()[1].get(), 12.0);
    assert_eq!(broadcasted.as_slice()[2].get(), 13.0);

    println!("   ✅ Arithmetic operations: PASS");
    Ok(())
}

fn test_shape_manipulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Testing: Shape Manipulation");

    // Reshape with dimension inference
    let matrix = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        (1..=6).map(|x| Float32::new(x as f32)).collect(),
        &[2, 3],
    )?;

    // Reshape to 3x2 (6 elements total preserved)
    let reshaped = matrix.reshape(&[3, 2])?;
    assert_eq!(reshaped.shape().dims(), &[3, 2]);

    // Transpose dimensions
    let transposed = reshaped.transpose(0, 1)?;
    assert_eq!(transposed.shape().dims(), &[2, 3]);

    println!("   ✅ Shape manipulation: PASS");
    Ok(())
}

fn test_matrix_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Testing: Matrix Operations");

    let m1 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )?;

    let m2 = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ],
        &[2, 2],
    )?;

    // Matrix multiplication
    let product = m1.matmul(&m2)?;

    // Validate shape
    assert_eq!(product.shape().dims(), &[2, 2]);

    // Validate result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(product.as_slice()[0].get(), 19.0);
    assert_eq!(product.as_slice()[1].get(), 22.0);
    assert_eq!(product.as_slice()[2].get(), 43.0);
    assert_eq!(product.as_slice()[3].get(), 50.0);

    println!("   ✅ Matrix operations: PASS");
    Ok(())
}

fn test_variable_wrapping() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Testing: Variable Wrapping");

    let tensor_x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0), Float32::new(3.0)],
        &[2],
    )?;

    let tensor_y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0)],
        &[2],
    )?;

    // Enable gradient tracking on tensors
    let x = tensor_x.requires_grad_(true);
    let y = tensor_y.requires_grad_(true);

    // Tensors track computation history when gradients are enabled
    let z = &x + &y;

    // Validate result
    assert_eq!(z.as_slice()[0].get(), 6.0);
    assert_eq!(z.as_slice()[1].get(), 8.0);

    println!("   ✅ Variable wrapping: PASS");
    Ok(())
}

fn test_gradient_computation() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Testing: Gradient Computation");

    let tensor_x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0), Float32::new(3.0)],
        &[2],
    )?;

    let tensor_y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0)],
        &[2],
    )?;

    let x = tensor_x.requires_grad_(true);
    let y = tensor_y.requires_grad_(true);

    // Build computation graph
    let z = &x + &y;
    let loss = &z * &z; // loss = z²

    // Compute gradients
    let loss_grad = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    backward_with_grad(&loss, &loss_grad)?;

    // Access gradients
    match x.grad() {
        Ok(grad) => {
            // ∂loss/∂x = 2z = 2(x+y)
            assert_eq!(grad.as_slice()[0].get(), 12.0); // 2*(2+4) = 12
            assert_eq!(grad.as_slice()[1].get(), 16.0); // 2*(3+5) = 16
        }
        Err(e) => {
            eprintln!("Error accessing gradient: {:?}", e);
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Gradient not computed: {:?}", e),
            )));
        }
    }

    println!("   ✅ Gradient computation: PASS");
    Ok(())
}

fn test_higher_order_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Testing: Higher-Order Operations");

    let tensor_x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[2],
    )?;

    let tensor_y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0), Float32::new(3.0)],
        &[2],
    )?;

    let x = tensor_x.requires_grad_(true);
    let y = tensor_y.requires_grad_(true);

    // Supported operations with automatic gradients
    let exp_result = x.exp();
    let log_result = x.log();
    let sin_result = x.sin();
    let cos_result = x.cos();
    let mul_result = &x * &y; // Use multiplication instead of pow

    // Validate operations execute without errors
    assert!(exp_result.as_slice()[0].get() > 0.0);
    assert!(log_result.as_slice()[0].get() == 0.0); // ln(1) = 0
    assert!(sin_result.as_slice()[0].get().abs() < 1.0);
    assert!(cos_result.as_slice()[0].get().abs() <= 1.0);
    assert_eq!(mul_result.as_slice()[0].get(), 2.0); // 1*2 = 2

    // Reductions
    let sum_result = x.sum(None, false)?;
    let mean_result = x.mean(None, false)?;

    assert_eq!(sum_result.as_slice()[0].get(), 3.0); // 1+2 = 3
    assert_eq!(mean_result.as_slice()[0].get(), 1.5); // (1+2)/2 = 1.5

    println!("   ✅ Higher-order operations: PASS");
    Ok(())
}

fn test_sequential_composition() -> Result<(), Box<dyn std::error::Error>> {
    println!("8. Testing: Sequential Composition");

    // Build network with Sequential container (need explicit type annotation)
    let mut model: Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Sequential::new();
    model.add_module(
        "fc1".to_string(),
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 8).unwrap(),
    );
    model.add_module(
        "fc2".to_string(),
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(8, 4).unwrap(),
    );
    model.add_module(
        "fc3".to_string(),
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 2).unwrap(),
    );

    // Validate parameter count
    assert_eq!(model.parameters().len(), 6); // 3 layers × 2 params (weight + bias)

    println!("   ✅ Sequential composition: PASS");
    Ok(())
}

fn test_optimizer_setup() -> Result<(), Box<dyn std::error::Error>> {
    println!("9. Testing: Optimizer Setup");

    let mut model: Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Sequential::new();
    model.add_module(
        "fc1".to_string(),
        Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap(),
    );

    // Stochastic Gradient Descent (actual API: lr, momentum, weight_decay, dampening, nesterov)
    let _optimizer_sgd: SGD<CpuBackend<Float32>, Float32> = SGD::with_momentum(0.01, 0.9);

    // Adam optimizer (actual API: parameters, lr)
    let dummy_param: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Tensor::from_vec(vec![Float32::new(1.0)], &[1])?;
    let _optimizer_adam: Adam<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Adam::new(vec![dummy_param], 0.001);

    println!("   ✅ Optimizer setup: PASS");
    Ok(())
}
