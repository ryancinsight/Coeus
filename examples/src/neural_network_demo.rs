//! Executable example demonstrating basic tensor operations

use coeus_backend::CpuBackend;
use coeus_tensor::Tensor;

pub fn run_neural_network_demo() {
    println!("🧠 Basic Tensor Operations Demo");
    println!("===============================");

    println!("\n📊 Matrix Operations:");
    println!("---------------------");

    // Create some matrices for demonstration
    let m1 = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let m2 = Tensor::from_vec(CpuBackend::default(), vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    println!("Matrix M1:");
    println!("{:?}", m1.data());
    println!("Shape: {:?}", m1.shape());

    println!("\nMatrix M2:");
    println!("{:?}", m2.data());
    println!("Shape: {:?}", m2.shape());

    // Matrix multiplication
    let matmul_result = m1.matmul(&m2).unwrap();
    println!("\nM1 @ M2 = {:?}", matmul_result.data());

    // Element-wise operations
    let add_result = (&m1 + &m2).unwrap();
    let mul_result = (&m1 * &m2).unwrap();

    println!("M1 + M2 = {:?}", add_result.data());
    println!("M1 * M2 = {:?}", mul_result.data());

    // Loss computation (MSE) using basic operations
    println!("\n📊 Loss Computation:");
    println!("--------------------");

    let pred = Tensor::from_vec(CpuBackend::default(), vec![0.8], vec![1]).unwrap();
    let target = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

    // MSE = mean((pred - target)^2)
    let diff = (&pred - &target).unwrap();
    let squared = (&diff * &diff).unwrap();
    let mse = squared.sum();

    println!(
        "Prediction: {:.3}, Target: {:.3}",
        pred.data()[0],
        target.data()[0]
    );
    println!("MSE Loss: {:.6}", mse.data()[0]);

    println!("\n✅ Basic tensor operations completed!");
    println!("Note: Neural network modules require API updates to match Tensor<T, B> interface.");
}
