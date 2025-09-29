//! Executable examples demonstrating Coeus tensor library functionality

mod autograd_demo;
mod neural_network_demo;

use coeus_backend::CpuBackend;
use coeus_tensor::Tensor;

fn main() {
    println!("🚀 Coeus Tensor Library - Basic Operations Demo");
    println!("================================================");

    // Basic tensor creation and operations
    println!("\n📊 Basic Tensor Operations:");
    println!("---------------------------");

    // Create tensors
    let a = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(CpuBackend::default(), vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    println!("Tensor A (2×2):");
    println!("{:?}", a.data());
    println!("Shape: {:?}", a.shape());

    println!("\nTensor B (2×2):");
    println!("{:?}", b.data());
    println!("Shape: {:?}", b.shape());

    // Arithmetic operations
    let sum = (&a + &b).unwrap();
    let diff = (&a - &b).unwrap();
    let prod = (&a * &b).unwrap();
    let quot = (&a / &Tensor::from_vec(CpuBackend::default(), vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap()).unwrap();

    println!("\nA + B = {:?}", sum.data());
    println!("A - B = {:?}", diff.data());
    println!("A * B = {:?}", prod.data());
    println!("A / 2 = {:?}", quot.data());

    // Matrix operations
    println!("\n🔢 Matrix Operations:");
    println!("--------------------");

    let matrix_prod = a.matmul(&b).unwrap();
    println!("A @ B (matrix multiplication) = {:?}", matrix_prod.data());

    let sum_all = sum.sum();
    println!("Sum of all elements in A+B = {:?}", sum_all.data()[0]);

    // Broadcasting
    println!("\n📡 Broadcasting Operations:");
    println!("----------------------------");

    let scalar = Tensor::from_vec(CpuBackend::default(), vec![10.0], vec![1]).unwrap();
    let broadcast_sum = (&a + &scalar).unwrap();
    println!("A + 10.0 (broadcasting) = {:?}", broadcast_sum.data());

    // Advanced indexing (simplified for now)
    println!("\n🎯 Advanced Indexing:");
    println!("---------------------");
    println!("Indexing operations temporarily disabled due to compilation issues");
    println!("Tensor shape: {:?}", a.shape());

    println!("\n✅ Basic operations completed successfully!");

    // Run autograd demo
    println!("\n");
    autograd_demo::run_autograd_demo();

    // Run neural network demo
    println!("\n");
    neural_network_demo::run_neural_network_demo();
}
