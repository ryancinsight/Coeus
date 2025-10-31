//! Basic Usage Example: Getting started with Coeus tensors
//!
//! This example demonstrates fundamental tensor operations in Coeus,
//! showing how to create tensors, perform arithmetic, and work with
//! the type-safe tensor hierarchy.

use dtype::float::Float32;
use storage::DenseStorage;
use tensor::CpuBackend;
use tensor::Tensor;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Coeus Basic Usage Example");
    println!("============================\n");

    // Creating tensors from vectors
    println!("1. Creating tensors from vectors:");
    let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )?;
    println!("   a = {:?}", a.as_slice());

    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
        &[3],
    )?;
    println!("   b = {:?}", b.as_slice());

    // Element-wise operations
    println!("\n2. Element-wise arithmetic:");
    let c = &a + &b;
    println!("   a + b = {:?}", c.as_slice());

    // Element-wise scaling (multiply by scalar tensor)
    let scalar = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?;
    let d = &c * &scalar;
    println!("   (a + b) * 2 = {:?}", d.as_slice());

    // Broadcasting
    println!("\n3. Broadcasting operations:");
    let scalar = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(10.0)],
        &[1],
    )?;
    let broadcasted = &scalar + &a;
    println!("   scalar + vector = {:?}", broadcasted.as_slice());

    // Shape operations
    println!("\n4. Shape manipulation:");
    let matrix = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 3],
    )?;
    println!("   Original matrix (2×3): {:?}", matrix.as_slice());
    println!("   Original shape: {:?}", matrix.shape().dims());

    let reshaped = matrix.reshape(&[3, 2])?;
    println!("   Reshaped to (3×2): {:?}", reshaped.as_slice());

    // Matrix operations
    println!("\n5. Matrix operations:");
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
    let product = m1.matmul(&m2)?;
    println!("   Matrix multiplication result: {:?}", product.as_slice());

    // Reduction operations
    println!("\n6. Reduction operations:");
    let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[4],
    )?;
    println!("   Data: {:?}", data.as_slice());
    println!("   Sum: {:?}", data.sum(None, false)?.as_slice());
    println!("   Mean: {:?}", data.mean(None, false)?.as_slice());

    println!("\n✅ Basic usage example completed successfully!");
    println!("\n💡 Key takeaways:");
    println!("   • Type-safe tensor operations with compile-time guarantees");
    println!("   • Automatic broadcasting for compatible shapes");
    println!("   • Zero-copy operations with borrowing semantics");
    println!("   • Memory-safe arithmetic without unsafe code");
    println!("   • PyTorch-compatible API for seamless migration");

    io::stdout().flush()?;
    Ok(())
}

