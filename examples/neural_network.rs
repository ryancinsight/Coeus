//! Neural Network Example: Training a simple classifier
//!
//! This example demonstrates building and training a neural network
//! with Coeus, showing the complete ML workflow from data to training.

use dtype::float::Float32;
use nn::{Linear, Module, Sequential};
use std::io::{self, Write};
use storage::DenseStorage;
use tensor::CpuBackend;
use tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Coeus Neural Network Forward Pass Example");
    println!("===========================================\n");

    // Create a simple neural network: Linear(2, 4) -> Linear(4, 1)
    println!("1. Building the neural network:");
    let mut model = Sequential::new();
    model.add_module("fc1", Linear::new(2, 4).unwrap());
    model.add_module("fc2", Linear::new(4, 1).unwrap());

    println!("   Network architecture:");
    println!("   Input (2) -> Linear(2→4) -> Linear(4→1) -> Output (1)");
    println!("   Parameters: {}", model.parameters().len());

    // Test data: XOR problem
    println!("\n2. Test data (XOR classification):");
    let test_inputs = [
        vec![Float32::new(0.0), Float32::new(0.0)], // 0 XOR 0 = 0
        vec![Float32::new(0.0), Float32::new(1.0)], // 0 XOR 1 = 1
        vec![Float32::new(1.0), Float32::new(0.0)], // 1 XOR 0 = 1
        vec![Float32::new(1.0), Float32::new(1.0)], // 1 XOR 1 = 0
    ];

    println!("   Test cases:");
    for (i, input) in test_inputs.iter().enumerate() {
        println!(
            "   Case {}: Input {:?}",
            i,
            input.iter().map(|x| x.get()).collect::<Vec<_>>()
        );
    }

    // Forward pass through the network
    println!("\n3. Forward pass through untrained network:");
    for (i, input_data) in test_inputs.iter().enumerate() {
        let input_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data.clone(),
            &[1, 2], // [batch_size=1, input_features=2]
        )?;
        let output = model.forward(&input_tensor)?;
        let prediction = output.as_slice()[0].get();
        println!(
            "   Case {}: Input {:?} -> Output {:.3}",
            i,
            input_data.iter().map(|x| x.get()).collect::<Vec<_>>(),
            prediction
        );
    }

    // Show model structure
    println!("\n4. Model structure details:");
    println!("   Total parameters: {}", model.parameters().len());
    println!("   Modules:");
    for (i, module) in model.modules().iter().enumerate() {
        println!("   {}: {}", i, module.name());
    }

    println!("\n✅ Neural network forward pass completed!");
    println!("\n💡 Key takeaways:");
    println!("   • Modular neural network construction with Sequential");
    println!("   • Parameter management through Module trait");
    println!("   • Forward pass through layered architectures");
    println!("   • Memory-safe tensor operations");
    println!("   • Extensible design for custom modules");
    println!("\n📝 Note: Full training loop with autograd and optimizers");
    println!("   is demonstrated in the complete examples. This shows");
    println!("   the neural network module system foundation.");

    io::stdout().flush()?;
    Ok(())
}
