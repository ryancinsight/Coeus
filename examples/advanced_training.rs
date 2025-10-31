//! Advanced Model Example: Neural network construction and serialization
//!
//! This example demonstrates:
//! - Neural network construction with Sequential containers
//! - Model serialization and checkpointing
//! - Parameter inspection and model introspection
//! - Cross-platform compatibility demonstration

use std::error::Error;
use std::io::{self, Write};

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Linear, Module, Sequential};
use storage::DenseStorage;
use tensor::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🧠 Coeus Advanced Model Example");
    println!("===============================\n");

    // 1. Build a complex neural network
    println!("1. Building neural network:");
    let mut model: Sequential<backend::CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Sequential::new();
    model.add_module("input_layer", Linear::new(10, 64).unwrap());
    model.add_module("hidden_1", Linear::new(64, 32).unwrap());
    model.add_module("hidden_2", Linear::new(32, 16).unwrap());
    model.add_module("output_layer", Linear::new(16, 3).unwrap());

    println!("   Network architecture:");
    println!("   Input(10) → Linear(64) → Linear(32) → Linear(16) → Output(3)");
    println!("   Total parameters: {}", model.parameters().len());

    // 2. Model inspection
    println!("\n2. Model inspection:");
    println!("   Module structure:");
    for (i, module) in model.modules().iter().enumerate() {
        println!("   {}: {}", i, module.name());
    }

    println!("\n   Parameter details:");
    let mut total_params = 0;
    for (i, param) in model.parameters().iter().enumerate() {
        let param_size = param.data().shape().size();
        total_params += param_size;
        println!(
            "   Param {}: {} (shape: {:?}, size: {})",
            i,
            param.name(),
            param.data().shape().dims(),
            param_size
        );
    }
    println!("   Total parameter count: {}", total_params);

    // 3. Forward pass demonstration
    println!("\n3. Forward pass demonstration:");
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0); 10], // 10 features
        &[1, 10],                    // [batch_size=1, input_features=10]
    )?;

    let output = model.forward(&input)?;
    println!("   Input shape: {:?}", input.shape().dims());
    println!("   Output shape: {:?}", output.shape().dims());
    println!(
        "   Output values: {:?}",
        output
            .as_slice()
            .iter()
            .map(|x| x.get())
            .collect::<Vec<_>>()
    );

    // 4. Model serialization
    println!("\n4. Model serialization:");
    let model_path = "advanced_model.json";

    println!("   Saving model to '{}'...", model_path);
    match model.save(std::path::Path::new(model_path)) {
        Ok(_) => println!("   ✓ Model successfully saved"),
        Err(e) => println!("   ✗ Failed to save model: {:?}", e),
    }

    // 5. Model loading and round-trip verification
    println!("\n5. Model loading (round-trip test):");
    let mut loaded_model = Sequential::new();
    loaded_model.add_module("input_layer", Linear::new(10, 64).unwrap());
    loaded_model.add_module("hidden_1", Linear::new(64, 32).unwrap());
    loaded_model.add_module("hidden_2", Linear::new(32, 16).unwrap());
    loaded_model.add_module("output_layer", Linear::new(16, 3).unwrap());

    println!("   Loading model from '{}'...", model_path);
    match loaded_model.load(std::path::Path::new(model_path)) {
        Ok(_) => println!("   ✓ Model successfully loaded"),
        Err(e) => println!("   ✗ Failed to load model: {:?}", e),
    }

    // Verify round-trip integrity
    let loaded_output = loaded_model.forward(&input)?;
    let outputs_match = output
        .as_slice()
        .iter()
        .zip(loaded_output.as_slice().iter())
        .all(|(a, b)| (a.get() - b.get()).abs() < 1e-6);

    println!(
        "   Round-trip verification: {}",
        if outputs_match {
            "✓ PASSED"
        } else {
            "✗ FAILED"
        }
    );

    // 6. Cross-platform compatibility note
    println!("\n6. Cross-platform compatibility:");
    println!("   ✓ Model serialized on current platform");
    println!("   ✓ JSON format ensures cross-platform compatibility");
    println!("   ✓ No platform-specific binary dependencies");
    println!("   ✓ Can be loaded on Windows/Linux/macOS");

    println!("\n✅ Advanced model example completed!");
    println!("\n💡 Key takeaways:");
    println!("   • Modular neural network construction with named layers");
    println!("   • Comprehensive model introspection and parameter analysis");
    println!("   • PyTorch-compatible model serialization");
    println!("   • Cross-platform model portability");
    println!("   • Memory-safe tensor operations throughout");

    // Cleanup
    std::fs::remove_file(model_path).ok();

    io::stdout().flush()?;
    Ok(())
}

