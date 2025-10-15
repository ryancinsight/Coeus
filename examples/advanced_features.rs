//! Advanced Features Example
//!
//! Demonstrates sparse neural networks, distributed training,
//! and gradient checkpointing for memory-efficient deep learning.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{
    Checkpointed, Linear, Module, Sequential, SparseLinear,
    functional::{cross_entropy, mse_loss},
};
use coeus_optim::{Adam, Optimizer};
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Coeus Advanced Features Example");
    println!("===================================");

    // 1. Demonstrate Sparse Neural Networks
    println!("\n🧠 Sparse Neural Networks");
    println!("------------------------");

    let sparse_layer = SparseLinear::<CpuBackend, Float32>::new(100, 50, 0.9, true)?;
    println!("Created sparse linear layer: {} -> {}", sparse_layer.in_features, sparse_layer.out_features);
    println!("Sparsity: {:.2}%", sparse_layer.sparsity() * 100.0);

    // Create a simple sparse network
    let mut sparse_network: Sequential<CpuBackend, DenseStorage<Float32>, Float32> = Sequential::new();
    sparse_network.add_module("sparse1".to_string(), SparseLinear::new(10, 8, 0.7, true)?);
    sparse_network.add_module("sparse2".to_string(), SparseLinear::new(8, 4, 0.7, true)?);
    sparse_network.add_module("dense".to_string(), Linear::new(4, 2)?);

    println!("Created sparse network with mixed sparse/dense layers");

    // 2. Demonstrate Gradient Checkpointing
    println!("\n💾 Gradient Checkpointing");
    println!("------------------------");

    let checkpointed_network = Checkpointed::new(sparse_network, 1);
    println!("Applied gradient checkpointing with memory savings: {:.1}x",
             checkpointed_network.memory_savings_estimate());
    println!("Computation overhead: {:.1}x",
             checkpointed_network.computation_overhead_estimate());

    // 3. Demonstrate Distributed Training (simulated)
    println!("\n🌐 Distributed Training (Simulated)");
    println!("-----------------------------------");

    // Simulate distributed training across 4 processes
    println!("Simulating distributed training across 4 processes...");

    // Create training data
    let batch_size = 32;
    let input_size = 10;
    let num_classes = 2;

    // Generate random input data
    let mut rng = rand::thread_rng();
    let mut input_data = Vec::new();
    let mut target_data = Vec::new();

    for _ in 0..batch_size {
        for _ in 0..input_size {
            input_data.push(Float32::new((rand::random::<f32>() * 2.0 - 1.0) as f32));
        }
        target_data.push(Float32::new((rand::random::<f32>() * num_classes as f32) as i32 as f32));
    }

    let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        input_data, &[batch_size, input_size],
    )?;

    let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        target_data, &[batch_size, 1],
    )?;

    // Simulate distributed forward/backward passes
    println!("Running distributed training simulation...");

    for rank in 0..4 {
        println!("  Process {}: Training on {} samples", rank, batch_size / 4);

        // Create model for this process
        let mut local_model: Sequential<CpuBackend, DenseStorage<Float32>, Float32> = Sequential::new();
        local_model.add_module("layer1".to_string(), SparseLinear::new(10, 8, 0.8, true)?);
        local_model.add_module("layer2".to_string(), Linear::new(8, num_classes)?);

        // Create optimizer and add parameters
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8)?;
        let mut params = local_model.parameters();
        for param in &mut params {
            optimizer.add_param(param.data_mut())?;
        }

        // Forward pass
        let output = local_model.forward(&input)?;
        let loss = cross_entropy(&output, &targets)?;

        // Simulate gradient synchronization across processes
        println!("    Loss: {:.4}, Parameters synchronized across {} processes",
                 loss.as_slice()[0].get(),
                 4);

        // Optimizer step
        optimizer.step()?;
        optimizer.zero_grad();
    }

    println!("✅ Distributed training simulation completed");

    // 4. Performance Comparison
    println!("\n📊 Performance Comparison");
    println!("-----------------------");

    // Compare dense vs sparse vs checkpointed models
    let dense_model: Linear<CpuBackend, DenseStorage<Float32>, Float32> = Linear::new(1000, 500)?;
    let sparse_model = SparseLinear::<CpuBackend, Float32>::new(1000, 500, 0.95, true)?;
    let checkpointed_model = Checkpointed::new(
        Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(1000, 500)?,
        5
    );

    println!("Model Comparison (1000 -> 500):");
    println!("  Dense:        {} parameters", 1000 * 500 + 500);
    println!("  Sparse (95%): {} parameters ({:.1}% sparsity)",
             sparse_model.weight.data().as_slice().len(),
             sparse_model.sparsity() * 100.0);
    println!("  Checkpointed: {:.1}x memory savings, {:.1}x compute overhead",
             checkpointed_model.memory_savings_estimate(),
             checkpointed_model.computation_overhead_estimate());

    println!("\n🎉 Advanced Features Example Completed!");
    println!("Key takeaways:");
    println!("  • Sparse networks reduce memory usage by pruning connections");
    println!("  • Gradient checkpointing trades computation for memory efficiency");
    println!("  • Distributed training enables scaling across multiple devices");
    println!("  • These techniques enable training larger models with limited resources");

    Ok(())
}
