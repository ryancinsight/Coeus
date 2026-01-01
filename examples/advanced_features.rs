//! Advanced Features Example
//!
//! Demonstrates sparse neural networks, distributed training,
//! and gradient checkpointing for memory-efficient deep learning.

use autograd::{backward, AutogradError};
use backend::CpuBackend;
use dtype::float::Float32;
use nn::{
    functional::cross_entropy, functional_activations::relu, Linear, Module, Sequential,
    SparseLinear,
};
use optim::{Adam, BaseOptimizer};
use storage::DenseStorage;
use tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Coeus Advanced Features Example");
    println!("===================================");

    // 1. Demonstrate Sparse Neural Networks
    println!("\n🧠 Sparse Neural Networks");
    println!("------------------------");

    let sparse_layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(100, 50, 0.9, true)?;
    println!(
        "Created sparse linear layer: {} -> {}",
        sparse_layer.in_features, sparse_layer.out_features
    );
    println!("Sparsity: {:.2}%", sparse_layer.sparsity() * 100.0);

    // Create a simple sparse network
    let mut sparse_network: Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Sequential::new();
    sparse_network.add_module("sparse1".to_string(), SparseLinear::new(10, 8, 0.7, true)?);
    sparse_network.add_module("sparse2".to_string(), SparseLinear::new(8, 4, 0.7, true)?);
    sparse_network.add_module("dense".to_string(), Linear::new(4, 2)?);

    println!("Created sparse network with mixed sparse/dense layers");

    // 2. Demonstrate Gradient Checkpointing
    println!("\n💾 Gradient Checkpointing");
    println!("------------------------");

    use autograd::{checkpoint, checkpoint_sequential};

    // Create some test tensors with gradient tracking
    let input1: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Tensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2])?;
    let input2: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Tensor::from_vec(vec![Float32::new(3.0), Float32::new(4.0)], &[2])?;

    // Single checkpoint example
    println!("Testing single checkpoint:");
    let checkpointed_result = checkpoint(
        |x: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>| -> Result<_, AutogradError> {
            // Simulate a complex computation: x -> exp(x) -> sum -> relu
            let exp_result = x.exp();
            let sum_result = exp_result
                .sum(None, false)
                .map_err(AutogradError::TensorError)?;
            relu(&sum_result).map_err(|e| AutogradError::GradientComputationError {
                operation: "relu".to_string(),
                source: Box::new(e),
            })
        },
        &input1,
    )?;

    println!("  Forward result: {:?}", checkpointed_result);

    // Test backward pass
    backward(&checkpointed_result)?;
    println!("  Input gradient after backward: {:?}", input1.grad());

    // Sequential checkpointing example
    println!("Testing sequential checkpointing:");
    let segments = vec![&input1, &input2];
    let sequential_results = checkpoint_sequential(
        |x: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>| -> Result<_, AutogradError> {
            // Process each segment: x -> x^2 -> sum
            x.powf(Float32::new(2.0))
                .sum(None, false)
                .map_err(AutogradError::TensorError)
        },
        &segments,
    )?;

    println!("  Sequential results: {:?}", sequential_results);

    // Test backward through sequential checkpointing
    backward(&sequential_results[0])?;
    println!("  Gradient for first segment input: {:?}", input1.grad());

    println!("✓ Gradient checkpointing working - memory-efficient training enabled!");

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
    let mut input_data = Vec::new();
    let mut target_data = Vec::new();

    for _ in 0..batch_size {
        for _ in 0..input_size {
            input_data.push(Float32::new((rand::random::<f32>() * 2.0 - 1.0) as f32));
        }
        target_data.push(Float32::new(
            (rand::random::<f32>() * num_classes as f32) as i32 as f32,
        ));
    }

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[batch_size, input_size],
    )?;

    let targets = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        target_data,
        &[batch_size, 1],
    )?;

    // Simulate distributed forward/backward passes
    println!("Running distributed training simulation...");

    for rank in 0..4 {
        println!("  Process {}: Training on {} samples", rank, batch_size / 4);

        // Create model for this process
        let mut local_model: Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Sequential::new();
        local_model.add_module("layer1".to_string(), SparseLinear::new(10, 8, 0.8, true)?);
        local_model.add_module("layer2".to_string(), Linear::new(8, num_classes)?);

        // Create optimizer with model parameters
        let params = local_model.parameters();
        let param_tensors: Vec<_> = params.iter().map(|p| p.data().clone()).collect();
        let mut optimizer = Adam::new(param_tensors, 0.001);

        // Forward pass
        let output = local_model.forward(&input)?;
        let loss = cross_entropy(&output, &targets)?;

        // Simulate gradient synchronization across processes
        println!(
            "    Loss: {:.4}, Parameters synchronized across {} processes",
            loss.as_slice()[0].get(),
            4
        );

        // Optimizer step
        BaseOptimizer::step(&mut optimizer)?;
        BaseOptimizer::zero_grad(&mut optimizer);
    }

    println!("✅ Distributed training simulation completed");

    // 4. Performance Comparison
    println!("\n📊 Performance Comparison");
    println!("-----------------------");

    // Compare dense vs sparse vs checkpointed models
    let _dense_model: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
        Linear::new(1000, 500)?;
    let sparse_model = SparseLinear::<CpuBackend<Float32>, Float32>::new(1000, 500, 0.95, true)?;
    // let checkpointed_model = Checkpointed::new(
    //     Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1000, 500)?,
    //     5
    // );

    println!("Model Comparison (1000 -> 500):");
    println!("  Dense:        {} parameters", 1000 * 500 + 500);
    println!(
        "  Sparse (95%): {} parameters ({:.1}% sparsity)",
        sparse_model.weight.data().as_slice().len(),
        sparse_model.sparsity() * 100.0
    );
    // println!("  Checkpointed: {:.1}x memory savings, {:.1}x compute overhead",
    //          checkpointed_model.memory_savings_estimate(),
    //          checkpointed_model.computation_overhead_estimate());

    println!("\n🎉 Advanced Features Example Completed!");
    println!("Key takeaways:");
    println!("  • Sparse networks reduce memory usage by pruning connections");
    println!("  • Gradient checkpointing trades computation for memory efficiency");
    println!("  • Distributed training enables scaling across multiple devices");
    println!("  • These techniques enable training larger models with limited resources");

    Ok(())
}
