//! Mixed Precision Training Example
//!
//! This example demonstrates automatic mixed precision (AMP) training
//! using FP16 operations with gradient scaling for numerical stability.
//!
//! ## Features Demonstrated
//!
//! - Loss scaling to prevent gradient underflow
//! - Gradient scaling for FP16 stability
//! - NaN/Inf detection and handling
//! - Automatic loss scale adjustment

use autograd::ops::backward;
use dtype::float::Float32;
use nn::{
    amp::{GradientScaler, MixedPrecision},
    MSELoss, Module, Sequential,
};
use std::time::Instant;
use storage::{DenseStorage, Storage};
use tensor::CpuBackend;
use tensor::Tensor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Coeus Mixed Precision Training Example");
    println!("==========================================");

    // Create a simple model
    println!("\n🏗️  Building model...");
    let model = Sequential::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();

    // Create mixed precision context
    println!("\n⚡ Setting up mixed precision training...");
    let amp = MixedPrecision::new().with_loss_scale(1024.0); // Start with higher scale for stability

    let mut scaler = GradientScaler::new(1024.0);

    // Generate some dummy training data
    println!("\n📊 Generating training data...");
    let batch_size = 32;
    let input_size = 784;

    // Training loop
    println!("\n🏃 Starting mixed precision training...");
    let start_time = Instant::now();

    for epoch in 0..5 {
        println!("\n📈 Epoch {}", epoch + 1);

        // Generate batch data (normally this would come from your dataset)
        let input_data: Vec<Float32> = (0..batch_size * input_size)
            .map(|i| Float32::new((i % 100) as f32 / 100.0))
            .collect();

        let target_data: Vec<Float32> = (0..batch_size * 10)
            .map(|i| Float32::new((i % 10) as f32 / 10.0))
            .collect();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[batch_size, input_size],
        )?;
        let target = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            target_data,
            &[batch_size, 10],
        )?;

        // Forward pass
        let output = model.forward(&input)?;

        // Compute loss
        let loss_fn = MSELoss::new();
        let loss = loss_fn.forward(&output, &target)?;

        println!("   Loss: {:.6}", loss.storage_ref().as_slice()[0].get());
        println!("   Loss scale: {:.1}", amp.loss_scale());
        println!("   Gradient scale: {:.1}", scaler.scale());

        // Scale loss for mixed precision
        let scaled_loss = amp.scale_loss(&loss)?;

        // Backward pass
        backward(&scaled_loss)?;

        // Get gradients and check for overflow
        let gradients: Vec<_> = model
            .parameters()
            .iter()
            .map(|p| p.data().grad())
            .collect::<Result<Vec<_>, _>>()?;
        let gradient_refs: Vec<_> = gradients.iter().collect::<Vec<_>>();
        scaler.check_gradients(&gradient_refs)?;

        // Step gradient scaler (simplified - would normally step optimizer)
        scaler.step()?;
        scaler.update();

        // Note: In a real implementation, you would step the optimizer here
        // For demonstration, we just show the AMP functionality
    }

    let training_time = start_time.elapsed();
    println!("\n✅ Mixed precision training completed!");
    println!("   Training time: {:.2}s", training_time.as_secs_f32());
    println!("   Final loss scale: {:.1}", amp.loss_scale());
    println!("   Final gradient scale: {:.1}", scaler.scale());

    // Demonstrate NaN/Inf detection
    println!("\n🔍 Testing NaN/Inf detection...");

    // Create tensor with NaN
    let nan_data = vec![Float32::new(1.0), Float32::new(f32::NAN), Float32::new(3.0)];
    let nan_tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(nan_data, &[3])?;

    if amp.has_nan_or_inf(&nan_tensor)? {
        println!("   ✅ NaN detected correctly");
    } else {
        println!("   ❌ NaN detection failed");
    }

    // Create tensor with Inf
    let inf_data = vec![
        Float32::new(1.0),
        Float32::new(f32::INFINITY),
        Float32::new(3.0),
    ];
    let inf_tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(inf_data, &[3])?;

    if amp.has_nan_or_inf(&inf_tensor)? {
        println!("   ✅ Inf detected correctly");
    } else {
        println!("   ❌ Inf detection failed");
    }

    println!("\n🎉 Mixed precision example completed successfully!");
    println!("   Demonstrated: Loss scaling, gradient scaling, NaN/Inf detection");

    Ok(())
}
