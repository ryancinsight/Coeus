//! Tracing Example: Setting up observability for Coeus
//!
//! This example demonstrates how to configure tracing for production observability
//! in Coeus applications. Tracing spans are automatically added to tensor operations
//! and autograd backward passes.

use std::io::{self, Write};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize tracing with console output and RUST_LOG filtering
fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "tensor=trace,autograd=trace,coeus_examples=info".into()
            }),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(false)
                .with_thread_names(false)
                .compact(),
        )
        .init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing - respects RUST_LOG environment variable
    init_tracing();

    info!("Starting Coeus tracing example");

    // Import Coeus crates
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::CpuBackend;
    use tensor::Tensor;

    info!("Creating tensors for demonstration");

    // Create some tensors - these operations will be traced
    let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )?;

    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)],
        &[3],
    )?;

    // Arithmetic operations are automatically traced with spans
    info!("Performing tensor addition (operations are traced with spans for observability)");
    let c = &a + &b;

    info!("Performing scalar multiplication");
    let scalar_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?;
    let _d = &c * &scalar_tensor;

    // Broadcasting example
    info!("Broadcasting scalar to vector");
    let scalar = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(10.0)],
        &[1],
    )?;
    let _broadcasted = &scalar + &a;

    // Autograd example with tracing
    use autograd::ops::backward_with_grad;

    info!("Creating autograd tensors");
    let mut x = a.clone();
    x = x.requires_grad_(true);
    let mut y = b.clone();
    y = y.requires_grad_(true);

    info!("Performing autograd operations (backward pass will show spans)");
    let z = &x + &y;
    let mut scalar_copy = scalar.clone();
    scalar_copy = scalar_copy.requires_grad_(true);
    let w = &z * &scalar_copy;

    // Backward pass will show autograd spans
    info!("Running backward pass");
    let w_grad = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0)],
        &[1],
    )?;
    backward_with_grad(&w, &w_grad)?;

    // Check gradients
    match w.grad() {
        Ok(grad) => info!("Gradient computed successfully: {:?}", grad.as_slice()),
        Err(e) => info!("No gradient available: {:?}", e),
    }

    info!("Tracing example completed successfully");

    // Flush stdout to ensure all tracing output is visible
    io::stdout().flush()?;

    Ok(())
}

