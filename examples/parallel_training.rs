//! Parallel Training Example: Concurrent Data Processing
//!
//! This example demonstrates parallel data processing patterns for training
//! using Rust's concurrency primitives (Arc, RwLock) for thread-safe shared state.
//!
//! Note: Full multi-threaded model training requires Send+Sync trait objects,
//! which is a known limitation. This example demonstrates the architectural
//! pattern for concurrent data processing and state management.
//!
//! Sprint 7.7: Advanced Example Development
//!
//! Run with: cargo run --example parallel_training

use dtype::float::Float32;
use nn::{Linear, Module};
use std::sync::{Arc, RwLock};
use std::thread;
use storage::DenseStorage;
use tensor::CpuBackend;
use tensor::Tensor;

/// Type alias for the complex tensor type used in this example
type TrainingTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Shared data processing state for parallel workers
///
/// This structure demonstrates thread-safe shared state management
/// for concurrent data processing pipelines.
struct SharedProcessingState {
    processed_batches: Arc<RwLock<Vec<f32>>>,
    total_loss: Arc<RwLock<f32>>,
    iterations: Arc<RwLock<usize>>,
}

impl SharedProcessingState {
    fn new() -> Self {
        Self {
            processed_batches: Arc::new(RwLock::new(Vec::new())),
            total_loss: Arc::new(RwLock::new(0.0)),
            iterations: Arc::new(RwLock::new(0)),
        }
    }

    fn clone_refs(&self) -> Self {
        Self {
            processed_batches: Arc::clone(&self.processed_batches),
            total_loss: Arc::clone(&self.total_loss),
            iterations: Arc::clone(&self.iterations),
        }
    }
}

/// Simulate a training batch
///
/// In production, this would load real data from a dataset.
fn generate_training_batch(
    batch_id: usize,
) -> Result<(TrainingTensor, TrainingTensor), Box<dyn std::error::Error>> {
    // Generate synthetic input: [batch_size=2, features=4]
    let input = TrainingTensor::from_vec(
        vec![
            Float32::new((batch_id as f32) * 0.1),
            Float32::new((batch_id as f32) * 0.2),
            Float32::new((batch_id as f32) * 0.3),
            Float32::new((batch_id as f32) * 0.4),
            Float32::new((batch_id as f32) * 0.5),
            Float32::new((batch_id as f32) * 0.6),
            Float32::new((batch_id as f32) * 0.7),
            Float32::new((batch_id as f32) * 0.8),
        ],
        &[2, 4],
    )?;

    // Generate synthetic target: [batch_size=2, features=4] (same shape as input)
    let target = TrainingTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(0.5),
            Float32::new(0.5),
            Float32::new(0.5),
            Float32::new(0.5),
        ],
        &[2, 4],
    )?;

    Ok((input, target))
}

/// Data processing worker function
///
/// Each worker processes batches independently and updates shared state.
fn processing_worker(worker_id: usize, state: SharedProcessingState, num_batches: usize) {
    println!("   Worker {} started", worker_id);

    for batch_id in 0..num_batches {
        // Generate training batch
        let (input, target) = match generate_training_batch(batch_id) {
            Ok(batch) => batch,
            Err(e) => {
                eprintln!("   Worker {} error generating batch: {}", worker_id, e);
                continue;
            }
        };

        // Simulate processing (compute simple loss without model)
        let diff = &input - &target;
        let squared = &diff * &diff;
        let loss_value =
            squared.as_slice().iter().map(|x| x.get()).sum::<f32>() / (squared.len() as f32);

        // Update shared state with thread-safe operations
        {
            let mut batches = state.processed_batches.write().unwrap();
            batches.push(loss_value);
        }

        {
            let mut total_loss = state.total_loss.write().unwrap();
            *total_loss += loss_value;
        }

        {
            let mut iterations = state.iterations.write().unwrap();
            *iterations += 1;
        }
    }

    println!("   Worker {} completed {} batches", worker_id, num_batches);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Parallel Data Processing Example");
    println!("====================================\n");

    // Demonstrate single-threaded model usage first
    println!("1. Single-Threaded Model Example");
    let model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 2).unwrap();
    let input = TrainingTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[1, 4],
    )?;
    let output = model.forward(&input)?;
    println!("   Model: Linear layer (4 -> 2)");
    println!("   Input shape: {:?}", input.shape().dims());
    println!("   Output shape: {:?}", output.shape().dims());
    println!("   ✅ Model forward pass successful\n");

    // Create shared processing state
    println!("2. Setting Up Parallel Data Processing");
    let state = SharedProcessingState::new();
    let num_workers = 4;
    let batches_per_worker = 5;
    println!("   Workers: {}", num_workers);
    println!("   Batches per worker: {}", batches_per_worker);
    println!("   Total batches: {}", num_workers * batches_per_worker);
    println!("   ✅ Processing state initialized\n");

    // Spawn worker threads
    println!("3. Spawning Worker Threads");
    let mut handles = vec![];

    for worker_id in 0..num_workers {
        let worker_state = state.clone_refs();
        let handle = thread::spawn(move || {
            processing_worker(worker_id, worker_state, batches_per_worker);
        });
        handles.push(handle);
    }

    // Wait for all workers to complete
    for handle in handles {
        handle.join().unwrap();
    }
    println!("   ✅ All workers completed\n");

    // Report results
    println!("4. Processing Results");
    let batches = state.processed_batches.read().unwrap();
    let total_loss = *state.total_loss.read().unwrap();
    let iterations = *state.iterations.read().unwrap();
    let avg_loss = total_loss / (iterations as f32);

    println!("   Total iterations: {}", iterations);
    println!("   Batches processed: {}", batches.len());
    println!("   Total loss: {:.4}", total_loss);
    println!("   Average loss: {:.4}", avg_loss);
    println!("   ✅ Processing completed\n");

    println!("✅ Parallel data processing example completed successfully!");
    println!("\n📚 Key Takeaways:");
    println!("   • Arc<RwLock<T>> enables thread-safe shared state");
    println!("   • Multiple workers can process data concurrently");
    println!("   • Read locks allow parallel data access");
    println!("   • Write locks synchronize state updates");
    println!("   • Rust's ownership prevents data races at compile time");
    println!("   • Models can be used in single-threaded contexts");
    println!("   • Full multi-threaded training requires Send+Sync trait objects");

    Ok(())
}
