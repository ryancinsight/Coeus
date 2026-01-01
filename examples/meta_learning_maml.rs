//! Meta-Learning Example with MAML (Model-Agnostic Meta-Learning)
//!
//! This example demonstrates how to use MAML for few-shot learning on regression tasks.
//! MAML learns model parameters that can be quickly adapted to new tasks with just a few examples.

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Linear, MAML};
use rand::Rng;
use storage::DenseStorage;

/// Sine wave regression task generator
struct SineTaskGenerator {
    /// Amplitude range
    amplitude_range: (f64, f64),
    /// Phase range
    phase_range: (f64, f64),
    /// Number of support examples per task
    num_support: usize,
    /// Number of query examples per task
    num_query: usize,
}

impl SineTaskGenerator {
    pub fn new(amplitude_range: (f64, f64), phase_range: (f64, f64)) -> Self {
        Self {
            amplitude_range,
            phase_range,
            num_support: 10,
            num_query: 10,
        }
    }

    /// Generate a task consisting of sine function examples
    pub fn generate_task(&self) -> Result<Task, Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        // Random amplitude and phase
        let amplitude = rng.gen_range(self.amplitude_range.0..=self.amplitude_range.1);
        let phase = rng.gen_range(self.phase_range.0..=self.phase_range.1);

        let mut support_set = Vec::new();
        let mut query_set = Vec::new();

        // Generate support set
        for _ in 0..self.num_support {
            let x = rng.gen_range(-2.0..=2.0);
            let y = amplitude * (x + phase).sin();
            support_set.push((x, y));
        }

        // Generate query set
        for _ in 0..self.num_query {
            let x = rng.gen_range(-2.0..=2.0);
            let y = amplitude * (x + phase).sin();
            query_set.push((x, y));
        }

        Ok(Task {
            support_set,
            query_set,
            task_id: format!("sine_task_{}", rng.gen::<u64>()),
        })
    }
}

/// Task definition for MAML
struct Task {
    pub support_set: Vec<(f64, f64)>,
    pub query_set: Vec<(f64, f64)>,
    pub task_id: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 MAML Meta-Learning Example");
    println!("=============================");

    // Create base model (simple 2-layer MLP)
    let base_model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, 1)?;

    println!(
        "Base model: {} -> {}",
        base_model.in_features, base_model.out_features
    );

    // Create task generator for sine wave regression tasks
    let task_generator = SineTaskGenerator::new((0.1, 5.0), (0.0, std::f64::consts::PI));

    // Set up MAML with reasonable hyperparameters
    let mut maml = MAML::new(base_model)
        .with_inner_lr(0.01) // Inner loop learning rate
        .with_outer_lr(0.001) // Outer loop learning rate
        .with_inner_steps(5) // Adaptation steps per task
        .with_first_order(true); // Use first-order approximation

    // Configure task distribution
    maml = maml.with_task_distribution(move || {
        task_generator
            .generate_task()
            .map_err(|e| crate::error::NNError::InvalidConfiguration {
                message: format!("Task generation failed: {}", e),
            })
    });

    println!("\nStarting meta-training...");
    println!("Tasks per step: 4, Total iterations: 100");

    // Train for several iterations
    let losses = maml.train(100, 4)?;

    println!("\nMeta-training completed!");
    println!("Final loss: {:.6}", losses.last().unwrap_or(&0.0));

    // Demonstrate few-shot adaptation
    println!("\nDemonstrating few-shot adaptation...");

    // Generate a new task for testing
    let test_task = task_generator.generate_task()?;
    println!(
        "Test task: {} support examples, {} query examples",
        test_task.support_set.len(),
        test_task.query_set.len()
    );

    // Adapt to the new task
    let adapted_model = maml.adapt_for_inference(
        &test_task
            .support_set
            .iter()
            .map(|(x, y)| {
                (
                    tensor::Tensor::from_vec(vec![Float32::from(*x)], &[1]).unwrap(),
                    tensor::Tensor::from_vec(vec![Float32::from(*y)], &[1]).unwrap(),
                )
            })
            .collect::<Vec<_>>(),
        Some(10), // 10 adaptation steps
    )?;

    // Evaluate on query set
    let mut predictions = Vec::new();
    let mut targets = Vec::new();

    for (x, y) in &test_task.query_set {
        let input = tensor::Tensor::from_vec(vec![Float32::from(*x)], &[1])?;
        let output = adapted_model.forward(&input)?;
        let pred = output.as_slice()[0].into();

        predictions.push(pred);
        targets.push(*y);
    }

    // Calculate MSE
    let mse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;

    println!("Few-shot adaptation MSE: {:.6}", mse);
    println!("✅ Meta-learning example completed successfully!");

    Ok(())
}
