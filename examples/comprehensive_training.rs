//! # Comprehensive Training Example
//!
//! This example demonstrates a complete machine learning workflow using the Coeus framework,
//! including data loading, model definition, training with monitoring, and evaluation.
//!
//! ## Features Demonstrated
//!
//! - Model architecture definition
//! - Data preprocessing and loading
//! - Training loop with monitoring
//! - Validation and early stopping
//! - Model checkpointing
//! - Performance profiling
//! - Inference and evaluation
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example comprehensive_training
//! ```

use std::collections::HashMap;
use std::sync::Arc;

// Import Coeus components
use coeus_tensor::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{
    Linear, Sequential, MSELoss, CrossEntropyLoss, Module,
    SGD, Adam, TrainingMonitor, TrainingMetrics,
};
use coeus_profiling::{Timer, Profiler};
use coeus_storage::DenseStorage;
use coeus_tensor::{Shape, Tensor};

/// Synthetic dataset for demonstration
struct SyntheticDataset {
    pub inputs: Vec<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
    pub targets: Vec<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>,
}

impl SyntheticDataset {
    /// Create a synthetic dataset for binary classification
    fn new_binary_classification(num_samples: usize, input_dim: usize) -> Self {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..num_samples {
            // Generate random input
            let input = Tensor::randn(Shape::from(vec![input_dim])).unwrap();
            inputs.push(input);

            // Generate target based on simple rule (for demonstration)
            let target_value = if input.data()[0].0 > 0.0 { 1.0 } else { 0.0 };
            let target = Tensor::from_vec(vec![Float32(target_value)], Shape::from(vec![1])).unwrap();
            targets.push(target);
        }

        Self { inputs, targets }
    }

    /// Get batch from dataset
    fn get_batch(&self, start_idx: usize, batch_size: usize) -> (Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>) {
        let end_idx = (start_idx + batch_size).min(self.inputs.len());

        // Stack inputs into batch
        let batch_inputs: Vec<_> = self.inputs[start_idx..end_idx].iter()
            .map(|t| t.unsqueeze(0).unwrap())
            .collect();

        let batch_targets: Vec<_> = self.targets[start_idx..end_idx].iter()
            .map(|t| t.unsqueeze(0).unwrap())
            .collect();

        // Concatenate along batch dimension
        let batched_inputs = Tensor::cat(&batch_inputs, 0).unwrap();
        let batched_targets = Tensor::cat(&batch_targets, 0).unwrap();

        (batched_inputs, batched_targets)
    }

    fn len(&self) -> usize {
        self.inputs.len()
    }
}

/// Neural network model for binary classification
fn create_model(input_dim: usize, hidden_dim: usize) -> Sequential<Box<dyn Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>> {
    Sequential::new(vec![
        // Input layer
        Box::new(Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(input_dim, hidden_dim).unwrap()),
        // Hidden layer
        Box::new(Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(hidden_dim, hidden_dim / 2).unwrap()),
        // Output layer (binary classification)
        Box::new(Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(hidden_dim / 2, 1).unwrap()),
    ])
}

/// Training configuration
struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub patience: usize, // Early stopping patience
    pub validation_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 32,
            num_epochs: 10,
            patience: 3,
            validation_split: 0.2,
        }
    }
}

/// Training metrics collector
struct MetricsCollector {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub train_accuracies: Vec<f32>,
    pub val_accuracies: Vec<f32>,
    pub best_val_loss: f32,
    pub best_epoch: usize,
    pub no_improvement_count: usize,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            best_val_loss: f32::INFINITY,
            best_epoch: 0,
            no_improvement_count: 0,
        }
    }

    fn update(&mut self, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.train_accuracies.push(train_acc);
        self.val_accuracies.push(val_acc);

        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.best_epoch = self.val_losses.len() - 1;
            self.no_improvement_count = 0;
        } else {
            self.no_improvement_count += 1;
        }
    }

    fn should_early_stop(&self, patience: usize) -> bool {
        self.no_improvement_count >= patience
    }

    fn print_summary(&self) {
        println!("\n📊 Training Summary");
        println!("Best validation loss: {:.4} (epoch {})", self.best_val_loss, self.best_epoch);
        println!("Final training loss: {:.4}", self.train_losses.last().unwrap_or(&0.0));
        println!("Final validation loss: {:.4}", self.val_losses.last().unwrap_or(&0.0));
        println!("Final training accuracy: {:.2}%", self.train_accuracies.last().unwrap_or(&0.0) * 100.0);
        println!("Final validation accuracy: {:.2}%", self.val_accuracies.last().unwrap_or(&0.0) * 100.0);
    }
}

/// Calculate binary classification accuracy
fn calculate_accuracy(predictions: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, targets: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>) -> f32 {
    let pred_data = predictions.data();
    let target_data = targets.data();

    let mut correct = 0;
    let mut total = 0;

    for (pred, target) in pred_data.iter().zip(target_data.iter()) {
        let pred_class = if pred.0 > 0.0 { 1.0 } else { 0.0 };
        let target_class = target.0;

        if (pred_class - target_class).abs() < 0.5 {
            correct += 1;
        }
        total += 1;
    }

    correct as f32 / total as f32
}

/// Training loop with monitoring and early stopping
fn train_model(
    model: &mut Sequential<Box<dyn Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>>,
    train_dataset: &SyntheticDataset,
    val_dataset: &SyntheticDataset,
    config: &TrainingConfig,
) -> Result<MetricsCollector, Box<dyn std::error::Error>> {
    println!("🚀 Starting training...");

    // Initialize optimizer
    let mut optimizer = Adam::new(config.learning_rate).unwrap();

    // Initialize loss function
    let loss_fn = MSELoss::new();

    // Initialize metrics collector
    let mut metrics = MetricsCollector::new();

    // Initialize training monitor
    let mut monitor = TrainingMonitor::new();

    // Initialize profiler
    let profiler = Profiler::new();

    for epoch in 0..config.num_epochs {
        println!("\n📈 Epoch {}/{}", epoch + 1, config.num_epochs);

        // Training phase
        let mut epoch_train_loss = 0.0;
        let mut epoch_train_correct = 0;
        let mut epoch_train_total = 0;

        let num_train_batches = train_dataset.len() / config.batch_size;

        for batch_idx in 0..num_train_batches {
            let (inputs, targets) = train_dataset.get_batch(batch_idx * config.batch_size, config.batch_size);

            // Profile forward pass
            let forward_profile = profiler.profile(|| {
                model.forward(&inputs).unwrap()
            });

            let outputs = model.forward(&inputs).unwrap();
            let loss = loss_fn.forward(&outputs, &targets).unwrap();

            // Backward pass
            loss.backward().unwrap();

            // Optimizer step
            optimizer.step(model).unwrap();
            optimizer.zero_grad(model).unwrap();

            // Accumulate metrics
            epoch_train_loss += loss.item();
            let batch_accuracy = calculate_accuracy(&outputs, &targets);
            epoch_train_correct += (batch_accuracy * config.batch_size as f32) as usize;
            epoch_train_total += config.batch_size;

            // Record training metrics
            monitor.record_metrics(TrainingMetrics {
                epoch,
                step: epoch * num_train_batches + batch_idx,
                loss: loss.item(),
                learning_rate: optimizer.learning_rate(),
                gradient_norm: 0.1, // Simplified calculation
                step_time_ms: Some(forward_profile.mean_time.as_millis() as f32),
                ..Default::default()
            });
        }

        let avg_train_loss = epoch_train_loss / num_train_batches as f32;
        let train_accuracy = epoch_train_correct as f32 / epoch_train_total as f32;

        // Validation phase
        let mut epoch_val_loss = 0.0;
        let mut epoch_val_correct = 0;
        let mut epoch_val_total = 0;

        let num_val_batches = val_dataset.len() / config.batch_size;

        for batch_idx in 0..num_val_batches {
            let (inputs, targets) = val_dataset.get_batch(batch_idx * config.batch_size, config.batch_size);

            let outputs = model.forward(&inputs).unwrap();
            let loss = loss_fn.forward(&outputs, &targets).unwrap();

            epoch_val_loss += loss.item();
            let batch_accuracy = calculate_accuracy(&outputs, &targets);
            epoch_val_correct += (batch_accuracy * config.batch_size as f32) as usize;
            epoch_val_total += config.batch_size;
        }

        let avg_val_loss = epoch_val_loss / num_val_batches as f32;
        let val_accuracy = epoch_val_correct as f32 / epoch_val_total as f32;

        // Update metrics
        metrics.update(avg_train_loss, avg_val_loss, train_accuracy, val_accuracy);

        println!("  Train Loss: {:.4}, Train Acc: {:.2}%, Val Loss: {:.4}, Val Acc: {:.2}%",
                avg_train_loss, train_accuracy * 100.0, avg_val_loss, val_accuracy * 100.0);

        // Early stopping check
        if metrics.should_early_stop(config.patience) {
            println!("⏹️  Early stopping triggered after {} epochs without improvement", config.patience);
            break;
        }
    }

    // Generate training report
    let report = monitor.generate_report();
    println!("\n📋 Detailed Training Report:");
    println!("{}", report.summary());

    Ok(metrics)
}

/// Evaluate model on test data
fn evaluate_model(
    model: &Sequential<Box<dyn Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>>,
    test_dataset: &SyntheticDataset,
    batch_size: usize,
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    println!("\n🧪 Evaluating model...");

    let loss_fn = MSELoss::new();
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_samples = 0;

    let num_batches = test_dataset.len() / batch_size;

    for batch_idx in 0..num_batches {
        let (inputs, targets) = test_dataset.get_batch(batch_idx * batch_size, batch_size);

        let outputs = model.forward(&inputs).unwrap();
        let loss = loss_fn.forward(&outputs, &targets).unwrap();

        total_loss += loss.item();
        let batch_accuracy = calculate_accuracy(&outputs, &targets);
        total_correct += (batch_accuracy * batch_size as f32) as usize;
        total_samples += batch_size;
    }

    let avg_loss = total_loss / num_batches as f32;
    let accuracy = total_correct as f32 / total_samples as f32;

    println!("📊 Test Results:");
    println!("  Loss: {:.4}", avg_loss);
    println!("  Accuracy: {:.2}%", accuracy * 100.0);

    Ok((avg_loss, accuracy))
}

/// Save model checkpoint
fn save_checkpoint(
    model: &Sequential<Box<dyn Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>>,
    epoch: usize,
    loss: f32,
    accuracy: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Saving checkpoint...");

    // Create mock optimizer for demonstration
    use coeus_nn::SGD;
    let optimizer = SGD::new(0.01).unwrap();

    let mut metadata = HashMap::new();
    metadata.insert("epoch".to_string(), epoch.to_string());
    metadata.insert("loss".to_string(), loss.to_string());
    metadata.insert("accuracy".to_string(), accuracy.to_string());

    let checkpoint_path = format!("comprehensive_training_checkpoint_epoch_{}.json", epoch);
    coeus_nn::save_checkpoint(model, &optimizer, &metadata, &checkpoint_path)?;

    println!("✅ Checkpoint saved to: {}", checkpoint_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Coeus Comprehensive Training Example");
    println!("=======================================");

    // Configuration
    let config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 32,
        num_epochs: 20,
        patience: 5,
        validation_split: 0.2,
    };

    // Create datasets
    println!("\n📚 Creating datasets...");
    let full_dataset = SyntheticDataset::new_binary_classification(1000, 20);

    // Split into train/validation/test
    let train_size = (full_dataset.len() as f32 * (1.0 - config.validation_split - 0.1)) as usize;
    let val_size = (full_dataset.len() as f32 * config.validation_split) as usize;

    let train_dataset = SyntheticDataset {
        inputs: full_dataset.inputs[..train_size].to_vec(),
        targets: full_dataset.targets[..train_size].to_vec(),
    };

    let val_dataset = SyntheticDataset {
        inputs: full_dataset.inputs[train_size..train_size + val_size].to_vec(),
        targets: full_dataset.targets[train_size..train_size + val_size].to_vec(),
    };

    let test_dataset = SyntheticDataset {
        inputs: full_dataset.inputs[train_size + val_size..].to_vec(),
        targets: full_dataset.targets[train_size + val_size..].to_vec(),
    };

    println!("  Train samples: {}", train_dataset.len());
    println!("  Validation samples: {}", val_dataset.len());
    println!("  Test samples: {}", test_dataset.len());

    // Create model
    println!("\n🏗️  Creating model...");
    let mut model = create_model(20, 64);
    println!("  Model architecture: 20 -> 64 -> 32 -> 1");

    // Count parameters
    let total_params = model.parameters().count();
    println!("  Total parameters: {}", total_params);

    // Train model
    let metrics = train_model(&mut model, &train_dataset, &val_dataset, &config)?;

    // Save checkpoint
    if let (Some(final_loss), Some(final_acc)) = (metrics.val_losses.last(), metrics.val_accuracies.last()) {
        save_checkpoint(&model, metrics.best_epoch, *final_loss, *final_acc)?;
    }

    // Evaluate on test set
    let (test_loss, test_accuracy) = evaluate_model(&model, &test_dataset, config.batch_size)?;

    // Final summary
    metrics.print_summary();
    println!("\n🎯 Final Test Performance:");
    println!("  Loss: {:.4}", test_loss);
    println!("  Accuracy: {:.2}%", test_accuracy * 100.0);

    println!("\n✅ Training complete! The model has been trained, evaluated, and saved.");
    println!("💡 Next steps:");
    println!("  - Load the checkpoint for inference");
    println!("  - Deploy the model to production");
    println!("  - Experiment with different architectures");
    println!("  - Try distributed training for larger datasets");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset() {
        let dataset = SyntheticDataset::new_binary_classification(100, 10);
        assert_eq!(dataset.len(), 100);

        let (inputs, targets) = dataset.get_batch(0, 10);
        assert_eq!(inputs.shape(), &Shape::from(vec![10, 10]));
        assert_eq!(targets.shape(), &Shape::from(vec![10, 1]));
    }

    #[test]
    fn test_model_creation() {
        let model = create_model(20, 64);
        let input = Tensor::randn(Shape::from(vec![1, 20])).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &Shape::from(vec![1, 1]));
    }

    #[test]
    fn test_metrics_collector() {
        let mut metrics = MetricsCollector::new();
        metrics.update(0.5, 0.4, 0.8, 0.85);

        assert_eq!(metrics.train_losses.len(), 1);
        assert_eq!(metrics.val_losses.len(), 1);
        assert_eq!(metrics.best_val_loss, 0.4);
        assert_eq!(metrics.best_epoch, 0);
        assert!(!metrics.should_early_stop(3));
    }

    #[test]
    fn test_accuracy_calculation() {
        let predictions = Tensor::from_vec(vec![Float32(0.6), Float32(-0.3)], Shape::from(vec![2, 1])).unwrap();
        let targets = Tensor::from_vec(vec![Float32(1.0), Float32(0.0)], Shape::from(vec![2, 1])).unwrap();

        let accuracy = calculate_accuracy(&predictions, &targets);
        assert!((accuracy - 0.5).abs() < 0.1); // Approximately 0.5 (1/2 correct)
    }
}

