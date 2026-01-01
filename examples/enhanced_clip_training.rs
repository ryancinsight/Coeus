//! Enhanced CLIP Training Example
//!
//! This example demonstrates the production-ready enhanced CLIP trainer
//! with gradient accumulation, learning rate scheduling, checkpointing,
//! and early stopping capabilities.
//!
//! Run with: cargo run --example enhanced_clip_training

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// Backend and tensor dependencies
use backend::CpuBackend;
use dtype::float::Float32;
use storage::{DenseStorage, StorageFromVec, StorageToDense};
use tensor::Tensor;

// NN modules
use nn::clip::{
    enhanced_trainer::{
        ClipBatch, EnhancedClipTrainer, EnhancedClipTrainingConfig, EnhancedTrainingReport,
    },
    ClipConfig,
};
use nn::datasets::{CocoDataset, DatasetSplit, Flickr30kDataset};
use nn::error::Result;

// Type aliases for clarity
type Backend = CpuBackend<Float32>;
type Storage = DenseStorage<Float32>;

/// Comprehensive enhanced CLIP training demonstration
async fn run_enhanced_clip_training() -> Result<()> {
    println!("🚀 Enhanced CLIP Training Demo");
    println!("==============================");

    let start_time = Instant::now();

    // Phase 1: Setup Enhanced Trainer Configuration
    println!("\n📋 Phase 1: Configuring Enhanced CLIP Trainer");
    println!("--------------------------------------------");

    let enhanced_config = EnhancedClipTrainingConfig {
        base_config: crate::clip::trainer::ClipTrainingConfig {
            clip_config: ClipConfig {
                vision_config: nn::clip::VisionConfig {
                    image_size: 224,
                    patch_size: 16,
                    hidden_size: 768,
                    num_layers: 12,
                    num_heads: 12,
                    mlp_ratio: 4.0,
                },
                text_config: nn::clip::TextConfig {
                    vocab_size: 49408,
                    max_position_embeddings: 77,
                    hidden_size: 512,
                    num_layers: 12,
                    num_heads: 8,
                    mlp_ratio: 4.0,
                },
                projection_dim: 512,
                temperature: 0.07,
            },
            learning_rate: 5e-4,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.2,
            num_epochs: 2, // Reduced for demo
            batch_size: 8, // Small batch for demo
            total_steps: 100,
            warmup_steps: 10,
            log_steps: 5,
            save_steps: 50,
            eval_steps: 25,
            output_dir: "./clip_output".to_string(),
        },
        gradient_accumulation_steps: 4, // Effective batch size = 32
        early_stopping_patience: 5,
        min_learning_rate: 1e-6,
        save_best_only: true,
        checkpoint_dir: "./checkpoints".to_string(),
        resume_from: None,
        max_grad_norm: 1.0,
        log_gradients: false,
    };

    println!("✅ Enhanced configuration created:");
    println!(
        "   - Gradient accumulation: {} steps",
        enhanced_config.gradient_accumulation_steps
    );
    println!(
        "   - Effective batch size: {}",
        enhanced_config.base_config.batch_size * enhanced_config.gradient_accumulation_steps
    );
    println!(
        "   - Early stopping patience: {}",
        enhanced_config.early_stopping_patience
    );
    println!(
        "   - Checkpoint directory: {}",
        enhanced_config.checkpoint_dir.display()
    );

    // Phase 2: Load Dataset
    println!("\n📚 Phase 2: Loading Training Dataset");
    println!("------------------------------------");

    // Create a synthetic data generator for demonstration
    let data_generator = SyntheticDataGenerator::new(100, enhanced_config.base_config.batch_size);

    println!(
        "✅ Using synthetic data generator with {} samples",
        data_generator.total_samples
    );

    // Phase 3: Initialize Enhanced Trainer
    println!("\n🏗️  Phase 3: Initializing Enhanced CLIP Trainer");
    println!("---------------------------------------------");

    let trainer = EnhancedClipTrainer::<Backend, Storage, Float32>::new(enhanced_config)?;

    println!("✅ Enhanced CLIP trainer initialized");
    println!(
        "   - Model: {} vision layers, {} text layers",
        trainer.model.vision_config().num_layers,
        trainer.model.text_config().num_layers
    );
    println!(
        "   - Optimizer: Adam with LR={}, WD={}",
        trainer.config.base_config.learning_rate, trainer.config.base_config.weight_decay
    );

    // Phase 4: Training Loop with Enhanced Features
    println!("\n🎯 Phase 4: Enhanced Training Execution");
    println!("--------------------------------------");

    let training_result = trainer
        .train(&mut |batch_idx| data_generator.get_batch(batch_idx))
        .await?;

    // Phase 5: Training Results Analysis
    println!("\n📊 Phase 5: Training Results Analysis");
    println!("------------------------------------");

    println!("🏆 Training completed successfully!");
    println!("📈 Final Results:");
    println!("   - Total epochs: {}", training_result.total_epochs);
    println!("   - Total steps: {}", training_result.total_steps);
    println!(
        "   - Best validation loss: {:.6}",
        training_result.best_validation_loss
    );
    println!(
        "   - Final learning rate: {:.6}",
        training_result.final_learning_rate
    );
    println!(
        "   - Total training time: {:.2}s",
        training_result.total_training_time
    );
    println!(
        "   - Training throughput: {:.1} samples/sec",
        training_result.training_samples as f64 / training_result.total_training_time
    );

    // Phase 6: Checkpoint Analysis
    println!("\n💾 Phase 6: Checkpoint Analysis");
    println!("-------------------------------");

    let training_state = trainer.training_state();
    println!("📋 Training state summary:");
    println!("   - Current epoch: {}", training_state.epoch);
    println!("   - Current step: {}", training_state.step);
    println!(
        "   - Best validation loss: {:.6}",
        training_state.best_val_loss
    );
    println!("   - Steps since best: {}", training_state.steps_since_best);
    println!(
        "   - Total samples processed: {}",
        training_state.total_samples
    );
    println!(
        "   - Learning rate history: {} entries",
        training_state.lr_history.len()
    );
    println!(
        "   - Loss history: {} entries",
        training_state.loss_history.len()
    );
    println!(
        "   - Validation loss history: {} entries",
        training_state.val_loss_history.len()
    );

    // Phase 7: Performance Analysis
    println!("\n⚡ Phase 7: Performance Analysis");
    println!("-------------------------------");

    let total_time = start_time.elapsed().as_secs_f64();
    println!("⏱️  Total demo time: {:.2}s", total_time);
    println!(
        "🚀 Training efficiency: {:.1} samples/sec",
        training_result.training_samples as f64 / training_result.total_training_time
    );

    // Demonstrate enhanced features
    println!("\n🎨 Enhanced Features Demonstrated:");
    println!(
        "   ✅ Gradient accumulation ({}-step)",
        enhanced_config.gradient_accumulation_steps
    );
    println!("   ✅ Cosine learning rate scheduling");
    println!("   ✅ Checkpoint saving/loading");
    println!("   ✅ Early stopping logic");
    println!("   ✅ Comprehensive training state tracking");
    println!("   ✅ Production-ready error handling");

    println!("\n🎉 Enhanced CLIP Training Demo Complete!");
    println!("=========================================");

    Ok(())
}

/// Synthetic data generator for demonstration
struct SyntheticDataGenerator {
    total_samples: usize,
    batch_size: usize,
    current_batch: usize,
}

impl SyntheticDataGenerator {
    fn new(total_samples: usize, batch_size: usize) -> Self {
        Self {
            total_samples,
            batch_size,
            current_batch: 0,
        }
    }

    fn get_batch(&mut self, _batch_idx: usize) -> Option<ClipBatch> {
        if self.current_batch * self.batch_size >= self.total_samples {
            return None; // End of data
        }

        // Generate synthetic batch
        let seq_length = 77; // CLIP sequence length
        let image_size = 224 * 224 * 3; // RGB image flattened

        let images: Vec<f32> = (0..self.batch_size * image_size)
            .map(|i| {
                // Generate pseudo-random normalized pixel values
                let val = ((i * 31) % 255) as f32 / 255.0;
                // Apply simple normalization (CLIP style)
                (val - 0.5) / 0.5
            })
            .collect();

        let text_tokens: Vec<u32> = (0..self.batch_size * seq_length)
            .map(|i| {
                // Generate synthetic token IDs (CLIP vocab range)
                ((i * 7 + self.current_batch) % 49400 + 100) as u32
            })
            .collect();

        let text_masks: Vec<u32> = (0..self.batch_size * seq_length)
            .map(|i| if i % seq_length < 10 { 1 } else { 0 }) // Simple mask
            .collect();

        let batch = ClipBatch {
            images,
            text_tokens,
            text_masks,
            batch_size: self.batch_size,
            seq_length,
        };

        self.current_batch += 1;
        Some(batch)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Enhanced CLIP Training Example");
    println!("=================================");

    match run_enhanced_clip_training().await {
        Ok(()) => {
            println!("\n✅ Enhanced CLIP training demo completed successfully!");
        }
        Err(e) => {
            eprintln!("❌ Enhanced training demo failed: {}", e);
            eprintln!("\n💡 This demo showcases the enhanced CLIP trainer with:");
            eprintln!("   - Gradient accumulation for larger effective batch sizes");
            eprintln!("   - Cosine annealing learning rate scheduling");
            eprintln!("   - Comprehensive checkpointing and state management");
            eprintln!("   - Early stopping based on validation metrics");
            eprintln!("   - Production-ready training infrastructure");

            std::process::exit(1);
        }
    }

    Ok(())
}
