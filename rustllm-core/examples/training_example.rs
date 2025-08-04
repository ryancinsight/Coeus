//! Example demonstrating training capabilities with zero-cost abstractions.
//!
//! This example shows how to use the TrainableModel trait with iterator
//! combinators for efficient batch processing.

use rustllm_core::prelude::*;
use std::f32::consts::PI;

/// Simple optimizer state implementation.
#[derive(Debug, Clone)]
struct AdamOptimizerState {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_count: usize,
    momentum: Vec<f32>,
    variance: Vec<f32>,
}

impl AdamOptimizerState {
    fn new(param_count: usize, learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step_count: 0,
            momentum: vec![0.0; param_count],
            variance: vec![0.0; param_count],
        }
    }
}

impl OptimizerState for AdamOptimizerState {
    fn reset(&mut self) {
        self.step_count = 0;
        self.momentum.fill(0.0);
        self.variance.fill(0.0);
    }
    
    fn learning_rate(&self) -> ModelFloat {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, lr: ModelFloat) {
        self.learning_rate = lr;
    }
    
    fn step_count(&self) -> usize {
        self.step_count
    }
    
    fn increment_step(&mut self) {
        self.step_count += 1;
    }
}

/// Mean squared error loss function.
#[derive(Debug, Clone)]
struct MSELoss;

impl Loss for MSELoss {
    fn compute(&self, predictions: &[ModelFloat], targets: &[ModelFloat]) -> Result<ModelFloat> {
        if predictions.len() != targets.len() {
            return Err(Error::Config("Predictions and targets must have same length".to_string()));
        }
        
        let mse = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f32>() / predictions.len() as f32;
        
        Ok(mse)
    }
    
    fn gradient(&self, predictions: &[ModelFloat], targets: &[ModelFloat]) -> Result<Vec<ModelFloat>> {
        if predictions.len() != targets.len() {
            return Err(Error::Config("Predictions and targets must have same length".to_string()));
        }
        
        let gradients = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| 2.0 * (pred - target) / predictions.len() as f32)
            .collect();
        
        Ok(gradients)
    }
}

/// Example trainable model - a simple sine wave predictor.
struct SinePredictor {
    weights: Vec<f32>,
    bias: f32,
    training: bool,
}

impl SinePredictor {
    fn new(input_dim: usize) -> Self {
        // Initialize with small random values (simplified for example)
        let weights = (0..input_dim)
            .map(|i| ((i as f32 + 1.0) / input_dim as f32 - 0.5) * 0.1)
            .collect();
        
        Self {
            weights,
            bias: 0.0,
            training: false,
        }
    }
}

impl Model for SinePredictor {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type Config = BasicModelConfig;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Simple linear transformation followed by sine activation
        let output = input.iter()
            .zip(self.weights.iter())
            .map(|(x, w)| (x * w + self.bias).sin())
            .collect();
        
        Ok(output)
    }
    
    fn config(&self) -> &Self::Config {
        // Simplified for example
        static CONFIG: BasicModelConfig = BasicModelConfig {
            model_dim: 128,
            head_count: 8,
            layer_count: 1,
            max_seq_len: 512,
            vocab_size: 1000,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        };
        &CONFIG
    }
    
    fn num_parameters(&self) -> usize {
        self.weights.len() + 1 // weights + bias
    }
}

impl TrainableModel for SinePredictor {
    type TrainingData = (Vec<f32>, Vec<f32>); // (input, target)
    type OptimizerState = AdamOptimizerState;
    type Loss = MSELoss;
    
    fn train_step(
        &mut self,
        batch: Self::TrainingData,
        optimizer_state: &mut Self::OptimizerState,
    ) -> Result<ModelFloat> {
        let (input, target) = batch;
        
        // Forward pass
        let predictions = self.forward(input.clone())?;
        
        // Compute loss
        let loss_fn = MSELoss;
        let loss = loss_fn.compute(&predictions, &target)?;
        
        // Compute gradients (simplified)
        let gradients = loss_fn.gradient(&predictions, &target)?;
        
        // Update parameters using optimizer
        self.update_parameters(&gradients, optimizer_state)?;
        
        Ok(loss)
    }
    
    fn compute_loss(&self, batch: Self::TrainingData) -> Result<ModelFloat> {
        let (input, target) = batch;
        let predictions = self.forward(input)?;
        MSELoss.compute(&predictions, &target)
    }
    
    fn update_parameters(
        &mut self,
        gradients: &[ModelFloat],
        optimizer_state: &mut Self::OptimizerState,
    ) -> Result<()> {
        optimizer_state.increment_step();
        let t = optimizer_state.step_count() as f32;
        
        // Adam optimizer update (simplified)
        for (i, grad) in gradients.iter().enumerate() {
            if i < self.weights.len() {
                // Update momentum
                optimizer_state.momentum[i] = optimizer_state.beta1 * optimizer_state.momentum[i]
                    + (1.0 - optimizer_state.beta1) * grad;
                
                // Update variance
                optimizer_state.variance[i] = optimizer_state.beta2 * optimizer_state.variance[i]
                    + (1.0 - optimizer_state.beta2) * grad * grad;
                
                // Bias correction
                let m_hat = optimizer_state.momentum[i] / (1.0 - optimizer_state.beta1.powf(t));
                let v_hat = optimizer_state.variance[i] / (1.0 - optimizer_state.beta2.powf(t));
                
                // Update weight
                self.weights[i] -= optimizer_state.learning_rate * m_hat / (v_hat.sqrt() + optimizer_state.epsilon);
            }
        }
        
        Ok(())
    }
    
    fn is_training(&self) -> bool {
        self.training
    }
    
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

fn main() -> Result<()> {
    println!("RustLLM Core Training Example");
    println!("=============================\n");
    
    // Create model and optimizer
    let mut model = SinePredictor::new(10);
    let mut optimizer = AdamOptimizerState::new(model.num_parameters(), 0.01);
    
    println!("Model initialized with {} parameters", model.num_parameters());
    
    // Generate training data using iterator combinators
    let training_data: Vec<(Vec<f32>, Vec<f32>)> = (0..100)
        .map(|i| {
            let x = i as f32 * 0.1;
            let input = vec![x; 10];
            let target = vec![(x * 2.0 * PI).sin(); 10];
            (input, target)
        })
        .collect();
    
    // Training loop using iterator combinators for efficient batch processing
    println!("\nTraining for 10 epochs...");
    model.set_training(true);
    
    for epoch in 0..10 {
        // Use lazy_batch iterator for efficient batching
        let epoch_loss: f32 = training_data.iter()
            .cloned()
            .lazy_batch(16)  // Batch size of 16
            .map(|batch| {
                // Process batch and accumulate loss
                batch.into_iter()
                    .map(|data| model.train_step(data, &mut optimizer).unwrap_or(0.0))
                    .sum::<f32>() / 16.0
            })
            .sum::<f32>() / (training_data.len() as f32 / 16.0);
        
        println!("Epoch {}: Average Loss = {:.6}", epoch + 1, epoch_loss);
    }
    
    model.set_training(false);
    
    // Test the trained model
    println!("\nTesting trained model:");
    for i in 0..5 {
        let x = i as f32 * 0.5;
        let input = vec![x; 10];
        let expected = (x * 2.0 * PI).sin();
        let output = model.forward(input)?;
        
        println!("Input: {:.2}, Expected: {:.4}, Predicted: {:.4}", 
                 x, expected, output[0]);
    }
    
    // Demonstrate zero-copy string building
    println!("\nZero-copy string building example:");
    let mut builder = ZeroCopyStringBuilder::new();
    builder
        .append_borrowed("Training completed with ")
        .append_owned(format!("{}", model.num_parameters()))
        .append_borrowed(" parameters");
    
    println!("{}", builder.build());
    
    // Demonstrate advanced iterator usage
    println!("\nAdvanced iterator example:");
    let values: Vec<f32> = (0..20)
        .map(|i| i as f32 * 0.1)
        .stream_map(|x| (x * PI).sin())
        .rolling_aggregate(5, |window| window.iter().sum::<f32>() / window.len() as f32)
        .take(10)
        .collect();
    
    println!("Rolling average of sine values: {:?}", values);
    
    Ok(())
}