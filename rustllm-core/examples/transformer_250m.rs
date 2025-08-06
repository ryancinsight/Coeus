//! Example demonstrating a 250M parameter transformer model.

use rustllm_core::prelude::*;
use rustllm_core::foundation::error::internal_error;
use rustllm_core::core::model::ForwardModel;
use rustllm_model_transformer::{TransformerModelPlugin, TransformerModelBuilder};
use std::time::Instant;

fn main() -> Result<()> {
    println!("RustLLM Core 250M Parameter Transformer Example");
    println!("==============================================\n");
    
    // Example 1: Model Configuration
    println!("Example 1: Model Configuration");
    println!("-----------------------------");
    
    let config = Transformer250MConfig::new();
    println!("Model configuration:");
    println!("  Hidden dimensions: {}", config.hidden_dim);
    println!("  Number of layers: {}", config.num_layers);
    println!("  Number of heads: {}", config.num_heads);
    println!("  Intermediate dim: {}", config.intermediate_dim);
    println!("  Vocabulary size: {}", config.vocab_size);
    println!("  Max sequence length: {}", config.max_seq_len);
    println!("  Dropout: {}", config.dropout);
    println!("  Layer norm epsilon: {:e}", config.layer_norm_eps);
    
    let param_count = config.num_parameters();
    println!("\nEstimated parameters: {} ({:.1}M)", param_count, param_count as f64 / 1_000_000.0);
    println!("Memory required: {:.1} MB (float32)", param_count as f64 * 4.0 / 1_000_000.0);
    
    // Example 2: Building the Model
    println!("\n\nExample 2: Building the Model");
    println!("-----------------------------");
    
    let start = Instant::now();
    let builder = TransformerModelBuilder::default();
    let model = builder.build(config.clone())?;
    let build_time = start.elapsed();
    
    println!("Model built in {:.2}s", build_time.as_secs_f64());
    println!("Actual parameters: {}", model.num_parameters());
    
    // Example 3: Forward Pass
    println!("\n\nExample 3: Forward Pass");
    println!("----------------------");
    
    // Simulate some token IDs (within vocabulary range)
    let input_tokens = vec![1234, 5678, 9101, 12131, 41516]; // Random token IDs
    println!("Input tokens: {:?}", input_tokens);
    println!("Input length: {}", input_tokens.len());
    
    let start = Instant::now();
    let logits = model.forward(input_tokens)?;
    let inference_time = start.elapsed();
    
    println!("Output logits shape: [{}]", logits.len());
    println!("Inference time: {:.3}ms", inference_time.as_micros() as f64 / 1000.0);
    
    // Show top 5 predictions
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 5 predicted tokens:");
    for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
        println!("  {}. Token {} with logit {:.4}", i + 1, token_id, logit);
    }
    
    // Example 4: Model Serialization
    println!("\n\nExample 4: Model Serialization");
    println!("------------------------------");
    
    use std::fs::File;
    use rustllm_core::core::serialization::ModelSerializable;
    
    let model_path = "transformer_250m.rustllm";
    
    // Save model
    println!("Saving model to {}...", model_path);
    let start = Instant::now();
    let mut file = File::create(model_path)
        .map_err(|e| internal_error(format!("Failed to create file: {}", e)))?;
    model.write_to(&mut file)?;
    let save_time = start.elapsed();
    
    let file_size = std::fs::metadata(model_path)
        .map_err(|e| internal_error(format!("Failed to get file metadata: {}", e)))?
        .len();
    
    println!("Model saved in {:.2}s", save_time.as_secs_f64());
    println!("File size: {:.1} MB", file_size as f64 / 1_000_000.0);
    
    // Load model back
    println!("\nLoading model from {}...", model_path);
    let start = Instant::now();
    let mut file = File::open(model_path)
        .map_err(|e| internal_error(format!("Failed to open file: {}", e)))?;
    let loaded_model = <TransformerModel as ModelSerializable>::read_from(&mut file)?;
    let load_time = start.elapsed();
    
    println!("Model loaded in {:.2}s", load_time.as_secs_f64());
    println!("Loaded parameters: {}", loaded_model.num_parameters());
    
    // Verify loaded model works
    let test_input = vec![100, 200, 300];
    let test_output = loaded_model.forward(test_input)?;
    println!("Loaded model inference successful, output shape: [{}]", test_output.len());
    
    // Clean up
    std::fs::remove_file(model_path)
        .map_err(|e| internal_error(format!("Failed to remove file: {}", e)))?;
    
    // Example 5: Plugin Usage
    println!("\n\nExample 5: Plugin Usage");
    println!("----------------------");
    
    let plugin = TransformerModelPlugin::default();
    println!("Plugin name: {}", plugin.id());
    println!("Plugin version: {}", plugin.version());
    println!("Plugin capabilities: {:?}", plugin.capabilities());
    
    // Register with plugin manager
    let manager = PluginManager::new();
    manager.register::<TransformerModelPlugin>()?;
    
    let available = manager.list_registered();
    println!("\nAvailable plugins: {:?}", available);
    
    // Example 6: Performance Metrics
    println!("\n\nExample 6: Performance Metrics");
    println!("-----------------------------");
    
    let batch_sizes = vec![1, 4, 8, 16];
    let seq_lengths = vec![128, 256, 512];
    
    println!("Theoretical throughput estimates:");
    println!("(Assuming 100 GFLOPS compute capability)");
    
    for &batch_size in &batch_sizes {
        for &seq_len in &seq_lengths {
            // Rough estimate of FLOPs for transformer
            // ~2 * params * seq_len for forward pass
            let flops = 2.0 * param_count as f64 * seq_len as f64 * batch_size as f64;
            let time_ms = flops / (100.0 * 1e9) * 1000.0; // 100 GFLOPS
            let throughput = batch_size as f64 / (time_ms / 1000.0);
            
            println!("  Batch={}, Seq={}: ~{:.1}ms, ~{:.1} samples/sec",
                     batch_size, seq_len, time_ms, throughput);
        }
    }
    
    println!("\n\nAll examples completed successfully!");
    
    Ok(())
}

// Re-export the TransformerModel type for the example
use rustllm_model_transformer::TransformerModel;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transformer_example() {
        assert!(main().is_ok());
    }
}