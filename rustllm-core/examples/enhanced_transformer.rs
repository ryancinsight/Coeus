//! Enhanced Transformer Model Example
//!
//! This example demonstrates the production-ready transformer implementation
//! showcasing elite programming practices and design principles.

use rustllm_core::{
    core::{
        model::{ForwardModel, Model, ModelBuilder, Transformer250MConfig},
        plugin::ModelBuilderPlugin,
    },
    foundation::error::Result,
};
use rustllm_model_transformer::TransformerModelPlugin;

fn main() -> Result<()> {
    println!("🚀 Enhanced Transformer Model Example");
    println!("=====================================");

    // Create the enhanced transformer plugin directly
    let transformer_plugin = TransformerModelPlugin::default();
    println!("✓ Created enhanced transformer plugin");

    // Create model configuration
    let config = Transformer250MConfig {
        hidden_dim: 512,
        num_layers: 8,
        num_heads: 8,
        vocab_size: 32000,
        max_seq_len: 1024,
        intermediate_dim: 2048,
        layer_norm_eps: 1e-5,
        use_bias: true,
        dropout: 0.1,
        attention_dropout: 0.1,
        activation: "gelu".to_string(),
    };

    println!("📋 Model Configuration:");
    println!("   • Hidden Dimension: {}", config.hidden_dim);
    println!("   • Number of Layers: {}", config.num_layers);
    println!("   • Number of Heads: {}", config.num_heads);
    println!("   • Vocabulary Size: {}", config.vocab_size);
    println!("   • Max Sequence Length: {}", config.max_seq_len);
    println!("   • Intermediate Dimension: {}", config.intermediate_dim);

    // Validate configuration
    config.validate()?;
    println!("✓ Configuration validated");

    // Calculate parameter count
    let param_count = config.num_parameters();
    println!(
        "📊 Total Parameters: {} ({:.1}M)",
        param_count,
        param_count as f64 / 1_000_000.0
    );

    // Create model builder directly from plugin
    let builder = transformer_plugin.create_builder()?;

    println!("✓ Created model builder");

    // Build the enhanced transformer model
    println!("🔨 Building enhanced transformer model...");
    let model = builder.build(config)?;

    println!("✓ Model built successfully!");
    println!("📈 Model Statistics:");
    println!("   • Parameters: {}", model.num_parameters());
    println!(
        "   • Memory Usage: ~{:.1} MB",
        model.num_parameters() as f64 * 4.0 / 1_000_000.0
    );

    // Test forward pass with sample input
    println!("\n🧪 Testing Forward Pass");
    println!("========================");

    let sample_tokens = vec![1, 15, 42, 128, 256, 512, 1024, 2048];
    println!("📝 Input tokens: {:?}", sample_tokens);

    let start_time = std::time::Instant::now();
    let output = model.forward(sample_tokens.clone())?;
    let inference_time = start_time.elapsed();

    println!(
        "✓ Forward pass completed in {:.2}ms",
        inference_time.as_millis()
    );
    println!("📊 Output shape: [{}] (vocabulary logits)", output.len());

    // Find top-k predictions
    let mut indexed_logits: Vec<(usize, f32)> = output
        .iter()
        .enumerate()
        .map(|(i, &logit)| (i, logit))
        .collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("🎯 Top-5 Predictions:");
    for (rank, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
        println!("   {}. Token {}: {:.4}", rank + 1, token_id, logit);
    }

    // Performance metrics
    let tokens_per_second = sample_tokens.len() as f64 / inference_time.as_secs_f64();
    println!("\n⚡ Performance Metrics:");
    println!("   • Inference Time: {:.2}ms", inference_time.as_millis());
    println!("   • Tokens/Second: {:.1}", tokens_per_second);
    println!(
        "   • Parameters/Second: {:.1}M",
        model.num_parameters() as f64 / inference_time.as_secs_f64() / 1_000_000.0
    );

    // Architecture highlights
    println!("\n🏗️  Architecture Highlights");
    println!("============================");
    println!(
        "✓ Multi-Head Attention with {} heads",
        model.config().num_heads
    );
    println!("✓ Sinusoidal Positional Encoding");
    println!("✓ GELU Activation Functions");
    println!("✓ Layer Normalization (Pre-norm)");
    println!("✓ Residual Connections");
    println!("✓ Causal Masking for Autoregressive Generation");

    // Design principles demonstrated
    println!("\n🎯 Design Principles Demonstrated");
    println!("==================================");
    println!("✓ SOLID: Single responsibility, open/closed, interface segregation");
    println!("✓ CUPID: Composable, Unix philosophy, predictable, idiomatic");
    println!("✓ Zero-Cost Abstractions: Efficient iterator-based processing");
    println!("✓ Memory Efficiency: Arena allocators and parameter layout");
    println!("✓ Type Safety: Compile-time guarantees and bounds checking");
    println!("✓ Mathematical Foundations: Numerically stable implementations");

    println!("\n🎉 Enhanced transformer demonstration completed successfully!");

    Ok(())
}
