//! WordPiece Tokenizer Example
//!
//! This example demonstrates the production-ready WordPiece tokenizer implementation
//! showcasing elite programming practices and design principles.

use rustllm_core::{
    core::{
        plugin::TokenizerPlugin,
        tokenizer::{Token, Tokenizer},
        traits::{foundation::Named, identity::Versioned},
    },
    foundation::error::Result,
};
use rustllm_tokenizer_wordpiece::WordPieceTokenizerPlugin;
use std::borrow::Cow;

fn main() -> Result<()> {
    println!("🔤 WordPiece Tokenizer Example");
    println!("==============================");
    
    // Create the WordPiece tokenizer plugin
    let plugin = WordPieceTokenizerPlugin::default();
    println!("✓ Created WordPiece tokenizer plugin");
    println!("   • Name: {}", plugin.name());
    println!("   • Version: {}", plugin.version());
    
    // Create tokenizer instance
    let tokenizer = plugin.create_tokenizer()?;
    println!("✓ Created tokenizer instance");
    println!("   • Name: {}", tokenizer.name());
    println!("   • Vocabulary Size: {}", tokenizer.vocab_size());
    
    // Test texts with different complexity levels
    let test_texts = vec![
        ("Simple", "hello world"),
        ("Complex", "The quick brown fox jumps over the lazy dog"),
        ("Technical", "WordPiece tokenization algorithm implementation"),
        ("Unicode", "café naïve résumé"),
        ("Mixed", "Hello, 世界! How are you today?"),
    ];
    
    println!("\n🧪 Tokenization Examples");
    println!("=========================");
    
    for (category, text) in test_texts {
        println!("\n📝 {} Text: \"{}\"", category, text);
        
        // Tokenize the text
        let start_time = std::time::Instant::now();
        let tokens: Vec<_> = tokenizer.tokenize(Cow::Borrowed(text)).collect();
        let tokenize_time = start_time.elapsed();
        
        println!("   🔤 Tokens ({}):", tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            if let Some(token_str) = token.as_str() {
                println!("      {}. \"{}\" (ID: {})", i + 1, token_str, token.id());
            }
        }
        
        // Decode back to text
        let start_time = std::time::Instant::now();
        let decoded = tokenizer.decode(tokens.clone())?;
        let decode_time = start_time.elapsed();
        
        println!("   📄 Decoded: \"{}\"", decoded);
        println!("   ⏱️  Timing:");
        println!("      • Tokenization: {:.2}μs", tokenize_time.as_micros());
        println!("      • Decoding: {:.2}μs", decode_time.as_micros());
        
        // Calculate compression ratio
        let original_chars = text.chars().count();
        let token_count = tokens.len();
        let compression_ratio = token_count as f64 / original_chars as f64;
        println!("   📊 Stats:");
        println!("      • Characters: {}", original_chars);
        println!("      • Tokens: {}", token_count);
        println!("      • Compression Ratio: {:.2}", compression_ratio);
    }
    
    // Demonstrate trie-based vocabulary lookup performance
    println!("\n⚡ Performance Demonstration");
    println!("============================");
    
    let large_text = "The WordPiece tokenization algorithm is a subword tokenization technique that was introduced by Google for their BERT model. It works by starting with a vocabulary of individual characters and iteratively merging the most frequent pairs of adjacent symbols to create new subword units. This approach allows the model to handle out-of-vocabulary words by breaking them down into smaller, known subword pieces.";
    
    println!("📝 Large Text ({} characters):", large_text.len());
    println!("   \"{}...\"", &large_text[..100]);
    
    // Benchmark tokenization
    let iterations = 100usize;
    let start_time = std::time::Instant::now();

    for _ in 0..iterations {
        let _tokens: Vec<_> = tokenizer.tokenize(Cow::Borrowed(large_text)).collect();
    }

    let total_time = start_time.elapsed();
    let avg_time = total_time / iterations as u32;
    let chars_per_second = (large_text.len() * iterations) as f64 / total_time.as_secs_f64();
    
    println!("⏱️  Benchmark Results ({} iterations):", iterations);
    println!("   • Total Time: {:.2}ms", total_time.as_millis());
    println!("   • Average Time: {:.2}μs", avg_time.as_micros());
    println!("   • Throughput: {:.0} chars/second", chars_per_second);
    
    // Demonstrate algorithm features
    println!("\n🏗️  Algorithm Features");
    println!("=======================");
    println!("✓ Trie-based vocabulary lookup: O(k) where k is token length");
    println!("✓ Unicode normalization: NFD normalization for consistency");
    println!("✓ Greedy longest-match: Optimal subword segmentation");
    println!("✓ Zero-copy processing: Iterator-based with minimal allocations");
    println!("✓ Special token handling: [CLS], [SEP], [UNK], [PAD], [MASK]");
    println!("✓ Subword continuation: ## prefix for word pieces");
    
    // Design principles demonstrated
    println!("\n🎯 Design Principles Demonstrated");
    println!("==================================");
    println!("✓ SOLID: Single responsibility, interface segregation");
    println!("✓ CUPID: Composable, Unix philosophy, predictable, idiomatic");
    println!("✓ Zero-Cost Abstractions: Efficient iterator-based processing");
    println!("✓ Memory Efficiency: Trie-based vocabulary and COW strings");
    println!("✓ Type Safety: Compile-time guarantees and bounds checking");
    println!("✓ Mathematical Foundations: Proper Unicode handling and normalization");
    
    println!("\n🎉 WordPiece tokenizer demonstration completed successfully!");
    
    Ok(())
}
