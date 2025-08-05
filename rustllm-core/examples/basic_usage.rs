//! Basic usage example for RustLLM Core.

use rustllm_core::prelude::*;
use rustllm_core::plugins::manager::PluginManager;

// Import the basic plugins
use rustllm_tokenizer_basic::BasicTokenizerPlugin;
use rustllm_model_basic::BasicModelPlugin;
use rustllm_loader_basic::BasicLoaderPlugin;

fn main() -> Result<()> {
    println!("RustLLM Core Basic Usage Example");
    println!("================================\n");
    
    // Create a plugin manager
    let manager = PluginManager::new();
    
    // Register plugins
    println!("Registering plugins...");
    manager.register::<BasicTokenizerPlugin>()?;
    manager.register::<BasicModelPlugin>()?;
    manager.register::<BasicLoaderPlugin>()?;
    
    // List available plugins
    let available = manager.list_available()?;
    println!("Available plugins: {:?}\n", available);
    
    // Example 1: Using the tokenizer directly
    println!("Example 1: Direct Tokenizer Usage");
    println!("---------------------------------");
    
    use rustllm_tokenizer_basic::BasicTokenizer;
    let tokenizer = BasicTokenizer::new();
    
    let text = "Hello, world! This is RustLLM Core.";
    println!("Input text: {}", text);
    
    let tokens: Vec<_> = tokenizer.tokenize_str(text).collect();
    println!("Tokens: {:?}", tokens);
    
    let decoded = tokenizer.decode(tokens)?;
    println!("Decoded: {}\n", decoded);
    
    // Example 2: Using iterators with tokenizer
    println!("Example 2: Iterator Combinators");
    println!("-------------------------------");
    
    let long_tokens: Vec<_> = tokenizer
        .tokenize_str(text)
        .filter(|token| token.as_str().map(|s| s.len() > 4).unwrap_or(false))
        .collect();
    
    println!("Tokens longer than 4 chars: {:?}\n", long_tokens);
    
    // Example 3: Using windows iterator
    println!("Example 3: Sliding Windows");
    println!("--------------------------");
    
    let window_tokens: Vec<_> = tokenizer
        .tokenize_str("one two three four five")
        .collect::<Vec<_>>()
        .windows(3)
        .map(|w| w.to_vec())
        .collect();
    
    println!("3-token windows: {:?}\n", window_tokens);
    
    // Example 4: Using the model builder
    println!("Example 4: Model Building");
    println!("------------------------");
    
    use rustllm_model_basic::BasicModelBuilder;
    use rustllm_core::core::model::{ModelBuilder, BasicModelConfig, ForwardModel};
    
    let builder = BasicModelBuilder::new();
    let config = BasicModelConfig::default();
    
    println!("Model config: {:?}", config);
    let model = builder.build(config)?;
    
    println!("Model parameters: {}", model.num_parameters());
    
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output = model.forward(input.clone())?;
    println!("Model forward pass: {:?} -> {:?}\n", input, output);
    
    // Example 5: Memory management
    println!("Example 5: Zero-Copy Memory Management");
    println!("-------------------------------------");
    
    use rustllm_core::foundation::memory::{Arena, CowStr, ZeroCopyStringBuilder};
    
    // Arena allocator
    let mut arena = Arena::new(1024);
    let x = arena.alloc(42);
    println!("Arena allocated value: {}", x);
    
    let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);
    println!("Arena allocated slice: {:?}", slice);
    
    // Copy-on-write strings
    let mut cow = CowStr::borrowed("hello");
    println!("Borrowed COW: {:?}", cow.as_str());
    
    let mutable = cow.to_mut();
    mutable.push_str(" world");
    println!("Modified COW: {:?}", cow.as_str());
    
    // Zero-copy string builder
    let mut builder = ZeroCopyStringBuilder::new();
    builder.push_borrowed("Hello");
    builder.push_borrowed(" ");
    builder.push_owned(String::from("RustLLM"));
    let result = builder.build();
    println!("Built string: {}\n", result);
    
    println!("All examples completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_runs() {
        assert!(main().is_ok());
    }
}