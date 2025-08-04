//! Integration tests for RustLLM Core.

use rustllm_core::prelude::*;
use rustllm_core::plugins::manager::PluginManager;
use rustllm_core::plugins::registry::PluginRegistryBuilder;
use rustllm_core::core::plugin::{TokenizerPlugin, ModelBuilderPlugin, PluginState};
use rustllm_core::core::tokenizer::VocabularyTokenizer;
use rustllm_core::core::model::{ModelBuilder, BasicModelConfig};
use rustllm_core::foundation::iterator::IteratorExt;

// Import plugins
use rustllm_tokenizer_basic::BasicTokenizerPlugin;
use rustllm_tokenizer_bpe::BpeTokenizerPlugin;
use rustllm_model_basic::BasicModelPlugin;
use rustllm_loader_basic::BasicLoaderPlugin;

#[test]
fn test_plugin_registration_and_loading() {
    let manager = PluginManager::new();
    
    // Register plugins
    assert!(manager.register::<BasicTokenizerPlugin>().is_ok());
    assert!(manager.register::<BpeTokenizerPlugin>().is_ok());
    assert!(manager.register::<BasicModelPlugin>().is_ok());
    assert!(manager.register::<BasicLoaderPlugin>().is_ok());
    
    // Check available plugins
    let available = manager.list_available().unwrap();
    assert!(available.contains(&"basic_tokenizer".to_string()));
    assert!(available.contains(&"bpe_tokenizer".to_string()));
    assert!(available.contains(&"basic_model".to_string()));
    assert!(available.contains(&"basic_loader".to_string()));
}

#[test]
fn test_tokenizer_plugin_usage() -> Result<()> {
    // Create tokenizer through plugin directly
    let plugin = BasicTokenizerPlugin::default();
    let tokenizer = plugin.create_tokenizer()?;
    
    // Test tokenization
    let text = "Hello, RustLLM!";
    let tokens: Vec<_> = tokenizer.tokenize_str(text).collect();
    assert!(!tokens.is_empty());
    
    // Test decode
    let decoded = tokenizer.decode(tokens)?;
    assert_eq!(decoded.trim(), text);
    
    Ok(())
}

#[test]
fn test_bpe_tokenizer_plugin() {
    let plugin = BpeTokenizerPlugin::default();
    let mut tokenizer = plugin.create_tokenizer().unwrap();
    
    // Test basic tokenization
    let text = "The quick brown fox";
    let tokens: Vec<_> = tokenizer.tokenize_str(text).collect();
    assert!(!tokens.is_empty());
    
    // Test vocabulary operations
    assert!(tokenizer.vocab_size() > 0);
    
    // Test special tokens
    assert!(tokenizer.contains_token("<PAD>"));
    assert!(tokenizer.contains_token("<UNK>"));
    
    // Add custom token
    let _token_id = tokenizer.add_token("custom_token").unwrap();
    assert!(tokenizer.contains_token("custom_token"));
    
    let vocab_size = tokenizer.vocab_size();
    assert!(vocab_size > 256); // Should have more than just byte tokens
}

#[test]
fn test_model_builder_plugin() {
    // Create model through plugin directly
    let plugin = BasicModelPlugin::default();
    let builder = plugin.create_builder().unwrap();
    
    // Build model
    let config = BasicModelConfig::default();
    let model = builder.build(config).unwrap();
    
    // Test model - BasicModel calculates parameters based on config
    assert!(model.num_parameters() > 0); // Model has parameters
    let input = vec![1.0, 2.0, 3.0];
    let output = model.forward(input.clone()).unwrap();
    assert_eq!(output, input); // Identity forward pass
}

#[test]
fn test_plugin_lifecycle() -> Result<()> {
    let manager = PluginManager::new();
    
    // Register plugin
    manager.register::<BasicTokenizerPlugin>()?;
    
    // Load plugin
    let plugin = manager.load_plugin("basic_tokenizer")?;
    assert_eq!(plugin.name(), "basic_tokenizer");
    
    // Get plugin info
    let info = manager.info("basic_tokenizer")?;
    assert_eq!(info.name.as_str(), "basic_tokenizer");
    assert_eq!(info.state, PluginState::Ready);
    
    // List loaded plugins
    let loaded = manager.list_loaded()?;
    assert!(loaded.contains(&"basic_tokenizer".to_string()));
    
    // Unload plugin
    manager.unload("basic_tokenizer")?;
    
    // Should not be in loaded list
    let loaded = manager.list_loaded()?;
    assert!(!loaded.contains(&"basic_tokenizer".to_string()));
    
    Ok(())
}

#[test]
fn test_iterator_extensions_with_tokenizer() {
    let tokenizer = rustllm_tokenizer_basic::BasicTokenizer::new();
    let text = "one two three four five six seven eight nine ten";
    
    // Test windows
    let tokens: Vec<_> = tokenizer.tokenize_str(text).collect();
    let windows: Vec<_> = tokens.iter()
        .cloned()
        .windows(3)
        .take(3)
        .collect();
    assert_eq!(windows.len(), 3);
    
    // Test chunks
    let chunks: Vec<_> = tokens.iter()
        .cloned()
        .chunks(2)
        .take(3)
        .collect();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].len(), 2);
    
    // Test stride
    let strided: Vec<_> = tokens.iter()
        .cloned()
        .stride(2)
        .collect();
    assert_eq!(strided.len(), 5); // Every other token
}

#[test]
fn test_memory_management() {
    use rustllm_core::foundation::memory::{Arena, Pool, CowStr, StrBuilder};
    
    // Test arena allocator
    let arena = Arena::new(1024);
    let x = arena.alloc(42);
    assert_eq!(*x, 42);
    
    let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);
    assert_eq!(slice, &[1, 2, 3, 4, 5]);
    
    // Test pool
    let pool: Pool<Vec<u8>> = Pool::new(10);
    let buffer = pool.take_or_else(|| Vec::with_capacity(100));
    assert!(buffer.capacity() >= 100);
    
    // Test COW string
    let mut cow = CowStr::borrowed("hello");
    assert!(matches!(cow, CowStr::Borrowed(_)));
    
    let mutable = cow.to_mut();
    mutable.push_str(" world");
    assert!(matches!(cow, CowStr::Owned(_)));
    
    // Test string builder
    let mut builder = StrBuilder::new();
    builder.push_borrowed("Hello");
    builder.push_borrowed(" ");
    builder.push_owned(String::from("World"));
    let result = builder.build();
    assert_eq!(result, "Hello World");
}

#[test]
fn test_error_handling() {
    use rustllm_core::foundation::error::{Error, ErrorExt, internal_error};
    
    // Test error creation
    let _base_error = internal_error("base error");
    
    // Test error with context
    let error = internal_error("test");
    let with_context = error.with_context("operation failed");
    
    // Check error message
    let error_string = format!("{}", with_context);
    assert!(error_string.contains("operation failed"));
}

#[test]
fn test_type_safety() {
    use rustllm_core::foundation::types::{Version, Shape};
    
    // Test version compatibility
    let v1 = Version::new(1, 0, 0);
    let v2 = Version::new(1, 1, 0);
    let v3 = Version::new(2, 0, 0);
    
    // Same major version, higher minor is compatible
    assert!(v2.is_compatible_with(&v1)); // 1.1.0 is compatible with 1.0.0
    assert!(!v1.is_compatible_with(&v2)); // 1.0.0 is NOT compatible with 1.1.0
    assert!(!v1.is_compatible_with(&v3)); // Different major version
    
    // Test shape operations
    let shape1 = Shape::new(vec![2, 3, 4]);
    let shape2 = Shape::new(vec![2, 3, 4]);
    let shape3 = Shape::new(vec![3, 4]);
    
    assert_eq!(shape1.dims().len(), 3);
    assert_eq!(shape1.numel(), 24);
    assert!(shape1.is_compatible_with(&shape2));
    assert!(!shape1.is_compatible_with(&shape3));
}

#[test]
fn test_concurrent_plugin_access() {
    use std::sync::Arc;
    use std::thread;
    
    let manager = Arc::new(PluginManager::new());
    
    // Register plugins
    manager.register::<BasicTokenizerPlugin>().unwrap();
    manager.register::<BpeTokenizerPlugin>().unwrap();
    
    // Spawn multiple threads to access plugins
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let manager_clone = Arc::clone(&manager);
            thread::spawn(move || {
                // Each thread checks available plugins
                let available = manager_clone.list_available().unwrap();
                assert!(!available.is_empty());
                
                // Create plugin instances directly
                if i % 2 == 0 {
                    let plugin = BasicTokenizerPlugin::default();
                    let _tokenizer = plugin.create_tokenizer().unwrap();
                } else {
                    let plugin = BpeTokenizerPlugin::default();
                    let _tokenizer = plugin.create_tokenizer().unwrap();
                }
            })
        })
        .collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Check available plugins
    let available = manager.list_available().unwrap();
    assert!(available.contains(&"basic_tokenizer".to_string()));
    assert!(available.contains(&"bpe_tokenizer".to_string()));
}

#[test]
fn test_plugin_registry_builder() {
    use rustllm_core::foundation::types::PluginName;
    
    let registry = PluginRegistryBuilder::new()
        .with_plugin::<BasicTokenizerPlugin>()
        .with_plugin::<BpeTokenizerPlugin>()
        .with_plugin::<BasicModelPlugin>()
        .build();
    
    // Check all plugins are registered
    let plugins = registry.list();
    assert_eq!(plugins.len(), 3);
    assert!(plugins.contains(&"basic_tokenizer".to_string()));
    assert!(plugins.contains(&"bpe_tokenizer".to_string()));
    assert!(plugins.contains(&"basic_model".to_string()));
    
    // Create instances
    assert!(registry.create(&PluginName::from("basic_tokenizer")).is_ok());
    assert!(registry.create(&PluginName::from("bpe_tokenizer")).is_ok());
    assert!(registry.create(&PluginName::from("basic_model")).is_ok());
    assert!(registry.create(&PluginName::from("nonexistent")).is_err());
}

#[test]
fn test_concurrent_plugin_loading() {
    use std::sync::Arc;
    use std::thread;
    
    let manager = Arc::new(PluginManager::new());
    
    // Register plugins
    manager.register::<BasicTokenizerPlugin>().unwrap();
    manager.register::<BasicModelPlugin>().unwrap();
    
    // Spawn multiple threads that try to load the same plugin simultaneously
    let handles: Vec<_> = (0..10)
        .map(|_i| {
            let manager_clone = Arc::clone(&manager);
            thread::spawn(move || {
                // All threads try to load the same plugins
                let tokenizer_plugin = manager_clone.load_plugin("basic_tokenizer").unwrap();
                assert_eq!(tokenizer_plugin.name(), "basic_tokenizer");
                
                let model_plugin = manager_clone.load_plugin("basic_model").unwrap();
                assert_eq!(model_plugin.name(), "basic_model");
                
                // Verify plugins are properly initialized
                assert_eq!(tokenizer_plugin.version().major, 0);
                assert_eq!(model_plugin.version().major, 0);
                
                // Each thread also lists loaded plugins
                let loaded = manager_clone.list_loaded().unwrap();
                assert!(loaded.contains(&"basic_tokenizer".to_string()));
                assert!(loaded.contains(&"basic_model".to_string()));
            })
        })
        .collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let loaded = manager.list_loaded().unwrap();
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains(&"basic_tokenizer".to_string()));
    assert!(loaded.contains(&"basic_model".to_string()));
}