//! Configuration system usage example.

use rustllm_core::prelude::*;

fn main() -> Result<()> {
    println!("RustLLM Core Configuration Example");
    println!("==================================\n");

    // Example 1: Basic configuration values
    println!("Example 1: Basic Configuration Values");
    println!("------------------------------------");

    let config = ConfigBuilder::map()
        .with("name", "MyModel")
        .with("version", 1)
        .with("batch_size", 32)
        .with("learning_rate", 0.001)
        .with("enabled", true)
        .with("layers", vec![128, 256, 512, 256, 128])
        .build();

    if let ConfigValue::Map(map) = &config {
        println!("Configuration created with {} keys", map.len());

        // Access values
        if let Some(name) = map.get("name").and_then(|v| v.as_string()) {
            println!("Model name: {}", name);
        }

        if let Some(batch_size) = map.get("batch_size").and_then(|v| v.as_integer()) {
            println!("Batch size: {}", batch_size);
        }

        if let Some(layers) = map.get("layers").and_then(|v| v.as_list()) {
            println!("Network layers: {} layers", layers.len());
        }
    }

    // Example 2: Configuration store
    println!("\n\nExample 2: Configuration Store");
    println!("------------------------------");

    let store = ConfigStore::new();

    // Set various configuration values
    store.set("model.name", "GPT-Mini")?;
    store.set("model.layers", 12)?;
    store.set("model.hidden_size", 768)?;
    store.set("training.batch_size", 16)?;
    store.set("training.epochs", 10)?;
    store.set("training.learning_rate", 0.0001)?;

    // List all keys
    let keys = store.keys()?;
    println!("Configuration keys: {:?}", keys);

    // Get values
    if let Some(value) = store.get("model.name")? {
        if let Some(name) = value.as_string() {
            println!("Model name: {}", name);
        }
    }

    // Get with default
    let default_dropout = store.get_or_default("model.dropout", 0.1)?;
    println!("Dropout rate: {:?}", default_dropout.as_float());

    // Example 3: Plugin configuration
    println!("\n\nExample 3: Plugin Configuration");
    println!("-------------------------------");

    let plugin_manager = PluginConfigManager::new();
    let tokenizer_plugin = PluginName::from("bpe_tokenizer");

    // Configure a plugin
    let tokenizer_config = ConfigBuilder::map()
        .with("vocab_size", 32000)
        .with("min_frequency", 2)
        .with("special_tokens", vec!["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        .build();

    plugin_manager.set_plugin_config(&tokenizer_plugin, tokenizer_config)?;

    // Set individual plugin keys
    plugin_manager.set_plugin_key(&tokenizer_plugin, "max_length", 512)?;
    plugin_manager.set_plugin_key(&tokenizer_plugin, "lowercase", false)?;

    // Retrieve plugin configuration
    if let Some(config) = plugin_manager.get_plugin_config(&tokenizer_plugin)? {
        println!("Plugin configuration retrieved");

        if let ConfigValue::Map(map) = config {
            if let Some(vocab_size) = map.get("vocab_size").and_then(|v| v.as_integer()) {
                println!("Vocabulary size: {}", vocab_size);
            }
        }
    }

    // Get specific plugin key
    if let Some(max_length) = plugin_manager.get_plugin_key(&tokenizer_plugin, "max_length")? {
        println!("Max sequence length: {:?}", max_length.as_integer());
    }

    // Example 4: Nested configuration
    println!("\n\nExample 4: Nested Configuration");
    println!("-------------------------------");

    let model_config = ConfigBuilder::map()
        .with(
            "architecture",
            ConfigBuilder::map()
                .with("type", "transformer")
                .with("layers", 6)
                .with(
                    "attention",
                    ConfigBuilder::map()
                        .with("heads", 8)
                        .with("dim_per_head", 64)
                        .with("dropout", 0.1)
                        .build(),
                )
                .build(),
        )
        .with(
            "training",
            ConfigBuilder::map()
                .with("optimizer", "adam")
                .with("scheduler", "cosine")
                .with("warmup_steps", 1000)
                .build(),
        )
        .build();

    if let ConfigValue::Map(root) = model_config {
        if let Some(ConfigValue::Map(arch)) = root.get("architecture") {
            println!(
                "Architecture type: {:?}",
                arch.get("type").and_then(|v| v.as_string())
            );

            if let Some(ConfigValue::Map(attention)) = arch.get("attention") {
                println!(
                    "Attention heads: {:?}",
                    attention.get("heads").and_then(|v| v.as_integer())
                );
            }
        }
    }

    // Example 5: Configuration merging
    println!("\n\nExample 5: Configuration Merging");
    println!("--------------------------------");

    let base_store = ConfigStore::new();
    base_store.set("model.type", "base")?;
    base_store.set("model.size", "small")?;

    let override_store = ConfigStore::new();
    override_store.set("model.size", "large")?;
    override_store.set("model.layers", 24)?;

    // Merge configurations
    base_store.merge(&override_store)?;

    println!("After merging:");
    let merged_keys = base_store.keys()?;
    for key in &merged_keys {
        if let Some(value) = base_store.get(key)? {
            println!("  {}: {:?}", key, value);
        }
    }

    println!("\n\nAll configuration examples completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_example() {
        assert!(main().is_ok());
    }
}
