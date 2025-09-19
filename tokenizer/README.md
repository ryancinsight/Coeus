# Coeus Tokenizer

A high-performance, memory-safe tokenizer implementation for the Coeus ML framework, providing tiktoken-compatible functionality with native Rust performance.

## Features

- **Byte-Pair Encoding (BPE)**: Core BPE algorithm with vocabulary management
- **Popular Models**: GPT-2, GPT-3/4, CLIP, BERT tokenizer variants (via feature flags)
- **Special Tokens**: Comprehensive special token handling ([CLS], [SEP], [MASK], <|endoftext|>)
- **Batch Processing**: Efficient batch encoding/decoding operations
- **Memory Safety**: Zero unsafe code with Rust's ownership guarantees
- **Performance**: Competitive with tiktoken, optimized for ML workflows

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
coeus-tokenizer = { path = "../tokenizer" }  # For local development

# Or from crates.io (when published)
# coeus-tokenizer = "0.1"
```

## Feature Flags

Enable specific tokenizer models:

```toml
[dependencies.coeus-tokenizer]
version = "0.1"
features = ["gpt2", "gpt3", "clip", "bert"]

# Or enable all models
features = ["all_models"]
```

## Quick Start

```rust
use coeus_tokenizer::Encoding;

// Create a GPT-2 tokenizer (when gpt2 feature is enabled)
// let tokenizer = Encoding::new("gpt2")?;

// For now, the core tokenizer infrastructure is ready
// Model implementations will be added in future sprints
```

## Architecture

The tokenizer crate follows a modular architecture:

- **`Tokenizer` trait**: Abstract interface for all tokenizer implementations
- **`Vocabulary`**: Efficient token-to-ID mapping with special token support
- **`BpeTokenizer`**: Core Byte-Pair Encoding implementation
- **`Encoding`**: High-level interface providing tiktoken-compatible API
- **Model-specific implementations**: GPT-2, GPT-3/4, CLIP, BERT variants

## Core Components

### Vocabulary Management

```rust
use coeus_tokenizer::Vocabulary;

let mut vocab = Vocabulary::new();
vocab.add_token("hello".to_string());
vocab.add_special_token("[CLS]".to_string());

assert_eq!(vocab.get_token_id("hello"), Some(0));
assert!(vocab.is_special_token("[CLS]"));
```

### BPE Algorithm

```rust
use coeus_tokenizer::BpeTokenizer;

// Create and train a BPE tokenizer
let mut tokenizer = BpeTokenizer::new("custom".to_string());

// Training would typically use a large corpus
// tokenizer.train(&corpus, 50000, Some(1000))?;
```

### Error Handling

The crate provides comprehensive error handling:

```rust
use coeus_tokenizer::TokenizerError;

match result {
    Ok(value) => println!("Success: {:?}", value),
    Err(TokenizerError::InvalidInput { message }) => {
        eprintln!("Invalid input: {}", message);
        // Handle recoverable error
    }
    Err(TokenizerError::ModelError { message }) => {
        eprintln!("Model error: {}", message);
        // Handle fatal error
    }
    _ => eprintln!("Other error occurred"),
}
```

## Development Status

### ✅ Completed (Sprint 1-3)
- **Sprint 1**: Requirements documentation and architecture validation
- **Sprint 2**: Core tokenizer traits and error handling
- **Sprint 3**: BPE algorithm implementation with vocabulary management

### 🔄 In Progress (Sprint 4-7)
- **Sprint 4**: Popular model implementations (GPT-2, GPT-3/4, CLIP, BERT)
- **Sprint 5**: Advanced features (batch processing, special tokens)
- **Sprint 6**: Integration with tensor operations
- **Sprint 7**: Performance optimization and production validation

## API Compatibility

The tokenizer provides tiktoken-compatible APIs:

- `Encoding::new(model_name)` - Create tokenizer for specific model
- `encode(text)` / `decode(tokens)` - Basic encoding/decoding
- `encode_with_special_tokens()` / `decode_with_special_tokens()` - Special token handling
- `encode_batch()` / `decode_batch()` - Batch operations

## Testing

Run the test suite:

```bash
cargo test --package coeus-tokenizer
```

Run with specific features:

```bash
cargo test --package coeus-tokenizer --features gpt2
```

## Performance

The tokenizer is designed for high performance:

- Zero-copy operations where possible
- Efficient vocabulary lookups using `fxhash`
- Parallel processing support via `rayon`
- Memory-efficient BPE implementation

## Safety

- **Zero unsafe code**: All operations use safe Rust
- **Memory safety**: Rust's ownership system prevents memory errors
- **Thread safety**: All tokenizers implement `Send + Sync`
- **Error handling**: Comprehensive error types with recovery guidance

## Contributing

The tokenizer crate follows the Coeus project's development methodology:

1. **Sprint-based development**: Iterative development with clear milestones
2. **Comprehensive testing**: 100% test coverage with edge cases
3. **Documentation**: Complete API documentation with examples
4. **Performance validation**: Benchmarks against tiktoken reference

## License

Licensed under MIT OR Apache-2.0, matching the Coeus project.
