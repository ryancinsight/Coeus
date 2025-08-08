# RustLLM Core

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal dependency, high-performance Large Language Model (LLM) building library written in pure Rust. Built with elite programming practices and zero-cost abstractions.

## 🎯 Features

- **Zero Dependencies**: Core library has no external dependencies
- **Plugin Architecture**: Extensible design for custom implementations
- **Iterator-Based**: Leverages Rust's powerful iterator ecosystem
- **Zero-Copy Design**: Minimal memory allocations and copies
- **Type-Safe**: Compile-time guarantees with Rust's type system
- **Concurrent**: Thread-safe by design with Send + Sync traits
- **Elite Practices**: Follows SOLID, GRASP, CUPID, KISS, DRY, and YAGNI principles

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustllm-core = "0.1.0"
```

### Basic Usage

```rust
use rustllm_core::prelude::*;
use rustllm_tokenizer_bpe::BpeTokenizerPlugin;
use rustllm_model_transformer::TransformerModelPlugin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tokenizer via plugin
    let tokenizer = BpeTokenizerPlugin::default().create_tokenizer()?;

    // Tokenize text using iterator chains
    let tokens: Vec<_> = tokenizer
        .tokenize_str("Hello, world!")
        .filter(|token| token.as_str().map(|s| s.len() > 2).unwrap_or(false))
        .collect();

    // Build a model via plugin
    let builder = TransformerModelPlugin::default().create_builder()?;
    let model = builder.build(Transformer250MConfig::default())?;

    // Process tokens (example forwards expects ids; adapt as needed)
    let _output = model.forward(vec![1,2,3])?;

    Ok(())
}
```

## 🏗️ Architecture

The library is organized into three main layers:

### Foundation Layer
- **Iterator Extensions**: Custom combinators for token processing
- **Memory Management**: Smart pointers and arena allocators
- **Error Handling**: Type-safe error propagation
- **Type Safety**: Compile-time guarantees

### Core API Layer
- **Traits**: Abstract interfaces for extensibility
- **Plugin Manager**: Dynamic plugin loading and lifecycle
- **Utilities**: Common functionality and helpers

### Plugin Layer
- **Tokenizers**: BPE, WordPiece, SentencePiece
- **Models**: Transformer, RNN, custom architectures
- **Loaders**: GGUF, SafeTensors, custom formats

## 📖 Examples

### Stream Processing

```rust
use rustllm_core::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use rustllm_tokenizer_bpe::BpeTokenizerPlugin;

fn process_large_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let tokenizer = BpeTokenizerPlugin::default().create_tokenizer()?;

    // Process file line by line with zero copies
    let token_count = reader
        .lines()
        .filter_map(Result::ok)
        .flat_map(|line| tokenizer.tokenize_str(&line))
        .count();

    println!("Total tokens: {}", token_count);
    Ok(())
}
```

### Custom Plugin

```rust
use rustllm_core::prelude::*;

#[derive(Debug)]
struct CustomTokenizer;

impl Tokenizer for CustomTokenizer {
    type Token = StringToken;
    type Error = rustllm_core::foundation::error::Error;

    fn tokenize<'a>(&self, input: std::borrow::Cow<'a, str>) -> TokenIterator<'a, Self::Token> {
        Box::new(input.as_ref().split_whitespace().map(|s| StringToken::new(s.to_string())))
    }

    fn decode<I>(&self, tokens: I) -> Result<String>
    where
        I: IntoIterator<Item = Self::Token>,
    {
        Ok(tokens.into_iter().filter_map(|t| t.as_str()).collect::<Vec<_>>().join(" "))
    }
}

impl Plugin for CustomTokenizer {
    fn capabilities(&self) -> PluginCapabilities { PluginCapabilities::standard() }
}
```

### Advanced Iterator Chains

```rust
use rustllm_core::prelude::*;
use rustllm_tokenizer_bpe::BpeTokenizerPlugin;

fn sliding_window_tokens(
    text: &str,
    window_size: usize,
) -> impl Iterator<Item = Vec<StringToken>> + '_ {
    let tokenizer = BpeTokenizerPlugin::default()
        .create_tokenizer()
        .expect("Failed to create tokenizer");

    let tokens = tokenizer.tokenize_str(text).collect::<Vec<_>>();
    tokens
        .windows(window_size)
        .map(|window| window.to_vec())
        .collect::<Vec<_>>()
}
```

## 🔧 Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/rustllm-core.git
cd rustllm-core

# Build the project
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build documentation
cargo doc --open
```

## 🧪 Testing

The library includes comprehensive tests:

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run specific test
cargo test test_tokenizer_basic

# Run integration tests
cargo test --test '*'
```

## 📊 Benchmarks

Performance benchmarks are included:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench bench_tokenizer

# Generate flamegraph
cargo flamegraph --bench bench_tokenizer
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Install Rust (latest stable)
2. Install development tools:
   ```bash
   cargo install cargo-tarpaulin cargo-flamegraph cargo-criterion
   ```
3. Fork and clone the repository
4. Create a feature branch
5. Make your changes with tests
6. Submit a pull request

### Code Style

- Follow Rust API Guidelines
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write comprehensive tests
- Document public APIs

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Rust Community for excellent tooling
- Contributors and maintainers
- LLM research community

## 📚 Resources

- [API Documentation](https://docs.rs/rustllm-core)
- [Examples](./examples)
- [Benchmarks](./benches)
- [Blog Posts](https://blog.rustllm.dev)

## 🗺️ Roadmap

- [ ] v0.1.0 - Core architecture and basic plugins
- [ ] v0.2.0 - Advanced tokenizers and model formats
- [ ] v0.3.0 - Performance optimizations
- [ ] v0.4.0 - Distributed processing support
- [ ] v1.0.0 - Stable API and production ready

---

Built with ❤️ in Rust 
