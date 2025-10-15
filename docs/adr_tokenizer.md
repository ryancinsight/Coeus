# Architecture Decision Record: Tokenizer Crate Implementation

## Context

Coeus requires NLP tokenization capabilities to support transformer-based models and natural language processing workloads. The tokenizer crate must provide PyTorch-compatible APIs while maintaining Coeus's design principles of memory safety, zero-cost abstractions, and clean architecture.

## Decision

Implement a comprehensive tokenizer crate (`coeus-tokenizer`) with the following architectural decisions:

### 1. Trait-Based Design
**Decision**: Use trait-based polymorphism for tokenizer algorithms
**Rationale**: Enables zero-cost abstraction and extensibility without runtime dispatch
**Trade-offs**:
- **Pro**: Compile-time polymorphism eliminates trait object overhead
- **Pro**: Easy to add new tokenizer algorithms
- **Con**: Monomorphization increases binary size
- **Mitigation**: Feature flags for optional tokenizer algorithms

### 2. Memory-Safe String Handling
**Decision**: Use `String` and `&str` with UTF-8 validation
**Rationale**: Ensures Unicode correctness and prevents encoding issues
**Trade-offs**:
- **Pro**: Automatic UTF-8 validation prevents encoding bugs
- **Pro**: Compatible with Rust's string ecosystem
- **Con**: Slight performance overhead vs byte-level processing
- **Mitigation**: Zero-copy string operations where possible

### 3. Vocabulary as HashMap
**Decision**: Use `HashMap<String, u32>` for token-to-ID mapping
**Rationale**: Provides O(1) lookup performance and serialization compatibility
**Trade-offs**:
- **Pro**: Fast lookups for large vocabularies
- **Pro**: Easy serialization to JSON
- **Con**: Memory overhead vs trie-based structures
- **Mitigation**: Lazy loading for large vocabularies

### 4. Iterator-Based Processing
**Decision**: Use iterators for tokenization pipelines
**Rationale**: Enables lazy evaluation and composability
**Trade-offs**:
- **Pro**: Memory-efficient processing of large texts
- **Pro**: Composable preprocessing/postprocessing pipelines
- **Con**: Potential iterator fusion complexity
- **Mitigation**: Extensive testing of iterator chains

### 5. Error Handling Strategy
**Decision**: Typed errors with `thiserror` for tokenizer-specific failures
**Rationale**: Provides rich error context while maintaining ergonomics
**Trade-offs**:
- **Pro**: Compile-time error exhaustiveness checking
- **Pro**: Rich error context for debugging
- **Con**: Error enum maintenance overhead
- **Mitigation**: Comprehensive error variants from day one

## Implementation Details

### Core Traits

```rust
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Encoding>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
}

pub trait PreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>>;
}

pub trait PostProcessor {
    fn post_process(&self, encoding: Encoding) -> Result<Encoding>;
}
```

### Vocabulary Management

```rust
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    special_tokens: HashMap<String, u32>,
}
```

### Encoding Structure

```rust
pub struct Encoding {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Vec<u32>,
}
```

## Algorithm Implementations

### BPE Tokenizer
**Decision**: Implement standard BPE with merge rules
**Rationale**: Most widely used algorithm (GPT, Llama, etc.)
**Implementation**: HashMap-based merge rule lookup for O(1) merges

### WordPiece Tokenizer
**Decision**: Implement Google's WordPiece algorithm
**Rationale**: Required for BERT compatibility
**Implementation**: Longest-match-first with continuation markers

### SentencePiece Tokenizer
**Decision**: Implement BPE-based SentencePiece
**Rationale**: Required for multilingual models (XLNet, ALBERT)
**Implementation**: Unicode-aware BPE with unigram fallback

## Performance Considerations

### Memory Layout
- **Decision**: Use contiguous vectors for token IDs
- **Rationale**: Optimal cache performance and SIMD compatibility
- **Trade-off**: Potential reallocation overhead
- **Mitigation**: Reserve capacity based on text length estimates

### String Interning
- **Decision**: No string interning for token storage
- **Rationale**: Simplicity and correctness over memory optimization
- **Trade-off**: Higher memory usage for large vocabularies
- **Mitigation**: Lazy vocabulary loading

### Parallel Processing
- **Decision**: Single-threaded tokenization with rayon for batch processing
- **Rationale**: Most tokenization is memory-bound, not CPU-bound
- **Trade-off**: No intra-text parallelism
- **Mitigation**: Batch processing for throughput

## Python Bindings

### PyO3 Integration
**Decision**: Use PyO3 classes with owned Rust data
**Rationale**: Memory safety across FFI boundary
**Trade-offs**:
- **Pro**: Automatic reference counting prevents leaks
- **Pro**: Thread-safe with Send + Sync bounds
- **Con**: Copy overhead for large vocabularies
- **Mitigation**: Shared reference counting for immutable data

### API Compatibility
**Decision**: Match HuggingFace transformers API exactly
**Rationale**: Drop-in replacement capability
**Trade-offs**:
- **Pro**: Zero migration effort for existing code
- **Pro**: Familiar API for Python developers
- **Con**: Some Rust idioms must be adapted
- **Mitigation**: Rust-specific methods alongside Python-compatible ones

## Serialization Format

### JSON Schema
```json
{
  "model": "BPE",
  "vocab": {"hello": 0, "world": 1},
  "merges": ["h e", "l l", "o r"],
  "special_tokens": {"[PAD]": 2, "[UNK]": 3}
}
```

**Decision**: JSON serialization with type-tagged models
**Rationale**: Human-readable, cross-language compatible
**Trade-offs**:
- **Pro**: Easy debugging and inspection
- **Pro**: Language-agnostic format
- **Con**: Larger file sizes vs binary formats
- **Mitigation**: Optional compression for large models

## Testing Strategy

### Unit Tests
- Algorithm correctness for each tokenizer type
- Edge cases: empty strings, Unicode, special characters
- Round-trip encode/decode consistency

### Property Tests
- Encoding determinism: same input → same output
- Vocabulary consistency: token ↔ ID bijection
- Unicode safety: valid UTF-8 in/out

### Integration Tests
- PyTorch API compatibility
- Batch processing correctness
- Memory safety under load

### Performance Benchmarks
- Tokenization throughput (>10k tokens/sec)
- Memory usage scaling
- Vocabulary loading time

## Future Extensions

### Planned Features
1. **GPU-Accelerated Tokenization**: wgpu-based batch processing
2. **Sparse Vocabularies**: Memory-efficient large vocabularies
3. **Custom Tokenizers**: User-defined tokenization rules
4. **Streaming Tokenization**: Incremental processing for large texts

### Compatibility Layers
1. **HuggingFace Hub**: Direct model loading from HF Hub
2. **ONNX Tokenizers**: ONNX-compatible serialization
3. **TensorFlow Hub**: TF Hub model compatibility

## Risks and Mitigations

### Performance Risk
**Risk**: Tokenization becomes bottleneck for large-scale NLP
**Mitigation**: SIMD acceleration, parallel batch processing, GPU offloading

### Unicode Risk
**Risk**: Incorrect handling of multilingual text
**Mitigation**: Comprehensive Unicode testing, normalization validation

### API Compatibility Risk
**Risk**: PyTorch API changes break compatibility
**Mitigation**: Extensive integration testing, semantic versioning

## Conclusion

The tokenizer crate architecture balances PyTorch compatibility with Rust's safety guarantees and performance characteristics. The trait-based design enables extensibility while maintaining zero-cost abstractions. Memory safety and Unicode correctness are prioritized over raw performance, aligning with Coeus's overall design philosophy.
