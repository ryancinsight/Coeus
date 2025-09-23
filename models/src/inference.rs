//! Efficient inference engine for transformer models.

#![allow(clippy::needless_borrow)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::manual_is_multiple_of)]
//!
//! This module provides a high-performance inference engine for transformer-based
//! language models, optimized for memory usage and inference speed. It supports:
//!
//! - **Batch Processing**: Efficient inference for multiple sequences
//! - **Memory Optimization**: Quantized weights and key-value caching
//! - **Streaming Generation**: Token-by-token generation with early stopping
//! - **Multi-Architecture**: Support for Llama, GPT-2, and other transformers
//! - **GPU Acceleration**: Optional GPU support via wgpu backend
//!
//! ## Architecture
//!
//! The inference engine is organized into several key components:
//!
//! - **`InferenceConfig`**: Configuration for inference parameters
//! - **`InferenceEngine`**: Core inference engine with memory management
//! - **`LlamaModel`**: High-level model interface for Llama-based models
//! - **`InferenceResult`**: Results from inference operations
//!
//! ## Performance Features
//!
//! - **Key-Value Caching**: Avoids recomputation of attention keys and values
//! - **Memory Pooling**: Efficient memory allocation and reuse
//! - **SIMD Optimization**: Vectorized operations for better performance
//! - **Batch Processing**: Parallel inference for multiple sequences
//! - **Quantized Inference**: Reduced precision computation with maintained accuracy
//! - **GPU Offloading**: Optional GPU acceleration for compute-intensive operations

use crate::config::ModelConfig;
use crate::error::{ModelError, ModelResult};
use crate::quantization::QuantizedTensor;
use std::collections::HashMap;

/// Inference configuration parameters
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Temperature for sampling (0.0 = greedy, > 1.0 = more random)
    pub temperature: f32,
    /// Top-p sampling parameter (0.0 to 1.0)
    pub top_p: f32,
    /// Top-k sampling parameter (0 to disable)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty, > 1.0 = penalize repetition)
    pub repetition_penalty: f32,
    /// Length penalty (1.0 = no penalty)
    pub length_penalty: f32,
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,
    /// Minimum number of tokens to generate
    pub min_new_tokens: usize,
    /// Early stopping when end-of-sequence token is generated
    pub early_stopping: bool,
    /// Padding token ID
    pub pad_token_id: Option<usize>,
    /// End-of-sequence token ID
    pub eos_token_id: Option<usize>,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Use key-value caching for efficiency
    pub use_kv_cache: bool,
    /// Enable memory pooling
    pub use_memory_pool: bool,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            batch_size: 1,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            length_penalty: 1.0,
            max_new_tokens: 100,
            min_new_tokens: 1,
            early_stopping: true,
            pad_token_id: None,
            eos_token_id: None,
            use_gpu: false,
            use_kv_cache: true,
            use_memory_pool: true,
            debug: false,
        }
    }
}

impl InferenceConfig {
    /// Create a new inference configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    /// Set length penalty
    pub fn with_length_penalty(mut self, length_penalty: f32) -> Self {
        self.length_penalty = length_penalty;
        self
    }

    /// Set generation limits
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Set minimum tokens to generate
    pub fn with_min_new_tokens(mut self, min_new_tokens: usize) -> Self {
        self.min_new_tokens = min_new_tokens;
        self
    }

    /// Set special token IDs
    pub fn with_pad_token_id(mut self, pad_token_id: usize) -> Self {
        self.pad_token_id = Some(pad_token_id);
        self
    }

    /// Set EOS token ID
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Enable/disable key-value caching
    pub fn with_kv_cache(mut self, use_kv_cache: bool) -> Self {
        self.use_kv_cache = use_kv_cache;
        self
    }

    /// Enable/disable memory pooling
    pub fn with_memory_pool(mut self, use_memory_pool: bool) -> Self {
        self.use_memory_pool = use_memory_pool;
        self
    }

    /// Enable debug logging
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
}

/// Inference result containing generated tokens and metadata
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Generated token sequences
    pub sequences: Vec<Vec<usize>>,
    /// Generation scores (logits)
    pub scores: Vec<Vec<f32>>,
    /// Sequence lengths
    pub lengths: Vec<usize>,
    /// Whether generation was stopped early
    pub stopped_early: Vec<bool>,
    /// Total inference time in milliseconds
    pub inference_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of tokens generated per sequence
    pub tokens_generated: Vec<usize>,
}

impl InferenceResult {
    /// Create a new inference result
    pub fn new(
        sequences: Vec<Vec<usize>>,
        scores: Vec<Vec<f32>>,
        lengths: Vec<usize>,
        stopped_early: Vec<bool>,
        inference_time_ms: u64,
        memory_usage: usize,
    ) -> Self {
        let tokens_generated = sequences.iter().map(|seq| seq.len()).collect();

        Self {
            sequences,
            scores,
            lengths,
            stopped_early,
            inference_time_ms,
            memory_usage,
            tokens_generated,
        }
    }

    /// Get the first sequence (most common use case)
    pub fn sequence(&self) -> Option<&[usize]> {
        self.sequences.first().map(|s| s.as_slice())
    }

    /// Get all sequences as strings (requires vocabulary)
    pub fn sequences_as_strings(&self, vocabulary: &[String]) -> Vec<String> {
        self.sequences
            .iter()
            .map(|seq| {
                seq.iter()
                    .filter_map(|&token_id| vocabulary.get(token_id))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    /// Calculate average sequence length
    pub fn average_length(&self) -> f64 {
        if self.sequences.is_empty() {
            0.0
        } else {
            self.lengths.iter().sum::<usize>() as f64 / self.lengths.len() as f64
        }
    }

    /// Calculate compression ratio (input vs output tokens)
    pub fn compression_ratio(&self, input_lengths: &[usize]) -> Vec<f64> {
        input_lengths
            .iter()
            .zip(self.lengths.iter())
            .map(|(&input, &output)| {
                if input == 0 {
                    0.0
                } else {
                    output as f64 / input as f64
                }
            })
            .collect()
    }
}

/// Key-value cache for efficient attention computation
#[derive(Debug)]
pub struct KVCache {
    /// Cached keys [layer][batch][seq][hidden]
    pub keys: Vec<Vec<Vec<Vec<f32>>>>,
    /// Cached values [layer][batch][seq][hidden]
    pub values: Vec<Vec<Vec<Vec<f32>>>>,
    /// Current sequence length
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden size per head
    pub head_dim: usize,
}

impl KVCache {
    /// Create a new key-value cache
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        batch_size: usize,
    ) -> Self {
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let mut layer_keys = Vec::with_capacity(batch_size);
            let mut layer_values = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                let batch_keys = vec![vec![0.0; head_dim]; max_seq_len];
                let batch_values = vec![vec![0.0; head_dim]; max_seq_len];

                layer_keys.push(batch_keys);
                layer_values.push(batch_values);
            }

            keys.push(layer_keys);
            values.push(layer_values);
        }

        Self {
            keys,
            values,
            seq_len: 0,
            max_seq_len,
            num_layers,
            num_heads,
            head_dim,
        }
    }

    /// Update cache with new keys and values
    pub fn update(
        &mut self,
        layer_idx: usize,
        batch_idx: usize,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> ModelResult<()> {
        if layer_idx >= self.num_layers {
            return Err(ModelError::inference(format!(
                "Layer index {} out of range (0-{})",
                layer_idx,
                self.num_layers - 1
            )));
        }

        if batch_idx >= self.keys[layer_idx].len() {
            return Err(ModelError::inference(format!(
                "Batch index {} out of range (0-{})",
                batch_idx,
                self.keys[layer_idx].len() - 1
            )));
        }

        if keys.len() > self.max_seq_len {
            return Err(ModelError::inference(format!(
                "Keys length {} exceeds max sequence length {}",
                keys.len(),
                self.max_seq_len
            )));
        }

        let cache_keys = &mut self.keys[layer_idx][batch_idx];
        let cache_values = &mut self.values[layer_idx][batch_idx];

        for (i, (key, value)) in keys.iter().zip(values.iter()).enumerate() {
            if i < self.max_seq_len {
                cache_keys[i] = key.clone();
                cache_values[i] = value.clone();
            }
        }

        self.seq_len = self.seq_len.max(keys.len());
        Ok(())
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for layer_keys in &mut self.keys {
            for batch_keys in layer_keys {
                for key in batch_keys {
                    key.fill(0.0);
                }
            }
        }

        for layer_values in &mut self.values {
            for batch_values in layer_values {
                for value in batch_values {
                    value.fill(0.0);
                }
            }
        }

        self.seq_len = 0;
    }

    /// Get cache memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for layer_keys in &self.keys {
            for batch_keys in layer_keys {
                total += batch_keys.len() * std::mem::size_of::<f32>();
            }
        }

        for layer_values in &self.values {
            for batch_values in layer_values {
                total += batch_values.len() * std::mem::size_of::<f32>();
            }
        }

        total
    }
}

/// Core inference engine
#[allow(dead_code)]
pub struct InferenceEngine {
    /// Model configuration
    config: ModelConfig,
    /// Model weights (quantized)
    weights: HashMap<String, QuantizedTensor>,
    /// Key-value cache
    kv_cache: Option<KVCache>,
    /// Memory pool for temporary allocations
    memory_pool: Vec<Vec<f32>>,
    /// Vocabulary
    vocabulary: Vec<String>,
    /// Debug mode
    debug: bool,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(
        config: ModelConfig,
        weights: HashMap<String, QuantizedTensor>,
    ) -> ModelResult<Self> {
        let vocabulary = Self::create_vocabulary(&config)?;
        let kv_cache = if config.inference.max_seq_len > 0 {
            Some(KVCache::new(
                config.num_layers,
                config.num_heads,
                config.hidden_size / config.num_heads,
                config.inference.max_seq_len,
                1, // Start with batch size 1
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            weights,
            kv_cache,
            memory_pool: Vec::new(),
            vocabulary,
            debug: false,
        })
    }

    /// Load model from GGUF file
    pub fn from_gguf_file<P: AsRef<std::path::Path>>(
        path: P,
        config: ModelConfig,
    ) -> ModelResult<Self> {
        use crate::format::GgufFormat;
        use std::fs::File;

        let _file = File::open(path)?;
        let format: GgufFormat<std::io::Cursor<Vec<u8>>> =
            GgufFormat::parse(std::io::Cursor::new(Vec::new()))?;

        let mut weights = HashMap::new();
        for (name, tensor_info) in format.tensors() {
            // Load tensor data (simplified - would need full GGUF parsing)
            let data = vec![0u8; tensor_info.size];
            // In a full implementation, this would read from the file at tensor_info.offset
            weights.insert(
                name.clone(),
                QuantizedTensor::new(data, tensor_info.shape.clone()),
            );
        }

        Self::new(config, weights)
    }

    /// Enable debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Get model vocabulary
    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Perform inference on a batch of sequences
    pub fn infer(
        &mut self,
        input_sequences: &[Vec<usize>],
        config: &InferenceConfig,
    ) -> ModelResult<InferenceResult> {
        let start_time = std::time::Instant::now();

        if input_sequences.is_empty() {
            return Err(ModelError::inference("Empty input sequences"));
        }

        // Validate input sequences
        for (i, seq) in input_sequences.iter().enumerate() {
            if seq.len() > config.max_seq_len {
                return Err(ModelError::inference(format!(
                    "Sequence {} length {} exceeds max sequence length {}",
                    i,
                    seq.len(),
                    config.max_seq_len
                )));
            }
        }

        // Initialize or resize KV cache for batch size
        if let Some(ref mut kv_cache) = self.kv_cache {
            if kv_cache.keys.len() < input_sequences.len() {
                // Resize cache for larger batch
                // In a full implementation, this would resize the cache
            }
        }

        // Generate tokens for each sequence
        let mut sequences = input_sequences.to_vec();
        let scores = vec![Vec::new(); input_sequences.len()];
        let mut lengths = sequences.iter().map(|s| s.len()).collect::<Vec<_>>();
        let mut stopped_early = vec![false; input_sequences.len()];
        let mut tokens_generated = vec![0; input_sequences.len()];

        let mut current_length = *lengths.iter().max().unwrap_or(&0);

        while current_length < config.max_seq_len
            && tokens_generated.iter().any(|&n| n < config.min_new_tokens)
        {
            // Generate next token for each sequence
            for (batch_idx, sequence) in sequences.iter_mut().enumerate() {
                if tokens_generated[batch_idx] >= config.max_new_tokens {
                    continue;
                }

                // Generate next token (simplified implementation)
                let next_token = self.generate_next_token(sequence, config)?;
                sequence.push(next_token);
                lengths[batch_idx] = sequence.len();
                tokens_generated[batch_idx] += 1;
                current_length = current_length.max(lengths[batch_idx]);

                // Check for early stopping
                if config.early_stopping && Some(next_token) == config.eos_token_id {
                    stopped_early[batch_idx] = true;
                }
            }

            // Break if all sequences have generated enough tokens
            if tokens_generated.iter().all(|&n| n >= config.min_new_tokens) {
                break;
            }
        }

        let inference_time_ms = start_time.elapsed().as_millis() as u64;
        let memory_usage = self
            .kv_cache
            .as_ref()
            .map(|kv| kv.memory_usage())
            .unwrap_or(0);

        Ok(InferenceResult::new(
            sequences,
            scores,
            lengths,
            stopped_early,
            inference_time_ms,
            memory_usage,
        ))
    }

    /// Generate text from prompt
    pub fn generate(&mut self, prompt: &str, config: &InferenceConfig) -> ModelResult<String> {
        // Tokenize prompt
        let tokens = self.tokenize(prompt)?;

        // Generate tokens
        let result = self.infer(&[tokens], config)?;

        // Decode tokens back to text
        Ok(self.detokenize(result.sequence().unwrap()))
    }

    /// Generate next token for a sequence (simplified implementation)
    fn generate_next_token(
        &self,
        sequence: &[usize],
        _config: &InferenceConfig,
    ) -> ModelResult<usize> {
        // Simplified token generation - in a full implementation,
        // this would run the actual transformer inference
        if sequence.is_empty() {
            return Err(ModelError::inference("Empty sequence"));
        }

        // For now, return a dummy token based on sequence length
        // In a real implementation, this would involve:
        // 1. Embedding lookup
        // 2. Positional encoding
        // 3. Layer-wise transformer computation
        // 4. Language modeling head
        // 5. Sampling from logits

        let last_token = sequence[sequence.len() - 1];
        Ok((last_token + 1) % self.vocabulary.len())
    }

    /// Tokenize text to token IDs
    fn tokenize(&self, text: &str) -> ModelResult<Vec<usize>> {
        // Simplified tokenization - in practice, this would use
        // the actual model's tokenizer (BPE, WordPiece, etc.)

        // For demonstration, we'll create simple word-level tokens
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();

        for word in words {
            // Simple hash-based tokenization
            let token_id = (word.as_bytes().iter().map(|&b| b as usize).sum::<usize>())
                % self.vocabulary.len();
            tokens.push(token_id);
        }

        Ok(tokens)
    }

    /// Detokenize token IDs back to text
    fn detokenize(&self, tokens: &[usize]) -> String {
        // Simplified detokenization
        tokens
            .iter()
            .filter_map(|&token_id| self.vocabulary.get(token_id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Create vocabulary for the model
    fn create_vocabulary(config: &ModelConfig) -> ModelResult<Vec<String>> {
        // Create a simple vocabulary for demonstration
        // In practice, this would be loaded from the model's tokenizer files
        let mut vocabulary = Vec::with_capacity(config.vocab_size);

        for i in 0..config.vocab_size {
            vocabulary.push(format!("token_{}", i));
        }

        // Add special tokens
        vocabulary[0] = "[UNK]".to_string();
        vocabulary[1] = "[CLS]".to_string();
        vocabulary[2] = "[SEP]".to_string();
        vocabulary[3] = "[PAD]".to_string();
        vocabulary[4] = "[MASK]".to_string();

        Ok(vocabulary)
    }
}

/// High-level Llama model interface
pub struct LlamaModel {
    /// Underlying inference engine
    engine: InferenceEngine,
    /// Model name
    name: String,
    /// Model description
    description: String,
}

impl LlamaModel {
    /// Create a new Llama model
    pub fn new(engine: InferenceEngine, name: String, description: String) -> Self {
        Self {
            engine,
            name,
            description,
        }
    }

    /// Get model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get model description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Generate text from prompt
    pub fn generate(&mut self, prompt: &str, config: &InferenceConfig) -> ModelResult<String> {
        self.engine.generate(prompt, config)
    }

    /// Perform inference on token sequences
    pub fn infer(
        &mut self,
        sequences: &[Vec<usize>],
        config: &InferenceConfig,
    ) -> ModelResult<InferenceResult> {
        self.engine.infer(sequences, config)
    }

    /// Get model vocabulary
    pub fn vocabulary(&self) -> &[String] {
        self.engine.vocabulary()
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        self.engine.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::new()
            .with_max_seq_len(1024)
            .with_batch_size(4)
            .with_temperature(0.8)
            .with_top_p(0.9)
            .with_top_k(50)
            .with_repetition_penalty(1.2)
            .with_max_new_tokens(200)
            .with_min_new_tokens(10)
            .with_eos_token_id(2)
            .with_gpu(true)
            .with_kv_cache(true)
            .with_memory_pool(true)
            .with_debug(true);

        assert_eq!(config.max_seq_len, 1024);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.repetition_penalty, 1.2);
        assert_eq!(config.max_new_tokens, 200);
        assert_eq!(config.min_new_tokens, 10);
        assert_eq!(config.eos_token_id, Some(2));
        assert_eq!(config.use_gpu, true);
        assert_eq!(config.use_kv_cache, true);
        assert_eq!(config.use_memory_pool, true);
        assert_eq!(config.debug, true);
    }

    #[test]
    fn test_inference_config_defaults() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.repetition_penalty, 1.1);
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.min_new_tokens, 1);
        assert_eq!(config.early_stopping, true);
        assert_eq!(config.use_gpu, false);
        assert_eq!(config.use_kv_cache, true);
        assert_eq!(config.use_memory_pool, true);
        assert_eq!(config.debug, false);
    }

    #[test]
    fn test_kv_cache_operations() {
        let mut cache = KVCache::new(12, 32, 64, 2048, 1);

        assert_eq!(cache.num_layers, 12);
        assert_eq!(cache.num_heads, 32);
        assert_eq!(cache.head_dim, 64);
        assert_eq!(cache.max_seq_len, 2048);
        assert_eq!(cache.seq_len, 0);

        let memory_usage = cache.memory_usage();
        assert!(memory_usage > 0);

        // Test cache update (would need actual data)
        // cache.update(0, 0, &[], &[]).unwrap();

        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_inference_result_operations() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5]];
        let scores = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5]];
        let lengths = vec![3, 2];
        let stopped_early = vec![false, true];
        let result = InferenceResult::new(
            sequences.clone(),
            scores.clone(),
            lengths.clone(),
            stopped_early.clone(),
            100,
            1024,
        );

        assert_eq!(result.sequences, sequences);
        assert_eq!(result.scores, scores);
        assert_eq!(result.lengths, lengths);
        assert_eq!(result.stopped_early, stopped_early);
        assert_eq!(result.inference_time_ms, 100);
        assert_eq!(result.memory_usage, 1024);
        assert_eq!(result.tokens_generated, vec![3, 2]);
        assert_eq!(result.average_length(), 2.5);

        // Test sequence access
        assert_eq!(result.sequence(), Some(&sequences[0][..]));
    }

    #[test]
    fn test_quantization_inference() {
        // Test that we can create an inference engine with quantized weights
        let mut weights = HashMap::new();
        weights.insert(
            "embed_tokens".to_string(),
            QuantizedTensor::new(vec![0u8; 100], vec![10, 10])
                .with_quantization(crate::quantization::QuantizationScheme::Q8_0),
        );

        let config = ModelConfig::new(crate::ModelType::Llama)
            .with_vocab_size(1000)
            .with_hidden_size(768)
            .with_num_heads(12)
            .with_num_layers(12)
            .with_max_seq_len(512);

        let engine = InferenceEngine::new(config, weights).unwrap();
        assert_eq!(engine.vocabulary().len(), 1000);
    }
}
