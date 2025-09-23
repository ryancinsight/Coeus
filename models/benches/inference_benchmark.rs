//! Benchmarks for the models crate inference engine.

#![allow(unused_imports)]
#![allow(unused_variables)]
//!
//! This module provides comprehensive benchmarks for model inference performance,
//! including memory usage, inference speed, and quantization efficiency.

use coeus_models::{
    config::ModelConfig,
    inference::{InferenceConfig, InferenceEngine, KVCache},
    quantization::{QuantizationScheme, QuantizedTensor},
    ModelType,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark inference performance with different configurations
fn bench_inference_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_performance");
    group.throughput(criterion::Throughput::Elements(1));

    // Create a test model configuration
    let config = ModelConfig::new(ModelType::Llama)
        .with_vocab_size(1000)
        .with_hidden_size(768)
        .with_num_heads(12)
        .with_num_layers(12)
        .with_max_seq_len(512);

    // Create test weights
    let mut weights = std::collections::HashMap::new();
    weights.insert(
        "embed_tokens".to_string(),
        QuantizedTensor::new(vec![0u8; 1000 * 768], vec![1000, 768])
            .with_quantization(QuantizationScheme::Q8_0),
    );

    let mut engine = InferenceEngine::new(config, weights).unwrap();

    let test_prompt = "The future of artificial intelligence is";
    let inference_config = InferenceConfig::new()
        .with_max_new_tokens(50)
        .with_temperature(0.8);

    group.bench_function("llama_inference_50_tokens", |b| {
        b.iter(|| {
            let result = engine.generate(black_box(test_prompt), black_box(&inference_config));
            black_box(result).unwrap();
        })
    });

    group.finish();
}

/// Benchmark memory usage with different quantization schemes
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test different quantization schemes
    let schemes = vec![
        QuantizationScheme::F32,
        QuantizationScheme::Q8_0,
        QuantizationScheme::Q4_0,
    ];

    for scheme in schemes {
        group.bench_function(format!("memory_usage_{}", scheme.name()), |b| {
            b.iter(|| {
                let data_size = 1000 * 768 * 4; // Original F32 size
                let quantized_size = scheme.memory_usage(data_size);
                black_box(quantized_size)
            })
        });
    }

    group.finish();
}

/// Benchmark key-value cache operations
fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");

    // Test cache creation
    group.bench_function("cache_creation", |b| {
        b.iter(|| {
            let cache = KVCache::new(12, 32, 64, 2048, 1);
            black_box(cache)
        })
    });

    // Test cache update
    let mut cache = KVCache::new(12, 32, 64, 2048, 1);
    let test_keys = vec![vec![1.0; 64]; 32];
    let test_values = vec![vec![2.0; 64]; 32];

    group.bench_function("cache_update", |b| {
        b.iter(|| {
            cache
                .update(
                    black_box(0),
                    black_box(0),
                    black_box(&test_keys),
                    black_box(&test_values),
                )
                .unwrap();
        })
    });

    group.finish();
}

/// Benchmark model loading performance
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");

    group.bench_function("model_config_creation", |b| {
        b.iter(|| {
            let config = ModelConfig::new(ModelType::Llama)
                .with_vocab_size(1000)
                .with_hidden_size(768)
                .with_num_heads(12)
                .with_num_layers(12)
                .with_max_seq_len(512);
            black_box(config)
        })
    });

    group.bench_function("inference_engine_creation", |b| {
        b.iter(|| {
            let config = ModelConfig::new(ModelType::Llama)
                .with_vocab_size(1000)
                .with_hidden_size(768)
                .with_num_heads(12)
                .with_num_layers(12)
                .with_max_seq_len(512);

            let weights = std::collections::HashMap::new();
            let engine = InferenceEngine::new(black_box(config), black_box(weights)).unwrap();
            black_box(engine)
        })
    });

    group.finish();
}

/// Benchmark tokenization performance
fn bench_tokenization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenization");

    // Create a test vocabulary
    let _vocabulary: Vec<String> = (0..1000).map(|i| format!("token_{}", i)).collect();

    group.bench_function("simple_tokenization", |b| {
        b.iter(|| {
            let text = "This is a test sentence for tokenization benchmarking purposes.";
            let tokens: Vec<usize> = text
                .split_whitespace()
                .map(|word| (word.as_bytes().iter().map(|&b| b as usize).sum::<usize>()) % 1000)
                .collect();
            black_box(tokens)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_inference_performance,
    bench_memory_usage,
    bench_kv_cache,
    bench_model_loading,
    bench_tokenization
);

criterion_main!(benches);
