//! Benchmarks for tokenizer implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustllm_core::core::tokenizer::Tokenizer;
use rustllm_tokenizer_basic::BasicTokenizer;
use rustllm_tokenizer_bpe::BpeTokenizer;

const SAMPLE_TEXTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the Word, and the Word was with God, and the Word was God.",
    "To be or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
];

const LONG_TEXT: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris \
    nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in \
    reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla \
    pariatur. Excepteur sint occaecat cupidatat non proident, sunt in \
    culpa qui officia deserunt mollit anim id est laborum.";

fn benchmark_tokenization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenization");
    
    // Basic tokenizer benchmarks
    let basic_tokenizer = BasicTokenizer::new();
    
    for (i, text) in SAMPLE_TEXTS.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("basic", i),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens: Vec<_> = basic_tokenizer.tokenize(black_box(text)).collect();
                    tokens
                });
            },
        );
    }
    
    // BPE tokenizer benchmarks
    let bpe_tokenizer = BpeTokenizer::new();
    
    for (i, text) in SAMPLE_TEXTS.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("bpe", i),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens: Vec<_> = bpe_tokenizer.tokenize(black_box(text)).collect();
                    tokens
                });
            },
        );
    }
    
    // Long text benchmark
    group.bench_function("basic_long", |b| {
        b.iter(|| {
            let tokens: Vec<_> = basic_tokenizer.tokenize(black_box(LONG_TEXT)).collect();
            tokens
        });
    });
    
    group.bench_function("bpe_long", |b| {
        b.iter(|| {
            let tokens: Vec<_> = bpe_tokenizer.tokenize(black_box(LONG_TEXT)).collect();
            tokens
        });
    });
    
    group.finish();
}

fn benchmark_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoding");
    
    // Prepare tokens for decoding
    let basic_tokenizer = BasicTokenizer::new();
    let bpe_tokenizer = BpeTokenizer::new();
    
    let basic_tokens: Vec<_> = basic_tokenizer.tokenize(LONG_TEXT).collect();
    let bpe_tokens: Vec<_> = bpe_tokenizer.tokenize(LONG_TEXT).collect();
    
    group.bench_function("basic_decode", |b| {
        b.iter(|| {
            let decoded = basic_tokenizer.decode(black_box(basic_tokens.clone())).unwrap();
            decoded
        });
    });
    
    group.bench_function("bpe_decode", |b| {
        b.iter(|| {
            let decoded = bpe_tokenizer.decode(black_box(bpe_tokens.clone())).unwrap();
            decoded
        });
    });
    
    group.finish();
}

fn benchmark_bpe_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_training");
    group.sample_size(10); // Training is slow, reduce sample size
    
    let corpus = vec![
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
        "a cat is not a dog",
        "the mat was on the floor",
        "the log was in the forest",
    ];
    
    for num_merges in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("merges", num_merges),
            num_merges,
            |b, &num_merges| {
                b.iter(|| {
                    let mut tokenizer = BpeTokenizer::new();
                    tokenizer.train(black_box(&corpus), black_box(num_merges)).unwrap();
                    tokenizer
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_vocabulary_operations(c: &mut Criterion) {
    use rustllm_core::core::tokenizer::VocabularyTokenizer;
    
    let mut group = c.benchmark_group("vocabulary");
    
    let mut tokenizer = BpeTokenizer::new();
    
    group.bench_function("add_token", |b| {
        let mut counter = 0;
        b.iter(|| {
            let token = format!("token_{}", counter);
            tokenizer.add_token(black_box(&token)).unwrap();
            counter += 1;
        });
    });
    
    group.bench_function("contains_token", |b| {
        b.iter(|| {
            tokenizer.contains_token(black_box("hello"))
        });
    });
    
    group.bench_function("id_from_token", |b| {
        b.iter(|| {
            tokenizer.id_from_token(black_box("hello"))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_tokenization,
    benchmark_decoding,
    benchmark_bpe_training,
    benchmark_vocabulary_operations
);
criterion_main!(benches);