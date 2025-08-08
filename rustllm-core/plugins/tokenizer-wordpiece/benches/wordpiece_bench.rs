use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustllm_core::core::tokenizer::{Tokenizer, TokenizerConfig};
use rustllm_tokenizer_wordpiece::WordPieceTokenizer;

fn benchmark_tokenization(c: &mut Criterion) {
    let config = TokenizerConfig::default();
    let tokenizer = WordPieceTokenizer::new(config).unwrap();

    let test_texts = vec![
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog. This is a test sentence for tokenization."),
        ("long", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."),
    ];

    let mut group = c.benchmark_group("wordpiece_tokenization");

    for (name, text) in test_texts {
        group.bench_with_input(BenchmarkId::new("encode", name), &text, |b, text| {
            b.iter(|| {
                let tokens = tokenizer.encode(black_box(text)).unwrap();
                black_box(tokens)
            })
        });

        // Benchmark decode as well
        let tokens = tokenizer.encode(text).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", name), &tokens, |b, tokens| {
            b.iter(|| {
                let decoded = tokenizer.decode(black_box(tokens)).unwrap();
                black_box(decoded)
            })
        });
    }

    group.finish();
}

fn benchmark_trie_operations(c: &mut Criterion) {
    use rustllm_tokenizer_wordpiece::*;

    // This would require making TrieNode public or adding benchmark methods
    // For now, we'll benchmark through the tokenizer interface
    let config = TokenizerConfig::default();
    let tokenizer = WordPieceTokenizer::new(config).unwrap();

    c.bench_function("vocab_lookup", |b| {
        b.iter(|| {
            let result = tokenizer.token_to_id(black_box("hello"));
            black_box(result)
        })
    });
}

criterion_group!(benches, benchmark_tokenization, benchmark_trie_operations);
criterion_main!(benches);
