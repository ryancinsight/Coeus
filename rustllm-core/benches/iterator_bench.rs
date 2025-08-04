//! Benchmarks for iterator combinators.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustllm_core::foundation::iterator::{IteratorExt, StrTokens};

fn benchmark_windows(c: &mut Criterion) {
    let mut group = c.benchmark_group("windows");
    
    let data: Vec<i32> = (0..1000).collect();
    
    for window_size in [2, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("collect", window_size),
            window_size,
            |b, &window_size| {
                b.iter(|| {
                    let windows: Vec<_> = data.iter()
                        .cloned()
                        .windows(window_size)
                        .collect();
                    windows
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("count", window_size),
            window_size,
            |b, &window_size| {
                b.iter(|| {
                    data.iter()
                        .cloned()
                        .windows(window_size)
                        .count()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunks");
    
    let data: Vec<i32> = (0..1000).collect();
    
    for chunk_size in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("collect", chunk_size),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let chunks: Vec<_> = data.iter()
                        .cloned()
                        .chunks(chunk_size)
                        .collect();
                    chunks
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("flatten", chunk_size),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let flattened: Vec<_> = data.iter()
                        .cloned()
                        .chunks(chunk_size)
                        .flatten()
                        .collect();
                    flattened
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_stride(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride");
    
    let data: Vec<i32> = (0..10000).collect();
    
    for step in [2, 5, 10, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("collect", step),
            step,
            |b, &step| {
                b.iter(|| {
                    let strided: Vec<_> = data.iter()
                        .cloned()
                        .stride(step)
                        .collect();
                    strided
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_str_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_tokens");
    
    let texts = [
        "The quick brown fox",
        "Lorem ipsum dolor sit amet consectetur adipiscing elit",
        include_str!("../src/lib.rs").lines().next().unwrap(), // First line of lib.rs
    ];
    
    for (i, text) in texts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("tokenize", i),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens: Vec<_> = StrTokens::new(black_box(text)).collect();
                    tokens
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_chained_iterators(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained");
    
    let data: Vec<i32> = (0..1000).collect();
    
    group.bench_function("windows_filter_map", |b| {
        b.iter(|| {
            let result: Vec<_> = data.iter()
                .cloned()
                .windows(3)
                .filter(|w| w[0] % 2 == 0)
                .map(|w| w.iter().sum::<i32>())
                .collect();
            result
        });
    });
    
    group.bench_function("chunks_stride", |b| {
        b.iter(|| {
            let result: Vec<_> = data.iter()
                .cloned()
                .chunks(10)
                .stride(2)
                .map(|chunk| chunk.iter().sum::<i32>())
                .collect();
            result
        });
    });
    
    group.bench_function("complex_chain", |b| {
        b.iter(|| {
            let result: Vec<_> = data.iter()
                .cloned()
                .windows(5)
                .stride(2)
                .filter(|w| w.iter().any(|&x| x % 7 == 0))
                .map(|w| w.iter().product::<i32>())
                .chunks(10)
                .map(|chunk| chunk.iter().max().copied().unwrap_or(0))
                .collect();
            result
        });
    });
    
    group.finish();
}

fn benchmark_collect_with_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("collect_capacity");
    
    let data: Vec<i32> = (0..10000).collect();
    
    group.bench_function("standard_collect", |b| {
        b.iter(|| {
            let result: Vec<_> = data.iter()
                .cloned()
                .filter(|&x| x % 2 == 0)
                .collect();
            result
        });
    });
    
    group.bench_function("collect_with_capacity", |b| {
        b.iter(|| {
            let result = data.iter()
                .cloned()
                .filter(|&x| x % 2 == 0)
                .collect_vec_with_capacity();
            result
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_windows,
    benchmark_chunks,
    benchmark_stride,
    benchmark_str_tokens,
    benchmark_chained_iterators,
    benchmark_collect_with_capacity
);
criterion_main!(benches);