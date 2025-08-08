//! Benchmarks for iterator combinators.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustllm_core::foundation::iterator::{IteratorExt, ZeroCopySplit};

fn benchmark_windows(c: &mut Criterion) {
    let mut group = c.benchmark_group("windows");

    let data: Vec<i32> = (0..1000).collect();

    group.bench_function("windows_2_collect", |b| {
        b.iter(|| {
            let windows: Vec<_> = data.iter().cloned().windows::<2>().collect();
            black_box(windows)
        });
    });
    group.bench_function("windows_5_collect", |b| {
        b.iter(|| {
            let windows: Vec<_> = data.iter().cloned().windows::<5>().collect();
            black_box(windows)
        });
    });
    group.bench_function("windows_10_count", |b| {
        b.iter(|| {
            let count = data.iter().cloned().windows::<10>().count();
            black_box(count)
        });
    });

    group.finish();
}

fn benchmark_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunks");

    let data: Vec<i32> = (0..1000).collect();

    group.bench_function("chunks_10_collect", |b| {
        b.iter(|| {
            let chunks: Vec<_> = data.iter().cloned().chunks::<10>().collect();
            black_box(chunks)
        });
    });

    group.bench_function("chunks_50_flatten", |b| {
        b.iter(|| {
            let flattened: Vec<_> = data.iter().cloned().chunks::<50>().flatten().collect();
            black_box(flattened)
        });
    });

    group.finish();
}

fn benchmark_stride(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride");

    let data: Vec<i32> = (0..10000).collect();

    group.bench_function("stride_2_collect", |b| {
        b.iter(|| {
            let strided: Vec<_> = data.iter().cloned().stride::<2>().collect();
            black_box(strided)
        });
    });

    group.bench_function("stride_5_collect", |b| {
        b.iter(|| {
            let strided: Vec<_> = data.iter().cloned().stride::<5>().collect();
            black_box(strided)
        });
    });

    group.finish();
}

fn benchmark_str_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("str_split");

    let texts = [
        "The quick brown fox",
        "Lorem ipsum dolor sit amet consectetur adipiscing elit",
        include_str!("../src/lib.rs").lines().next().unwrap(),
    ];

    for (i, text) in texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("tokenize", i), text, |b, text| {
            b.iter(|| {
                let tokens: Vec<_> = ZeroCopySplit::new(black_box(text), ' ').collect();
                black_box(tokens)
            });
        });
    }

    group.finish();
}

fn benchmark_chained_iterators(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained");

    let data: Vec<i32> = (0..1000).collect();

    group.bench_function("windows3_filter_map", |b| {
        b.iter(|| {
            let result: Vec<_> = data
                .iter()
                .cloned()
                .windows::<3>()
                .filter(|w| w[0] % 2 == 0)
                .map(|w| w.iter().sum::<i32>())
                .collect();
            black_box(result)
        });
    });

    group.bench_function("chunks10_stride2", |b| {
        b.iter(|| {
            let result: Vec<_> = data
                .iter()
                .cloned()
                .chunks::<10>()
                .stride::<2>()
                .map(|chunk| chunk.iter().sum::<i32>())
                .collect();
            black_box(result)
        });
    });

    group.bench_function("complex_chain", |b| {
        b.iter(|| {
            let result: Vec<_> = data
                .iter()
                .cloned()
                .windows::<5>()
                .stride::<2>()
                .filter(|w| w.iter().any(|&x| x % 7 == 0))
                .map(|w| w.iter().map(|&x| x as i64).product::<i64>())
                .chunks::<10>()
                .map(|chunk| chunk.iter().max().copied().unwrap_or(0))
                .collect();
            black_box(result)
        });
    });

    group.finish();
}

fn benchmark_collect_with_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("collect_capacity");

    let data: Vec<i32> = (0..10000).collect();

    group.bench_function("standard_collect", |b| {
        b.iter(|| {
            let result: Vec<_> = data.iter().cloned().filter(|&x| x % 2 == 0).collect();
            black_box(result)
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
