//! Benchmarks for advanced iterator combinators.
//!
//! This benchmark suite demonstrates the zero-cost nature of our
//! iterator abstractions and memory management utilities.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustllm_core::prelude::*;

/// Benchmark map iterator combinator.
fn bench_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("map");

    for size in [100, 1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::new("map", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

            b.iter(|| {
                let result: Vec<f32> = data.iter().copied().map(|x| x * 2.0 + 1.0).collect();
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark the batch iterator combinator.
fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");

    for batch_size in [16, 32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &batch_size| {
                let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

                b.iter(|| {
                    let batches: Vec<Vec<f32>> =
                        data.iter().copied().batch(batch_size, batch_size).collect();
                    black_box(batches)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("chunks", batch_size),
            &batch_size,
            |b, &batch_size| {
                let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

                b.iter(|| {
                    let batches: Vec<Vec<f32>> = data
                        .chunks(batch_size)
                        .map(|chunk| chunk.to_vec())
                        .collect();
                    black_box(batches)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark rolling average using windows with const generics.
fn bench_rolling_avg(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_avg");

    for window_size in [5usize, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("avg", window_size),
            &window_size,
            |b, &window_size| {
                let data: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();

                b.iter(|| match window_size {
                    5 => {
                        let result: Vec<f32> = data
                            .iter()
                            .copied()
                            .windows::<5>()
                            .map(|w| w.iter().sum::<f32>() / 5.0)
                            .collect();
                        black_box(result)
                    },
                    10 => {
                        let result: Vec<f32> = data
                            .iter()
                            .copied()
                            .windows::<10>()
                            .map(|w| w.iter().sum::<f32>() / 10.0)
                            .collect();
                        black_box(result)
                    },
                    20 => {
                        let result: Vec<f32> = data
                            .iter()
                            .copied()
                            .windows::<20>()
                            .map(|w| w.iter().sum::<f32>() / 20.0)
                            .collect();
                        black_box(result)
                    },
                    50 => {
                        let result: Vec<f32> = data
                            .iter()
                            .copied()
                            .windows::<50>()
                            .map(|w| w.iter().sum::<f32>() / 50.0)
                            .collect();
                        black_box(result)
                    },
                    _ => unreachable!(),
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the prefetch iterator combinator.
fn bench_prefetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch");

    for prefetch_size in [16usize, 32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("with_prefetch", prefetch_size),
            &prefetch_size,
            |b, &prefetch_size| {
                let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

                b.iter(|| match prefetch_size {
                    16 => black_box(
                        data.iter()
                            .copied()
                            .prefetch::<16>()
                            .map(|x| x * 2.0)
                            .sum::<f32>(),
                    ),
                    32 => black_box(
                        data.iter()
                            .copied()
                            .prefetch::<32>()
                            .map(|x| x * 2.0)
                            .sum::<f32>(),
                    ),
                    64 => black_box(
                        data.iter()
                            .copied()
                            .prefetch::<64>()
                            .map(|x| x * 2.0)
                            .sum::<f32>(),
                    ),
                    128 => black_box(
                        data.iter()
                            .copied()
                            .prefetch::<128>()
                            .map(|x| x * 2.0)
                            .sum::<f32>(),
                    ),
                    _ => unreachable!(),
                });
            },
        );
    }

    // Baseline without prefetch
    group.bench_function("baseline", |b| {
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        b.iter(|| {
            let result: f32 = data.iter().copied().map(|x| x * 2.0).sum();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark zero-copy string builder.
fn bench_zero_copy_string_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_builder");

    group.bench_function("zero_copy_builder", |b| {
        b.iter(|| {
            let mut builder = ZeroCopyStringBuilder::new();
            for i in 0..100 {
                builder
                    .append_borrowed("Item ")
                    .append_owned(i.to_string())
                    .append_borrowed(", ");
            }
            let result = builder.build();
            black_box(result)
        });
    });

    group.bench_function("string_push", |b| {
        b.iter(|| {
            let mut result = String::new();
            for i in 0..100 {
                result.push_str("Item ");
                result.push_str(&i.to_string());
                result.push_str(", ");
            }
            black_box(result)
        });
    });

    group.bench_function("format_macro", |b| {
        b.iter(|| {
            let mut result = String::new();
            for i in 0..100 {
                result.push_str(&format!("Item {}, ", i));
            }
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark slice view with lazy transformation.
fn bench_slice_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_view");

    let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

    group.bench_function("slice_view_lazy", |b| {
        b.iter(|| {
            let view = SliceView::new(&data).map(|x| x * 2.0 + 1.0);

            // Access only a few elements
            let mut sum = 0.0;
            for i in (0..data.len()).step_by(100) {
                if let Some(val) = view.get(i) {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("eager_transform", |b| {
        b.iter(|| {
            let transformed: Vec<f32> = data.iter().map(|&x| x * 2.0 + 1.0).collect();

            // Access only a few elements
            let mut sum = 0.0;
            for i in (0..transformed.len()).step_by(100) {
                sum += transformed[i];
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Benchmark combined iterator chains.
fn bench_combined_iterators(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_iterators");

    group.bench_function("complex_chain", |b| {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();

        b.iter(|| {
            let result: Vec<f32> = data
                .iter()
                .copied()
                                 .map(|x| x.sin())
                 .windows::<5>()
                 .map(|w| w.iter().sum::<f32>() / 5.0)
                 .batch(32, 32)
                 .flat_map(|batch| batch.into_iter())
                 .take(500)
                 .collect();
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_map,
    bench_batch,
    bench_rolling_avg,
    bench_prefetch,
    bench_zero_copy_string_builder,
    bench_slice_view,
    bench_combined_iterators
);
criterion_main!(benches);
