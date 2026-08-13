//! Wall-clock benchmarks for the flat-index decode kernels in `coeus-ops`.
//!
//! These time the real production paths — `gather`, `index_select`, `scatter_add`,
//! `repeat_interleave`, and `topk` — on shapes whose decode work is dominated by
//! the per-element index arithmetic rather than by the element copy. The harness
//! body is never tuned to move a number; if a result regresses, the fix belongs
//! in the kernel.
//!
//! ## Runtime budget
//!
//! The suite is budgeted analytically rather than left at criterion's defaults,
//! which would spend ~6 s per case (3 s warm-up + 3 s measurement) and put this
//! binary far past any reasonable bound.
//!
//! ```text
//! per case  = WARM_UP_MS + MEASUREMENT_MS            = 0.25 s + 1.10 s = 1.35 s
//! cases     = 5 kernels x 2 shapes                   = 10
//! estimate  = 10 x 1.35 s                            = 13.5 s
//! bound     = BUDGET_SECS                            = 30 s
//! ```
//!
//! The ~2x headroom absorbs criterion's own analysis and slower hosts. The bound
//! is enforced, not merely documented: [`main`] fails with a non-zero exit if the
//! suite exceeds [`BUDGET_SECS`], so a breach surfaces as a failure instead of
//! silently drifting. A breach is a defect to root-cause — either an oversized
//! workload here or a genuinely slower kernel — and is never resolved by raising
//! the bound in the same change that caused it.
//!
//! Run everything:
//!   `cargo bench -p coeus-ops --bench index_ops_bench`
//! Run one kernel:
//!   `cargo bench -p coeus-ops --bench index_ops_bench -- gather`

use std::time::{Duration, Instant};

use criterion::{black_box, Criterion};

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

/// Hard wall-clock bound for this binary. See the module budget table.
const BUDGET_SECS: u64 = 30;
/// Per-case warm-up. Long enough to reach steady state on these kernels.
const WARM_UP_MS: u64 = 250;
/// Per-case measurement window.
const MEASUREMENT_MS: u64 = 1_100;
/// Samples per case. Below criterion's default of 100 to fit the budget while
/// still giving the median a usable interval.
const SAMPLE_SIZE: usize = 30;

/// Shapes chosen so the decode cost dominates: many elements, rank 3, and an
/// interior selection axis (the case that exercises coordinates on both sides
/// of `dim`).
const SHAPES: [(&str, [usize; 3]); 2] = [("small", [8, 16, 8]), ("large", [32, 64, 32])];

fn tensor(shape: &[usize], values: &[f64]) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice_on(shape.to_vec(), values, &SequentialBackend::default())
}

/// Ramp so every element is distinct — a wrong index cannot alias a right one.
fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 + 1.0).collect()
}

/// Index tensor over `dim`, cycling deterministically through the valid range.
fn indices(n: usize, extent: usize) -> Vec<f64> {
    (0..n).map(|i| (i % extent) as f64).collect()
}

fn bench_gather(c: &mut Criterion, label: &str, shape: [usize; 3]) {
    let numel = shape.iter().product::<usize>();
    let input = tensor(&shape, &ramp(numel));
    // Index matches the input on every axis but `dim`, halving the gathered axis.
    let idx_shape = [shape[0], shape[1] / 2, shape[2]];
    let idx_numel = idx_shape.iter().product::<usize>();
    let index = tensor(&idx_shape, &indices(idx_numel, shape[1]));
    let backend = SequentialBackend::default();

    c.bench_function(&format!("gather/dim1/{label}"), |b| {
        b.iter(|| {
            let out = coeus_ops::gather(black_box(&input), 1, black_box(&index), &backend);
            black_box(out)
        });
    });
}

fn bench_index_select(c: &mut Criterion, label: &str, shape: [usize; 3]) {
    let numel = shape.iter().product::<usize>();
    let input = tensor(&shape, &ramp(numel));
    let take = shape[1] / 2;
    let index = tensor(&[take], &indices(take, shape[1]));
    let backend = SequentialBackend::default();

    c.bench_function(&format!("index_select/dim1/{label}"), |b| {
        b.iter(|| {
            let out = coeus_ops::index_select(black_box(&input), 1, black_box(&index), &backend);
            black_box(out)
        });
    });
}

fn bench_scatter_add(c: &mut Criterion, label: &str, shape: [usize; 3]) {
    let numel = shape.iter().product::<usize>();
    let input = tensor(&shape, &ramp(numel));
    // `index` and `src` share a shape; scattering into a halved axis keeps the
    // decode loop bounded by the source rather than the destination.
    let src_shape = [shape[0], shape[1] / 2, shape[2]];
    let src_numel = src_shape.iter().product::<usize>();
    let src = tensor(&src_shape, &ramp(src_numel));
    let index = tensor(&src_shape, &indices(src_numel, shape[1]));
    let backend = SequentialBackend::default();

    c.bench_function(&format!("scatter_add/dim1/{label}"), |b| {
        b.iter(|| {
            let out = coeus_ops::scatter_add(
                black_box(&input),
                1,
                black_box(&index),
                black_box(&src),
                &backend,
            );
            black_box(out)
        });
    });
}

fn bench_repeat_interleave(c: &mut Criterion, label: &str, shape: [usize; 3]) {
    let numel = shape.iter().product::<usize>();
    let input = tensor(&shape, &ramp(numel));
    let backend = SequentialBackend::default();

    c.bench_function(&format!("repeat_interleave/dim1/{label}"), |b| {
        b.iter(|| {
            let out = coeus_ops::repeat_interleave(black_box(&input), 2, 1, &backend);
            black_box(out)
        });
    });
}

fn bench_topk(c: &mut Criterion, label: &str, shape: [usize; 3]) {
    let numel = shape.iter().product::<usize>();
    let input = tensor(&shape, &ramp(numel));
    let k = (shape[1] / 4).max(1);

    c.bench_function(&format!("topk/dim1/{label}"), |b| {
        b.iter(|| {
            let out = coeus_ops::topk(black_box(&input), k, 1, true);
            black_box(out)
        });
    });
}

fn main() {
    let started = Instant::now();

    // Logging: state the budget and the workload up front so a recorded run is
    // self-describing and a later comparison can confirm the shapes matched.
    eprintln!("index_ops_bench: budget {BUDGET_SECS}s (hard), warm-up {WARM_UP_MS}ms, measurement {MEASUREMENT_MS}ms, samples {SAMPLE_SIZE}");
    for (label, shape) in SHAPES {
        eprintln!("index_ops_bench: shape {label} = {shape:?}");
    }

    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(WARM_UP_MS))
        .measurement_time(Duration::from_millis(MEASUREMENT_MS))
        .sample_size(SAMPLE_SIZE)
        .configure_from_args();

    for (label, shape) in SHAPES {
        bench_gather(&mut criterion, label, shape);
        bench_index_select(&mut criterion, label, shape);
        bench_scatter_add(&mut criterion, label, shape);
        bench_repeat_interleave(&mut criterion, label, shape);
        bench_topk(&mut criterion, label, shape);
    }

    criterion.final_summary();

    let elapsed = started.elapsed();
    eprintln!(
        "index_ops_bench: completed in {:.2}s of {BUDGET_SECS}s budget",
        elapsed.as_secs_f64()
    );

    // Enforce the bound rather than trusting the estimate. Exceeding it is a
    // defect in the kernel or in this instrument's workload sizing.
    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "index_ops_bench exceeded its {BUDGET_SECS}s budget ({:.2}s). Root-cause the \
         slowdown; do not raise the bound in the change that caused it.",
        elapsed.as_secs_f64()
    );
}
