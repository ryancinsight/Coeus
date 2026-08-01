//! Benchmarks for `coeus-optim`: per-step optimizer throughput across parameter
//! sizes. Optimizer updates are element-wise and memory-bandwidth-bound, so these
//! track regressions in the parameter-update path (allocation, moment buffers,
//! backend dispatch) as tensor size grows.

use coeus_autograd::{Parameter, Var};
use coeus_core::MoiraiBackend;
use coeus_optim::{Adam, AdamW, Optimizer, SGD};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Representative parameter tensor sizes: small layer, medium, large embedding.
const SIZES: [usize; 5] = [4_096, 16_384, 65_536, 262_144, 1_048_576];

fn make_param(n: usize) -> Var<f32, MoiraiBackend> {
    let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    Var::new(Tensor::from_slice([n], &data), true)
}

fn grad_of(n: usize) -> Tensor<f32, MoiraiBackend> {
    let g: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.05 - 0.1).collect();
    Tensor::from_slice([n], &g)
}

fn bench_sgd_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("coeus-optim SGD step (momentum=0.9)");
    for &n in &SIZES {
        let mut opt = SGD::new(vec![Parameter::new(make_param(n), "weight")], 0.01, 0.9);
        opt.params[0].set_grad(grad_of(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                opt.step().expect("SGD benchmark step");
                black_box(&opt.params[0].tensor);
            });
        });
    }
    group.finish();
}

fn bench_adam_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("coeus-optim Adam step");
    for &n in &SIZES {
        let mut opt = Adam::new(
            vec![Parameter::new(make_param(n), "weight")],
            0.001,
            0.9,
            0.999,
            1e-8,
        );
        opt.params[0].set_grad(grad_of(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                opt.step().expect("Adam benchmark step");
                black_box(&opt.params[0].tensor);
            });
        });
    }
    group.finish();
}

fn bench_adamw_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("coeus-optim AdamW step (weight_decay=0.01)");
    for &n in &SIZES {
        let mut opt = AdamW::new(
            vec![Parameter::new(make_param(n), "weight")],
            0.001,
            0.9,
            0.999,
            1e-8,
            0.01,
        );
        opt.params[0].set_grad(grad_of(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                opt.step().expect("AdamW benchmark step");
                black_box(&opt.params[0].tensor);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sgd_step, bench_adam_step, bench_adamw_step);
criterion_main!(benches);
