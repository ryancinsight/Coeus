//! Benchmarks for `coeus-fft`: Apollo-backed FFT forward throughput and the
//! autograd forward+backward round-trip (measuring the reverse-mode overhead
//! over the raw transform).

use coeus_autograd::Var;
use coeus_core::{Complex, MoiraiBackend};
use coeus_fft::{fft_1d, fft_1d_var};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Deterministic non-trivial real signal of length `n`.
fn signal_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.1).sin() + 0.5).collect()
}

const SIZES: [usize; 3] = [256, 1024, 4096];

fn bench_fft_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("coeus-fft fft_1d forward");
    for &n in &SIZES {
        let signal = Tensor::<f64, MoiraiBackend>::from_slice([n], &signal_data(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &signal, |b, s| {
            b.iter(|| black_box(fft_1d(black_box(s))));
        });
    }
    group.finish();
}

fn bench_fft_autograd_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("coeus-fft fft_1d_var forward+backward");
    for &n in &SIZES {
        let data = signal_data(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, d| {
            b.iter(|| {
                let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([d.len()], d), true);
                let y = fft_1d_var(black_box(&x));
                let seed = Tensor::from_slice([d.len()], &vec![Complex::new(1.0, 0.0); d.len()]);
                y.backward_with_seed(seed);
                black_box(x.grad());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fft_forward, bench_fft_autograd_roundtrip);
criterion_main!(benches);
