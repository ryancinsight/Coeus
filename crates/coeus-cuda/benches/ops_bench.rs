// On-device (CUDA) vs CPU (SequentialBackend) baselines for matmul and
// transposed convolution — the CUDA counterpart to coeus-wgpu's ops_bench.
//
// Meaningful only with a live CUDA device AND `--features cuda`:
//   cargo bench -p coeus-cuda --features cuda --bench ops_bench
// Without a CUDA context the on-device groups register nothing (the guard
// returns early), so a plain run produces no misleading fallback numbers.
//
// Machine class: depends on the local GPU/driver and CPU; record the host when
// committing a baseline. Not run in headless CI.

use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::conv_transpose2d;
use coeus_tensor::Tensor;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn cuda_available() -> bool {
    coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some()
}

fn fill(n: usize, phase: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + phase) * 0.013).sin()).collect()
}

fn bench_matmul(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let mut group = c.benchmark_group("CUDA Matmul (square)");
    for &sz in &[128usize, 512] {
        let a = fill(sz * sz, 1.0);
        let b = fill(sz * sz, 2.0);
        let a_cpu = Tensor::<f32, SequentialBackend>::from_slice([sz, sz], &a);
        let b_cpu = Tensor::<f32, SequentialBackend>::from_slice([sz, sz], &b);
        let a_g = a_cpu.to_backend_on(&seq, &cuda);
        let b_g = b_cpu.to_backend_on(&seq, &cuda);
        let id = format!("{sz}x{sz}");

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(coeus_ops::matmul(
                    black_box(&a_cpu),
                    black_box(&b_cpu),
                    &seq,
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("Coeus CUDA", &id), &id, |bn, _| {
            bn.iter(|| black_box(coeus_ops::matmul(black_box(&a_g), black_box(&b_g), &cuda)));
        });
    }
    group.finish();
}

fn bench_conv_transpose2d(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();
    const CASES: &[(usize, usize, usize, usize, usize, usize)] =
        &[(4, 8, 16, 16, 8, 3), (4, 16, 32, 32, 16, 3)];
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 1usize, 1usize);

    let mut group = c.benchmark_group("CUDA ConvTranspose2d (stride 2)");
    for &(n, c_in, h, w, c_out, k) in CASES {
        let input = fill(n * c_in * h * w, 1.0);
        let weight = fill(c_in * c_out * k * k, 3.0);
        let in_cpu = Tensor::<f32, SequentialBackend>::from_slice([n, c_in, h, w], &input);
        let w_cpu = Tensor::<f32, SequentialBackend>::from_slice([c_in, c_out, k, k], &weight);
        let in_g = in_cpu.to_backend_on(&seq, &cuda);
        let w_g = w_cpu.to_backend_on(&seq, &cuda);
        let id = format!("{n}x{c_in}x{h}x{w}_k{k}");

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(conv_transpose2d(
                    black_box(&in_cpu),
                    black_box(&w_cpu),
                    None,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    &seq,
                ))
                .expect("CPU transposed convolution benchmark dispatch");
            });
        });
        group.bench_with_input(BenchmarkId::new("Coeus CUDA", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(conv_transpose2d(
                    black_box(&in_g),
                    black_box(&w_g),
                    None,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    &cuda,
                ))
                .expect("CUDA transposed convolution benchmark dispatch");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_conv_transpose2d);
criterion_main!(benches);
