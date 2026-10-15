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

use coeus_core::{ComputeBackend, CpuAddressableStorageMut, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_ops::conv_transpose2d;
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn cuda_available() -> bool {
    hephaestus_cuda::CudaDevice::try_default().is_ok()
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
        let a_cpu = Tensor::<f32, SequentialBackend>::from_slice([sz, sz], &a)
            .expect("invariant: test backend operation succeeds");
        let b_cpu = Tensor::<f32, SequentialBackend>::from_slice([sz, sz], &b)
            .expect("invariant: test backend operation succeeds");
        let a_g = a_cpu
            .to_backend_on(&seq, &cuda)
            .expect("invariant: test backend transfer succeeds");
        let b_g = b_cpu
            .to_backend_on(&seq, &cuda)
            .expect("invariant: test backend transfer succeeds");
        let id = format!("{sz}x{sz}");

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    coeus_ops::matmul(black_box(&a_cpu), black_box(&b_cpu), &seq)
                        .expect("invariant: test operation succeeds"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("Coeus CUDA", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    coeus_ops::matmul(black_box(&a_g), black_box(&b_g), &cuda)
                        .expect("invariant: test operation succeeds"),
                )
            });
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
        let in_cpu = Tensor::<f32, SequentialBackend>::from_slice([n, c_in, h, w], &input)
            .expect("invariant: test backend operation succeeds");
        let w_cpu = Tensor::<f32, SequentialBackend>::from_slice([c_in, c_out, k, k], &weight)
            .expect("invariant: test backend operation succeeds");
        let in_g = in_cpu
            .to_backend_on(&seq, &cuda)
            .expect("invariant: test backend transfer succeeds");
        let w_g = w_cpu
            .to_backend_on(&seq, &cuda)
            .expect("invariant: test backend transfer succeeds");
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

/// Scalar-sum placement crossover, CUDA counterpart to the WGPU group: the
/// CUDA submit floor differs from WDDM, so the crossover is measured
/// separately rather than inherited.
///
/// The GPU arm routes through `host_staging_crossover`, so re-measuring the
/// threshold itself requires forcing the constant to zero first; otherwise
/// sizes below the current threshold benchmark the staged path twice.
fn bench_reduction_crossover(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let mut group = c.benchmark_group("Reduction crossover (scalar sum f32)");
    for &n in &[256usize, 1024, 4096, 16384, 65536, 262144, 1048576] {
        let values = fill(n, 1.0);
        let a_g = Tensor::<f32, SequentialBackend>::from_slice([n], &values)
            .expect("invariant: test backend operation succeeds")
            .to_backend_on(&seq, &cuda)
            .expect("invariant: test backend operation succeeds");
        let mut staged = Tensor::<f32, SequentialBackend>::zeros([n]).expect("host staging tensor");
        let gpu = coeus_ops::sum(&a_g, &cuda).expect("gpu sum succeeds");
        {
            let dst = staged
                .storage_mut()
                .expect("staging storage")
                .as_mut_slice()
                .expect("host slice");
            cuda.copy_to_host(a_g.storage(), dst)
                .expect("staging download");
        }
        let cpu = coeus_ops::sum(&staged, &seq).expect("cpu sum succeeds");
        assert!(
            (gpu - cpu).abs() < 1e-4 * n as f32,
            "placement mismatch at n={n}: gpu={gpu} cpu={cpu}"
        );
        let id = format!("n{n}");

        group.bench_with_input(BenchmarkId::new("GPU resident", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    coeus_ops::sum(black_box(&a_g), &cuda)
                        .expect("GPU scalar sum benchmark dispatch"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("Host staged", &id), &id, |bn, _| {
            bn.iter(|| {
                let dst = staged
                    .storage_mut()
                    .expect("staging storage")
                    .as_mut_slice()
                    .expect("host slice");
                cuda.copy_to_host(black_box(a_g.storage()), black_box(dst))
                    .expect("staging download");
                black_box(
                    coeus_ops::sum(black_box(&staged), &seq)
                        .expect("host staged sum benchmark dispatch"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_conv_transpose2d,
    bench_reduction_crossover
);
criterion_main!(benches);
