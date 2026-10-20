// On-device (WGPU) vs CPU (SequentialBackend) baselines for the core compute
// kernels: matmul and transposed convolution. Complements `attention_bench`.
//
// `tensor_bench` (coeus-tensor) covers matmul on the CPU backends and Leto;
// this adds the WGPU on-device angle so the device crossover (where the GPU
// overtakes the CPU as size grows) is visible and a profiling baseline exists
// for future kernel tiling per the profile-first discipline.
//
// Machine class: depends on the local GPU/driver and CPU; record the host when
// committing a baseline. Not run in headless CI (no GPU) — an on-demand tool.

use coeus_core::{ComputeBackend, CpuAddressableStorageMut, FloatElement, SequentialBackend};
use coeus_ops::conv_transpose2d;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn fill(n: usize, phase: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + phase) * 0.013).sin()).collect()
}

fn bench_matmul(c: &mut Criterion) {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let mut group = c.benchmark_group("Matmul (square)");
    for &sz in &[128usize, 512] {
        let a = fill(sz * sz, 1.0);
        let b = fill(sz * sz, 2.0);
        let a_cpu = Tensor::<f32, SequentialBackend>::from_slice([sz, sz], &a)
            .expect("invariant: test backend operation succeeds");
        let b_cpu = Tensor::<f32, SequentialBackend>::from_slice([sz, sz], &b)
            .expect("invariant: test backend operation succeeds");
        let a_g = a_cpu
            .to_backend_on(&seq, &wgpu)
            .expect("invariant: test backend operation succeeds");
        let b_g = b_cpu
            .to_backend_on(&seq, &wgpu)
            .expect("invariant: test backend operation succeeds");
        let id = format!("{sz}x{sz}");

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    coeus_ops::matmul(black_box(&a_cpu), black_box(&b_cpu), &seq)
                        .expect("invariant: test operation succeeds"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("Coeus WGPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    coeus_ops::matmul(black_box(&a_g), black_box(&b_g), &wgpu)
                        .expect("invariant: test operation succeeds"),
                )
            });
        });
    }
    group.finish();
}

fn bench_conv_transpose2d(c: &mut Criterion) {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    // (n, c_in, h, w, c_out, k) with stride 2 (upsampling, the common use).
    const CASES: &[(usize, usize, usize, usize, usize, usize)] =
        &[(4, 8, 16, 16, 8, 3), (4, 16, 32, 32, 16, 3)];
    let (stride, padding, output_padding, dilation) = (2usize, 1usize, 1usize, 1usize);

    let mut group = c.benchmark_group("ConvTranspose2d (stride 2)");
    for &(n, c_in, h, w, c_out, k) in CASES {
        let input = fill(n * c_in * h * w, 1.0);
        let weight = fill(c_in * c_out * k * k, 3.0);
        let in_cpu = Tensor::<f32, SequentialBackend>::from_slice([n, c_in, h, w], &input)
            .expect("invariant: test backend operation succeeds");
        let w_cpu = Tensor::<f32, SequentialBackend>::from_slice([c_in, c_out, k, k], &weight)
            .expect("invariant: test backend operation succeeds");
        let in_g = in_cpu
            .to_backend_on(&seq, &wgpu)
            .expect("invariant: test backend operation succeeds");
        let w_g = w_cpu
            .to_backend_on(&seq, &wgpu)
            .expect("invariant: test backend operation succeeds");
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
        group.bench_with_input(BenchmarkId::new("Coeus WGPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(conv_transpose2d(
                    black_box(&in_g),
                    black_box(&w_g),
                    None,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    &wgpu,
                ))
                .expect("WGPU transposed convolution benchmark dispatch");
            });
        });
    }
    group.finish();
}

/// Scalar-sum placement crossover: on-device GPU reduction (ending with the
/// scalar readback `sum` always performs) against host staging (download the
/// full array once, then reduce on `SequentialBackend`). Both arms end with a
/// host scalar, so the comparison is end to end. Resident chains whose result
/// feeds another GPU op are out of scope: pipelined dispatch has no host
/// round trip to beat.
fn bench_reduction_crossover(c: &mut Criterion) {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let mut group = c.benchmark_group("Reduction crossover (scalar sum f32)");
    for &n in &[256usize, 1024, 4096, 16384, 65536, 262144, 1048576] {
        let values = fill(n, 1.0);
        let a_g = Tensor::<f32, SequentialBackend>::from_slice([n], &values)
            .expect("invariant: test backend operation succeeds")
            .to_backend_on(&seq, &wgpu)
            .expect("invariant: test backend operation succeeds");
        let mut staged = Tensor::<f32, SequentialBackend>::zeros([n]).expect("host staging tensor");
        // Correctness guard: both placements must agree before timing.
        let gpu = coeus_ops::sum(&a_g, &wgpu).expect("gpu sum succeeds");
        {
            let dst = staged
                .storage_mut()
                .expect("staging storage")
                .as_mut_slice()
                .expect("host slice");
            wgpu.copy_to_host(a_g.storage(), dst)
                .expect("staging download");
        }
        let cpu = coeus_ops::sum(&staged, &seq).expect("cpu sum succeeds");
        assert!(
            (gpu - cpu).abs() < 1e-4 * f32::from_count(n),
            "placement mismatch at n={n}: gpu={gpu} cpu={cpu}"
        );

        let id = format!("n{n}");

        group.bench_with_input(BenchmarkId::new("GPU resident", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    coeus_ops::sum(black_box(&a_g), &wgpu)
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
                wgpu.copy_to_host(black_box(a_g.storage()), black_box(dst))
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
