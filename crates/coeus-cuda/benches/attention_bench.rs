// On-device (CUDA) vs CPU (SequentialBackend) scaled dot-product attention —
// the CUDA counterpart to coeus-wgpu's attention_bench, completing the
// cross-backend attention profiling matrix.
//
// Meaningful only with a live CUDA device AND `--features cuda`:
//   cargo bench -p coeus-cuda --features cuda --bench attention_bench
// Without a CUDA context the groups register nothing (the guard returns early),
// so a plain run produces no misleading fallback numbers.
//
// Machine class: depends on the local GPU/driver and CPU; record the host when
// committing a baseline. Not run in headless CI.

use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::{scaled_dot_product_attention, scaled_dot_product_attention_backward};
use coeus_tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn cuda_available() -> bool {
    hephaestus_cuda::CudaDevice::try_default().is_ok()
}

/// (batch*heads, seq_q, seq_k, d_k, d_v) — heads folded into batch.
const SHAPES: &[(usize, usize, usize, usize, usize)] =
    &[(8, 64, 64, 32, 32), (16, 128, 128, 64, 64)];

fn fill(n: usize, phase: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + phase) * 0.017).sin()).collect()
}

fn bench_attention_forward(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let scale = 0.125f32;

    let mut group = c.benchmark_group("CUDA SDP Attention Forward");
    for &(b, sq, sk, dk, dv) in SHAPES {
        let id = format!("{b}x{sq}x{sk}x{dk}x{dv}");
        let q = Tensor::<f32, SequentialBackend>::from_slice([b, sq, dk], &fill(b * sq * dk, 1.0));
        let k = Tensor::<f32, SequentialBackend>::from_slice([b, sk, dk], &fill(b * sk * dk, 3.0));
        let v = Tensor::<f32, SequentialBackend>::from_slice([b, sk, dv], &fill(b * sk * dv, 5.0));
        let q_g = q.to_backend_on(&seq, &cuda);
        let k_g = k.to_backend_on(&seq, &cuda);
        let v_g = v.to_backend_on(&seq, &cuda);

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    scaled_dot_product_attention(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        None,
                        false,
                        scale,
                        &seq,
                    )
                    .expect("CPU attention forward must succeed"),
                );
            })
        });
        group.bench_with_input(BenchmarkId::new("Coeus CUDA", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(
                    scaled_dot_product_attention(
                        black_box(&q_g),
                        black_box(&k_g),
                        black_box(&v_g),
                        None,
                        false,
                        scale,
                        &cuda,
                    )
                    .expect("CUDA attention forward must succeed"),
                );
            })
        });
    }
    group.finish();
}

fn bench_attention_backward(c: &mut Criterion) {
    if !cuda_available() {
        return;
    }
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let scale = 0.125f32;

    let mut group = c.benchmark_group("CUDA SDP Attention Backward");
    for &(b, sq, sk, dk, dv) in SHAPES {
        let id = format!("{b}x{sq}x{sk}x{dk}x{dv}");
        let q = Tensor::<f32, SequentialBackend>::from_slice([b, sq, dk], &fill(b * sq * dk, 1.0));
        let k = Tensor::<f32, SequentialBackend>::from_slice([b, sk, dk], &fill(b * sk * dk, 3.0));
        let v = Tensor::<f32, SequentialBackend>::from_slice([b, sk, dv], &fill(b * sk * dv, 5.0));
        let go = Tensor::<f32, SequentialBackend>::from_slice([b, sq, dv], &fill(b * sq * dv, 7.0));
        let (_, aw) = scaled_dot_product_attention(&q, &k, &v, None, false, scale, &seq)
            .expect("CPU attention setup must succeed");

        let q_g = q.to_backend_on(&seq, &cuda);
        let k_g = k.to_backend_on(&seq, &cuda);
        let v_g = v.to_backend_on(&seq, &cuda);
        let go_g = go.to_backend_on(&seq, &cuda);
        let aw_g = aw.to_backend_on(&seq, &cuda);

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                let mut gq = Tensor::<f32, SequentialBackend>::zeros_on([b, sq, dk], &seq);
                let mut gk = Tensor::<f32, SequentialBackend>::zeros_on([b, sk, dk], &seq);
                let mut gv = Tensor::<f32, SequentialBackend>::zeros_on([b, sk, dv], &seq);
                scaled_dot_product_attention_backward(
                    black_box(&go),
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(&aw),
                    scale,
                    Some(&mut gq),
                    Some(&mut gk),
                    Some(&mut gv),
                    &seq,
                )
                .expect("CPU attention backward must succeed");
                black_box((gq, gk, gv));
            })
        });
        group.bench_with_input(BenchmarkId::new("Coeus CUDA", &id), &id, |bn, _| {
            bn.iter(|| {
                let mut gq = Tensor::<f32, CudaBackend>::zeros_on([b, sq, dk], &cuda);
                let mut gk = Tensor::<f32, CudaBackend>::zeros_on([b, sk, dk], &cuda);
                let mut gv = Tensor::<f32, CudaBackend>::zeros_on([b, sk, dv], &cuda);
                scaled_dot_product_attention_backward(
                    black_box(&go_g),
                    black_box(&q_g),
                    black_box(&k_g),
                    black_box(&v_g),
                    black_box(&aw_g),
                    scale,
                    Some(&mut gq),
                    Some(&mut gk),
                    Some(&mut gv),
                    &cuda,
                )
                .expect("CUDA attention backward must succeed");
                black_box((gq, gk, gv));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_attention_forward, bench_attention_backward);
criterion_main!(benches);
