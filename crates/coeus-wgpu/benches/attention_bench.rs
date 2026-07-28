// On-device (WGPU) vs CPU (SequentialBackend) scaled dot-product attention.
//
// Establishes the baseline for the on-device attention kernels (forward +
// backward) so any future tiling/flash-attention optimization can be measured
// against it per the profile-first discipline. The WGPU path runs the WGSL
// kernels directly on-device; the CPU path runs the verified sequential
// reference. Each group reports both so the device speedup (and the H2D/D2H
// elimination the on-device kernels deliver) is visible.
//
// Machine class: results depend on the local GPU/driver and CPU; record the
// host when committing a baseline (`cargo bench -p coeus-wgpu`). The bench is
// not run in headless CI (no GPU); it is an on-demand profiling instrument.

use coeus_core::SequentialBackend;
use coeus_ops::{scaled_dot_product_attention, scaled_dot_product_attention_backward};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// (batch*heads, seq_q, seq_k, d_k, d_v) — heads folded into batch, matching
/// the attention contract. Two representative transformer-ish shapes.
const SHAPES: &[(usize, usize, usize, usize, usize)] =
    &[(8, 64, 64, 32, 32), (16, 128, 128, 64, 64)];

fn fill(n: usize, phase: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + phase) * 0.017).sin()).collect()
}

fn bench_attention_forward(c: &mut Criterion) {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let scale = 0.125f32;

    let mut group = c.benchmark_group("SDP Attention Forward");
    for &(b, sq, sk, dk, dv) in SHAPES {
        let id = format!("{b}x{sq}x{sk}x{dk}x{dv}");
        let q_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sq, dk], &fill(b * sq * dk, 1.0))
                .expect("construct tensor");
        let k_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sk, dk], &fill(b * sk * dk, 3.0))
                .expect("construct tensor");
        let v_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sk, dv], &fill(b * sk * dv, 5.0))
                .expect("construct tensor");
        let q_g = q_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");
        let k_g = k_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");
        let v_g = v_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(scaled_dot_product_attention(
                    black_box(&q_cpu),
                    black_box(&k_cpu),
                    black_box(&v_cpu),
                    None,
                    false,
                    scale,
                    black_box(&seq),
                ).expect("evaluate attention"));
            })
        });

        group.bench_with_input(BenchmarkId::new("Coeus WGPU", &id), &id, |bn, _| {
            bn.iter(|| {
                black_box(scaled_dot_product_attention(
                    black_box(&q_g),
                    black_box(&k_g),
                    black_box(&v_g),
                    None,
                    false,
                    scale,
                    black_box(&wgpu),
                ).expect("evaluate attention"));
            })
        });
    }
    group.finish();
}

fn bench_attention_backward(c: &mut Criterion) {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let scale = 0.125f32;

    let mut group = c.benchmark_group("SDP Attention Backward");
    for &(b, sq, sk, dk, dv) in SHAPES {
        let id = format!("{b}x{sq}x{sk}x{dk}x{dv}");
        let q_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sq, dk], &fill(b * sq * dk, 1.0))
                .expect("construct tensor");
        let k_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sk, dk], &fill(b * sk * dk, 3.0))
                .expect("construct tensor");
        let v_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sk, dv], &fill(b * sk * dv, 5.0))
                .expect("construct tensor");
        let go_cpu =
            Tensor::<f32, SequentialBackend>::from_slice([b, sq, dv], &fill(b * sq * dv, 7.0))
                .expect("construct tensor");

        // Stored attention weights from the forward pass feed the backward.
        let (_, aw_cpu) =
            scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, None, false, scale, &seq)
                .expect("evaluate attention");

        let q_g = q_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");
        let k_g = k_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");
        let v_g = v_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");
        let go_g = go_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");
        let aw_g = aw_cpu.to_backend_on(&seq, &wgpu).expect("transfer tensor");

        group.bench_with_input(BenchmarkId::new("Coeus CPU", &id), &id, |bn, _| {
            bn.iter(|| {
                let mut gq = Tensor::<f32, SequentialBackend>::zeros_on([b, sq, dk], &seq)
                    .expect("construct tensor");
                let mut gk = Tensor::<f32, SequentialBackend>::zeros_on([b, sk, dk], &seq)
                    .expect("construct tensor");
                let mut gv = Tensor::<f32, SequentialBackend>::zeros_on([b, sk, dv], &seq)
                    .expect("construct tensor");
                scaled_dot_product_attention_backward(
                    black_box(&go_cpu),
                    black_box(&q_cpu),
                    black_box(&k_cpu),
                    black_box(&v_cpu),
                    black_box(&aw_cpu),
                    scale,
                    Some(&mut gq),
                    Some(&mut gk),
                    Some(&mut gv),
                    black_box(&seq),
                ).expect("evaluate attention backward");
                black_box((gq, gk, gv));
            })
        });

        group.bench_with_input(BenchmarkId::new("Coeus WGPU", &id), &id, |bn, _| {
            bn.iter(|| {
                let mut gq = Tensor::<f32, WgpuBackend>::zeros_on([b, sq, dk], &wgpu)
                    .expect("construct tensor");
                let mut gk = Tensor::<f32, WgpuBackend>::zeros_on([b, sk, dk], &wgpu)
                    .expect("construct tensor");
                let mut gv = Tensor::<f32, WgpuBackend>::zeros_on([b, sk, dv], &wgpu)
                    .expect("construct tensor");
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
                    black_box(&wgpu),
                ).expect("evaluate attention backward");
                black_box((gq, gk, gv));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_attention_forward, bench_attention_backward);
criterion_main!(benches);
