mod bce_with_logits;
mod convolution_pooling;
mod cross_entropy;
mod elementwise;
mod matmul;
mod optimizer;
mod random_init;
mod reduction;
mod rotate_half;
mod strided;

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

const WGPU_TOL: f32 = 1e-4;

fn seq() -> SequentialBackend {
    SequentialBackend::new()
}

fn wgpu() -> WgpuBackend {
    WgpuBackend::new()
}

/// Transfer a CPU tensor to the WgpuBackend.
fn to_gpu(t: &Tensor<f32, SequentialBackend>) -> Tensor<f32, WgpuBackend> {
    t.to_backend_on(&seq(), &wgpu())
}

/// Transfer a WgpuBackend tensor back to CPU.
fn to_cpu(t: &Tensor<f32, WgpuBackend>) -> Tensor<f32, SequentialBackend> {
    t.to_backend_on(&wgpu(), &seq())
}

fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        assert!(
            diff < WGPU_TOL,
            "{label}[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e}"
        );
    }
}
