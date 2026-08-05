// CUDA vs CPU parity tests share one backend oracle and are partitioned by operation family.

pub(super) use coeus_autograd::Var;
pub(super) use coeus_core::SequentialBackend;
pub(super) use coeus_cuda::CudaBackend;
pub(super) use coeus_ops::{ConvOps, OptimizerOps, PoolOps};
pub(super) use coeus_tensor::Tensor;

#[path = "parity/bce_with_logits.rs"]
mod bce_with_logits;
#[path = "parity/convolution.rs"]
mod convolution;
#[path = "parity/convolution_transpose.rs"]
mod convolution_transpose;
#[path = "parity/cross_entropy.rs"]
mod cross_entropy;
#[path = "parity/matmul.rs"]
mod matmul;
#[path = "parity/optimizer.rs"]
mod optimizer;
#[path = "parity/pooling.rs"]
mod pooling;
#[path = "parity/reduction.rs"]
mod reduction;
#[path = "parity/unfold_fold.rs"]
mod unfold_fold;

/// Element-wise tolerance for direct (non-accumulating) ops.
pub(super) const CUDA_TOL: f32 = 1e-4;
/// Tolerance for accumulating ops whose GPU reduction order differs from CPU.
pub(super) const CUDA_ACC_TOL: f32 = 1e-3;

pub(super) fn backends() -> Option<(SequentialBackend, CudaBackend)> {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return None;
    }
    let cuda_b = CudaBackend::new();
    if coeus_cuda::CudaDriver::get().is_none() || coeus_cuda::get_cuda_context().is_none() {
        return None;
    }
    Some((SequentialBackend::new(), cuda_b))
}

pub(super) fn to_gpu(
    t: &Tensor<f32, SequentialBackend>,
    s: &SequentialBackend,
    c: &CudaBackend,
) -> Tensor<f32, CudaBackend> {
    t.to_backend_on(s, c)
}

pub(super) fn to_cpu(
    t: &Tensor<f32, CudaBackend>,
    c: &CudaBackend,
    s: &SequentialBackend,
) -> Tensor<f32, SequentialBackend> {
    t.to_backend_on(c, s)
}

pub(super) fn assert_parity_tol(label: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "{label}[{i}]: expected NaN, got {g}");
            continue;
        }
        if c.is_infinite() {
            assert!(
                g.is_infinite() && g.is_sign_positive() == c.is_sign_positive(),
                "{label}[{i}]: cpu={c} gpu={g}"
            );
            continue;
        }
        let diff = (c - g).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: cpu={c:.6} gpu={g:.6} diff={diff:.2e} tol={tol:.0e}"
        );
    }
}
