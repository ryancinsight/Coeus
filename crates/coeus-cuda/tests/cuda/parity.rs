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
#[path = "parity/elementwise.rs"]
mod elementwise;
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
    if !crate::availability::device_available() {
        return None;
    }
    let cuda_b = CudaBackend::new();
    Some((SequentialBackend::new(), cuda_b))
}

pub(super) fn to_gpu<T: coeus_core::Scalar>(
    t: &Tensor<T, SequentialBackend>,
    s: &SequentialBackend,
    c: &CudaBackend,
) -> Tensor<T, CudaBackend> {
    t.to_backend_on(s, c)
}

pub(super) fn to_cpu<T: coeus_core::Scalar>(
    t: &Tensor<T, CudaBackend>,
    c: &CudaBackend,
    s: &SequentialBackend,
) -> Tensor<T, SequentialBackend> {
    t.to_backend_on(c, s)
}

/// Element-wise tolerance for f64 parity (absolute; values are O(1)).
pub(super) const CUDA_TOL_F64: f64 = 1e-12;

pub(super) fn assert_parity_tol_f64(label: &str, cpu: &[f64], gpu: &[f64], tol: f64) {
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
            "{label}[{i}]: cpu={c:.12} gpu={g:.12} diff={diff:.2e} tol={tol:.0e}"
        );
    }
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
