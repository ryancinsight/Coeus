#[cfg(feature = "cuda")]
mod backend;
#[cfg(not(feature = "cuda"))]
#[path = "backend_stub.rs"]
mod backend;

#[cfg(feature = "cuda")]
pub mod driver;
#[cfg(not(feature = "cuda"))]
#[path = "driver_stub.rs"]
pub mod driver;

mod fallback;
#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
mod storage;
#[cfg(not(feature = "cuda"))]
#[path = "storage_stub.rs"]
mod storage;

pub use backend::{CudaBackend, CudaScalar};
pub use driver::{get_cuda_context, CudaDriver};
pub use storage::CudaStorage;

#[cfg(feature = "cuda")]
use coeus_core::Layout;
use coeus_tensor::Tensor;

/// Evaluate a fused element-wise expression on the CUDA device.
///
/// Compiles and dispatches a dynamic kernel on the GPU, falling back to CPU if unavailable.
pub fn evaluate_fused<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
) -> Tensor<T, CudaBackend> {
    #[cfg(not(feature = "cuda"))]
    {
        coeus_ops::fuse::evaluate_fused_cpu(expr, &CudaBackend::new())
    }

    #[cfg(feature = "cuda")]
    {
        let out_shape = expr
            .shape()
            .expect("Fused expression must have at least one tensor input to determine shape");
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        if kernels::dispatch_fused(expr, out.storage_mut(), &out_layout) {
            out
        } else {
            coeus_ops::fuse::evaluate_fused_cpu(expr, &CudaBackend::new())
        }
    }
}

/// Evaluate a fused reduction along an axis on the CUDA device.
pub fn evaluate_fused_reduce<T: CudaScalar, E: coeus_ops::fuse::ExprNode<T, CudaBackend> + Copy>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Tensor<T, CudaBackend> {
    #[cfg(not(feature = "cuda"))]
    {
        coeus_ops::fuse::evaluate_fused_reduce_cpu(expr, op, axis, &CudaBackend::new())
    }

    #[cfg(feature = "cuda")]
    {
        let expr_shape = expr
            .shape()
            .expect("Fused expression must have at least one tensor input to determine shape");
        assert!(
            axis < expr_shape.len(),
            "Axis out of bounds in evaluate_fused_reduce"
        );

        let mut out_shape = expr_shape;
        out_shape[axis] = 1;
        let out_layout = Layout::new(out_shape.clone());
        let mut out = Tensor::zeros_on(out_shape, &CudaBackend::new());

        if kernels::dispatch_fused_reduce(expr, op, axis, out.storage_mut(), &out_layout) {
            out
        } else {
            coeus_ops::fuse::evaluate_fused_reduce_cpu(expr, op, axis, &CudaBackend::new())
        }
    }
}
