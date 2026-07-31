#![cfg(not(feature = "cuda"))]

use coeus_core::ComputeBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::TensorExprExt;
use coeus_tensor::Tensor;

#[test]
fn disabled_provider_reports_unavailable_backend_identity() {
    let backend = CudaBackend::new();
    assert_eq!(backend.name(), "cuda-unavailable");
}

#[test]
fn disabled_provider_rejects_fused_execution() {
    let backend = CudaBackend::new();
    let input = Tensor::<f32, CudaBackend>::from_slice_on([2], &[1.0, -2.0], &backend);
    let expression = input.expr().relu();

    let error = match coeus_cuda::evaluate_fused(&expression) {
        Ok(_) => panic!("disabled CUDA provider executed a fused expression"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        coeus_cuda::CudaBackendError::Kernel {
            operation: "fused elementwise",
            reason: "the CUDA provider feature is disabled",
        }
    ));
}

#[test]
fn disabled_provider_rejects_fused_reduction() {
    let backend = CudaBackend::new();
    let input = Tensor::<f32, CudaBackend>::from_slice_on([2], &[1.0, -2.0], &backend);
    let expression = input.expr().relu();

    let error = match coeus_cuda::evaluate_fused_reduce(&expression, coeus_ops::ReductionOp::Sum, 0)
    {
        Ok(_) => panic!("disabled CUDA provider executed a fused reduction"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        coeus_cuda::CudaBackendError::Kernel {
            operation: "fused reduction",
            reason: "the CUDA provider feature is disabled",
        }
    ));
}
