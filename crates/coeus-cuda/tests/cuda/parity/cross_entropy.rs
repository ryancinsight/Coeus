use super::{assert_parity_tol, backends, to_cpu, to_gpu, Tensor, Var, CUDA_TOL};
use coeus_cuda::CudaBackendError;

#[test]
fn cross_entropy_dispatches_with_cuda_value_and_gradient_parity() {
    let Some((cpu, cuda)) = backends() else {
        return;
    };
    let logits = Tensor::from_slice_on(
        [3, 4],
        &[
            1.5_f32, 0.5, -0.5, 0.25, -1.0, 2.0, 0.0, 0.75, 3.0, -2.0, 1.0, 0.0,
        ],
        &cpu,
    );
    let cpu_logits = Var::new(logits.clone(), true);
    let cuda_logits = Var::new(to_gpu(&logits, &cpu, &cuda), true);
    let targets = [0_usize, 1, 2];

    let invalid_target = coeus_nn::cross_entropy_loss(&cuda_logits, &[0, 1, 4]);
    assert!(matches!(
        invalid_target,
        Err(CudaBackendError::Validation {
            source: coeus_core::BackendError::IndexOutOfRange {
                position: 2,
                index: 4,
                bound: 4,
                ..
            }
        })
    ));

    let cpu_loss = coeus_nn::cross_entropy_loss(&cpu_logits, &targets)
        .expect("CPU cross-entropy dispatch must succeed");
    let cuda_loss = coeus_nn::cross_entropy_loss(&cuda_logits, &targets)
        .expect("CUDA cross-entropy dispatch must succeed");
    cpu_loss
        .backward()
        .expect("CPU cross-entropy backward must succeed");
    cuda_loss
        .backward()
        .expect("CUDA cross-entropy backward must succeed");

    let cuda_loss = to_cpu(&cuda_loss.tensor, &cuda, &cpu);
    let cpu_gradient = cpu_logits.grad().expect("tracked CPU logits gradient");
    let cuda_gradient = to_cpu(
        &cuda_logits.grad().expect("tracked CUDA logits gradient"),
        &cuda,
        &cpu,
    );
    assert_parity_tol(
        "cross-entropy loss",
        cpu_loss.tensor.as_slice(),
        cuda_loss.as_slice(),
        CUDA_TOL,
    );
    assert_parity_tol(
        "cross-entropy logits gradient",
        cpu_gradient.as_slice(),
        cuda_gradient.as_slice(),
        CUDA_TOL,
    );
}
