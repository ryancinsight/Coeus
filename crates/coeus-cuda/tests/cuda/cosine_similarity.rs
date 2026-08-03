use coeus_autograd::{cosine_similarity, Var};
use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

fn backends() -> Option<(SequentialBackend, CudaBackend)> {
    if hephaestus_cuda::CudaDevice::try_default().is_err()
        || coeus_cuda::CudaDriver::get().is_none()
        || coeus_cuda::get_cuda_context().is_none()
    {
        return None;
    }
    Some((SequentialBackend::new(), CudaBackend::new()))
}

#[test]
fn cosine_similarity_dispatches_with_cuda_parity() {
    let Some((cpu, cuda)) = backends() else {
        return;
    };
    let x1_cpu = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[0.0, 0.0, 2.0, 1.0]);
    let x2_cpu = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 0.0, 1.0, 0.0]);
    let x1_cuda = x1_cpu.to_backend_on(&cpu, &cuda);
    let x2_cuda = x2_cpu.to_backend_on(&cpu, &cuda);

    let x1_cpu = Var::new(x1_cpu, true);
    let x2_cpu = Var::new(x2_cpu, true);
    let x1_cuda = Var::new(x1_cuda, true);
    let x2_cuda = Var::new(x2_cuda, true);
    let cpu_output = cosine_similarity(&x1_cpu, &x2_cpu, 1, 0.5);
    let cuda_output = cosine_similarity(&x1_cuda, &x2_cuda, 1, 0.5);

    cpu_output
        .backward()
        .expect("CPU cosine backward must succeed");
    cuda_output
        .backward()
        .expect("CUDA cosine backward must succeed");

    let cuda_output = cuda_output.tensor.to_backend_on(&cuda, &cpu);
    let cuda_x1_gradient = x1_cuda
        .grad()
        .expect("tracked CUDA x1 gradient")
        .to_backend_on(&cuda, &cpu);
    let cuda_x2_gradient = x2_cuda
        .grad()
        .expect("tracked CUDA x2 gradient")
        .to_backend_on(&cuda, &cpu);
    let cpu_x1_gradient = x1_cpu.grad().expect("tracked CPU x1 gradient");
    let cpu_x2_gradient = x2_cpu.grad().expect("tracked CPU x2 gradient");

    for (operation, expected, actual) in [
        (
            "forward",
            cpu_output.tensor.as_slice(),
            cuda_output.as_slice(),
        ),
        (
            "x1 gradient",
            cpu_x1_gradient.as_slice(),
            cuda_x1_gradient.as_slice(),
        ),
        (
            "x2 gradient",
            cpu_x2_gradient.as_slice(),
            cuda_x2_gradient.as_slice(),
        ),
    ] {
        assert_eq!(expected.len(), actual.len());
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            assert!(
                (expected - actual).abs() <= 8.0 * f32::EPSILON,
                "{operation}[{index}]: expected {expected}, got {actual}"
            );
        }
    }
}
