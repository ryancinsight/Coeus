use coeus_autograd::{rotate_half, sum, Var};
use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

#[test]
fn rotate_half_dispatches_with_cuda_parity() {
    let available = hephaestus_cuda::CudaDevice::try_default().is_ok()
        && coeus_cuda::CudaDriver::get().is_some()
        && coeus_cuda::get_cuda_context().is_some();
    if !available {
        assert_ne!(
            std::env::var("HEPHAESTUS_CUDA_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "CUDA CI requires an acquired device"
        );
        return;
    }
    let cpu = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let input = Tensor::from_slice_on([2, 4], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &cpu)
        .to_backend_on(&cpu, &cuda);
    let input = Var::new(input, true);
    let output = rotate_half(&input).expect("CUDA rotate-half dispatch");
    sum(&output).backward().expect("CUDA rotate-half backward");
    assert_eq!(
        output.tensor.to_backend_on(&cuda, &cpu).as_slice(),
        &[-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]
    );
    assert_eq!(
        input
            .grad()
            .expect("tracked CUDA input gradient")
            .to_backend_on(&cuda, &cpu)
            .as_slice(),
        &[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    );
}
