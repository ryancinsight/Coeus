use coeus_core::{ComputeBackend, CpuAddressableStorage, Layout, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_ops::RandomInitOps;

fn backends() -> Option<(SequentialBackend, CudaBackend)> {
    if hephaestus_cuda::CudaDevice::try_default().is_err()
        || coeus_cuda::CudaDriver::get().is_none()
        || coeus_cuda::get_cuda_context().is_none()
    {
        return None;
    }
    Some((SequentialBackend::new(), CudaBackend::new()))
}

fn assert_seeded_parity(
    operation: &str,
    cpu: &[f32],
    gpu: &<CudaBackend as ComputeBackend>::DeviceBuffer<f32>,
    backend: &CudaBackend,
) {
    let mut actual = vec![0.0; cpu.len()];
    backend.copy_to_host(gpu, &mut actual);
    assert_eq!(actual, cpu, "{operation} must preserve seeded values");
}

#[test]
fn random_initialization_dispatches_with_seeded_provider_parity() {
    let Err(invalid) = CudaBackend::new().uniform_random(
        &Layout::new(Vec::<usize>::new().into()),
        -0.75,
        1.25,
        17,
    ) else {
        panic!("rank-zero dispatch must fail before device acquisition");
    };
    assert!(matches!(
        invalid,
        coeus_cuda::CudaBackendError::Dispatch {
            operation: "uniform initialization",
            source: hephaestus_core::HephaestusError::InvalidConfiguration { .. },
        }
    ));

    let Some((cpu, cuda)) = backends() else {
        return;
    };
    let layout = Layout::new([2, 3].into());

    let cpu_uniform = cpu
        .uniform_random(&layout, -0.75, 1.25, 17)
        .expect("CPU uniform provider must initialize a valid layout");
    let cuda_uniform = cuda
        .uniform_random(&layout, -0.75, 1.25, 17)
        .expect("CUDA uniform provider must initialize a valid layout");
    assert_seeded_parity("uniform", cpu_uniform.as_slice(), &cuda_uniform, &cuda);

    let cpu_normal = cpu
        .normal_random(&layout, 0.25, 0.5, 29)
        .expect("CPU normal provider must initialize a valid layout");
    let cuda_normal = cuda
        .normal_random(&layout, 0.25, 0.5, 29)
        .expect("CUDA normal provider must initialize a valid layout");
    assert_seeded_parity("normal", cpu_normal.as_slice(), &cuda_normal, &cuda);
}
