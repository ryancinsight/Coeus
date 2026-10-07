use coeus_core::{ComputeBackend, CpuAddressableStorage, Layout, Scalar};
use coeus_ops::RandomInitOps;

use super::{seq, wgpu};

fn assert_seeded_parity<T: Scalar + PartialEq + core::fmt::Debug>(
    operation: &str,
    cpu: impl AsRef<[T]>,
    gpu: &<coeus_wgpu::WgpuBackend as ComputeBackend>::DeviceBuffer<T>,
) {
    let expected = cpu.as_ref();
    let mut actual = expected.to_vec();
    wgpu().copy_to_host(gpu, &mut actual);
    assert_eq!(
        actual.as_slice(),
        expected,
        "{operation} must preserve seeded values"
    );
}

#[test]
fn random_initialization_dispatches_with_seeded_provider_parity() {
    let Err(invalid) =
        wgpu().uniform_random(&Layout::new(Vec::<usize>::new().into()), -0.75, 1.25, 17)
    else {
        panic!("rank-zero dispatch must fail before device acquisition");
    };
    assert!(matches!(
        invalid,
        coeus_wgpu::WgpuBackendError::Dispatch {
            operation: "uniform initialization",
            source: hephaestus_core::HephaestusError::InvalidConfiguration { .. },
        }
    ));

    if !crate::availability::device_available("coeus-wgpu-random-init-test") {
        return;
    }
    let layout = Layout::new([2, 3].into());
    let cpu = seq();
    let gpu = wgpu();

    let cpu_uniform = cpu
        .uniform_random(&layout, -0.75, 1.25, 17)
        .expect("CPU uniform provider must initialize a valid layout");
    let gpu_uniform = gpu
        .uniform_random(&layout, -0.75, 1.25, 17)
        .expect("WGPU uniform provider must initialize a valid layout");
    assert_seeded_parity("uniform", cpu_uniform.as_slice(), &gpu_uniform);

    let cpu_normal = cpu
        .normal_random(&layout, 0.25, 0.5, 29)
        .expect("CPU normal provider must initialize a valid layout");
    let gpu_normal = gpu
        .normal_random(&layout, 0.25, 0.5, 29)
        .expect("WGPU normal provider must initialize a valid layout");
    assert_seeded_parity("normal", cpu_normal.as_slice(), &gpu_normal);
}

#[test]
fn random_initialization_dispatches_with_seeded_provider_parity_f64() {
    // Plain device gate: generation is host-delegated through Leto, so no
    // f64 shader compiles and SHADER_F64 is not required.
    if !crate::availability::device_available("coeus-wgpu-random-init-test-f64") {
        return;
    }
    let layout = Layout::new([2, 3].into());
    let cpu = seq();
    let gpu = wgpu();

    let cpu_uniform = cpu
        .uniform_random(&layout, -0.75_f64, 1.25, 17)
        .expect("CPU uniform provider must initialize a valid layout");
    let gpu_uniform = gpu
        .uniform_random(&layout, -0.75_f64, 1.25, 17)
        .expect("WGPU uniform provider must initialize a valid layout");
    assert_seeded_parity("uniform", cpu_uniform.as_slice(), &gpu_uniform);

    let cpu_normal = cpu
        .normal_random(&layout, 0.25_f64, 0.5, 29)
        .expect("CPU normal provider must initialize a valid layout");
    let gpu_normal = gpu
        .normal_random(&layout, 0.25_f64, 0.5, 29)
        .expect("WGPU normal provider must initialize a valid layout");
    assert_seeded_parity("normal", cpu_normal.as_slice(), &gpu_normal);
}
