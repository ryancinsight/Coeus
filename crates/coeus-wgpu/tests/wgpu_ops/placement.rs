use super::availability::device_available;
use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

fn staged_scalar_fns_agree(n: usize) {
    assert!(n < WgpuBackend::new().host_staging_crossover());
    let seq = SequentialBackend::new();
    let backend = WgpuBackend::new();
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 2.0).collect();
    let host = Tensor::<f32, SequentialBackend>::from_slice([n], &values).unwrap();
    let device = host.to_backend_on(&seq, &backend).unwrap();

    // Staged device scalars run the same host kernel as the CPU reference:
    // bitwise equality, not tolerance.
    assert_eq!(
        coeus_ops::sum(&device, &backend).unwrap(),
        coeus_ops::sum(&host, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::mean(&device, &backend).unwrap(),
        coeus_ops::mean(&host, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::amax(&device, &backend).unwrap(),
        coeus_ops::amax(&host, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::amin(&device, &backend).unwrap(),
        coeus_ops::amin(&host, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::prod(&device, &backend).unwrap(),
        coeus_ops::prod(&host, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::norm(&device, &backend).unwrap(),
        coeus_ops::norm(&host, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::norm_p(&device, 3.0, &backend).unwrap(),
        coeus_ops::norm_p(&host, 3.0, &seq).unwrap()
    );
    assert_eq!(
        coeus_ops::var_mean(&device, true, &backend).unwrap(),
        coeus_ops::var_mean(&host, true, &seq).unwrap()
    );
}

#[test]
fn crossover_is_measured_constant() {
    assert_eq!(WgpuBackend::new().host_staging_crossover(), 65_536);
    assert_eq!(SequentialBackend::new().host_staging_crossover(), 0);
}

#[test]
fn staged_scalars_match_host_below_crossover() {
    if !device_available("placement:staged") {
        return;
    }
    staged_scalar_fns_agree(1);
    staged_scalar_fns_agree(1000);
    staged_scalar_fns_agree(65_535);
}

#[test]
fn device_scalars_match_host_above_crossover() {
    if !device_available("placement:resident") {
        return;
    }
    // Above the crossover the device kernel runs; rounding may differ from
    // the host kernel, so compare with tolerance instead of bitwise equality.
    let seq = SequentialBackend::new();
    let backend = WgpuBackend::new();
    let n = 100_000usize;
    assert!(n >= backend.host_staging_crossover());
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 2.0).collect();
    let host = Tensor::<f32, SequentialBackend>::from_slice([n], &values).unwrap();
    let device = host.to_backend_on(&seq, &backend).unwrap();
    let gpu = coeus_ops::sum(&device, &backend).unwrap();
    let cpu = coeus_ops::sum(&host, &seq).unwrap();
    assert!(
        (gpu - cpu).abs() < 1e-3 * n as f32,
        "resident sum mismatch: gpu={gpu} cpu={cpu}"
    );
}
