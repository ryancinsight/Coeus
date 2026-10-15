use super::availability::device_available;
use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

fn staged_scalar_fns_agree(n: usize) {
    assert!(n < CudaBackend::new().host_staging_crossover());
    let seq = SequentialBackend::new();
    let backend = CudaBackend::new();
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 2.0).collect();
    let host = Tensor::<f32, SequentialBackend>::from_slice([n], &values)
        .expect("invariant: test backend operation succeeds");
    let device = host
        .to_backend_on(&seq, &backend)
        .expect("invariant: test backend operation succeeds");

    // Staged device scalars run the same host kernel as the CPU reference:
    // bitwise equality, not tolerance.
    assert_eq!(
        coeus_ops::sum(&device, &backend).expect("invariant: test backend operation succeeds"),
        coeus_ops::sum(&host, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::mean(&device, &backend).expect("invariant: test backend operation succeeds"),
        coeus_ops::mean(&host, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::amax(&device, &backend).expect("invariant: test backend operation succeeds"),
        coeus_ops::amax(&host, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::amin(&device, &backend).expect("invariant: test backend operation succeeds"),
        coeus_ops::amin(&host, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::prod(&device, &backend).expect("invariant: test backend operation succeeds"),
        coeus_ops::prod(&host, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::norm(&device, &backend).expect("invariant: test backend operation succeeds"),
        coeus_ops::norm(&host, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::norm_p(&device, 3.0, &backend)
            .expect("invariant: test backend operation succeeds"),
        coeus_ops::norm_p(&host, 3.0, &seq).expect("invariant: test backend operation succeeds")
    );
    assert_eq!(
        coeus_ops::var_mean(&device, true, &backend)
            .expect("invariant: test backend operation succeeds"),
        coeus_ops::var_mean(&host, true, &seq).expect("invariant: test backend operation succeeds")
    );
}

#[test]
fn crossover_is_measured_constant() {
    assert_eq!(CudaBackend::new().host_staging_crossover(), 131_072);
}

#[test]
fn staged_scalars_match_host_below_crossover() {
    if !device_available() {
        return;
    }
    staged_scalar_fns_agree(1);
    staged_scalar_fns_agree(1000);
    staged_scalar_fns_agree(65_535);
}

#[test]
fn device_scalars_match_host_above_crossover() {
    if !device_available() {
        return;
    }
    // Above the crossover the device kernel runs; rounding may differ from
    // the host kernel, so compare with tolerance instead of bitwise equality.
    let seq = SequentialBackend::new();
    let backend = CudaBackend::new();
    let n = 200_000usize;
    assert!(n >= backend.host_staging_crossover());
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 2.0).collect();
    let host = Tensor::<f32, SequentialBackend>::from_slice([n], &values)
        .expect("invariant: test backend operation succeeds");
    let device = host
        .to_backend_on(&seq, &backend)
        .expect("invariant: test backend operation succeeds");
    let gpu =
        coeus_ops::sum(&device, &backend).expect("invariant: test backend operation succeeds");
    let cpu = coeus_ops::sum(&host, &seq).expect("invariant: test backend operation succeeds");
    assert!(
        (gpu - cpu).abs() < 1e-3 * n as f32,
        "resident sum mismatch: gpu={gpu} cpu={cpu}"
    );
}
