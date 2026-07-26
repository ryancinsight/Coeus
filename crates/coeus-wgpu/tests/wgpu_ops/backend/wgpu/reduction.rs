use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn cumulative_scans_match_cpu_on_rank_two_device_tensors() {
    if hephaestus_wgpu::WgpuDevice::try_default("coeus-wgpu-scan-test").is_err() {
        return;
    }

    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let device_input = input.to_backend_on(&sequential, &wgpu);

    let expected_prefix = coeus_ops::cumsum(&input, 1);
    let actual_prefix = coeus_ops::cumsum(&device_input, 1);
    let actual_prefix = actual_prefix.to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_prefix.as_slice(), expected_prefix.as_slice());

    let expected_suffix = coeus_ops::suffix_sum(&input, 0);
    let actual_suffix = coeus_ops::suffix_sum(&device_input, 0);
    let actual_suffix = actual_suffix.to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_suffix.as_slice(), expected_suffix.as_slice());
}
