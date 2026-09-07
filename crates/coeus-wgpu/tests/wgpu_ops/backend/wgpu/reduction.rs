use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn cumulative_scans_match_cpu_on_rank_two_device_tensors() {
    if !crate::availability::device_available("coeus-wgpu-scan-test") {
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

    let expected_prefix_product = coeus_ops::cumprod(&input, 1, &sequential);
    let actual_prefix_product = coeus_ops::cumprod(&device_input, 1, &wgpu);
    let actual_prefix_product = actual_prefix_product.to_backend_on(&wgpu, &sequential);
    assert_eq!(
        actual_prefix_product.as_slice(),
        expected_prefix_product.as_slice()
    );

    let expected_suffix_product = coeus_ops::suffix_prod(&input, 0, &sequential);
    let actual_suffix_product = coeus_ops::suffix_prod(&device_input, 0, &wgpu);
    let actual_suffix_product = actual_suffix_product.to_backend_on(&wgpu, &sequential);
    assert_eq!(
        actual_suffix_product.as_slice(),
        expected_suffix_product.as_slice()
    );
}

#[test]
fn product_axis_matches_cpu_on_rank_two_device_tensors() {
    if !crate::availability::device_available("coeus-wgpu-product-axis-test") {
        return;
    }

    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, -2.0, 3.0, 4.0, 0.5, 6.0]);
    let device_input = input.to_backend_on(&sequential, &wgpu);

    let expected = coeus_ops::prod_axis(&input, 1, &sequential).expect("valid CPU product axis");
    let actual = coeus_ops::prod_axis(&device_input, 1, &wgpu)
        .expect("valid WGPU product axis")
        .to_backend_on(&wgpu, &sequential);

    assert_eq!(actual.shape(), &[2, 1]);
    assert_eq!(actual.as_slice(), expected.as_slice());
}

#[test]
fn norm_p_dispatches_with_wgpu_provider_parity() {
    if !crate::availability::device_available("coeus-wgpu-norm-p-test") {
        return;
    }

    let sequential = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let input =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    let device_input = input.to_backend_on(&sequential, &wgpu);

    let expected = coeus_ops::norm_p(&input, 2.0, &sequential);
    let actual = coeus_ops::norm_p(&device_input, 2.0, &wgpu);
    assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);

    let expected_axis = coeus_ops::norm_p_axis(&input, 2.0, 1, &sequential);
    let actual_axis =
        coeus_ops::norm_p_axis(&device_input, 2.0, 1, &wgpu).to_backend_on(&wgpu, &sequential);
    assert_eq!(actual_axis.shape(), &[2, 1]);
    for (&actual, &expected) in actual_axis.as_slice().iter().zip(expected_axis.as_slice()) {
        assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);
    }
}
