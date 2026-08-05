use coeus_core::{ComputeBackend, Layout, ReductionOp};
use coeus_ops::ReductionOps;
use coeus_rocm::RocmBackend;

fn require_device() {
    if hephaestus_rocm::RocmDevice::try_default().is_err() {
        assert_ne!(
            std::env::var("HEPHAESTUS_ROCM_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "ROCm CI requires an acquired device"
        );
    }
}

#[test]
fn native_reductions_and_scans_match_leto() {
    require_device();
    if hephaestus_rocm::RocmDevice::try_default().is_err() {
        return;
    }

    let backend = RocmBackend::new();
    assert_eq!(backend.name(), "rocm");
    let layout = Layout::new([2, 3].into());
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut device_input = backend.allocate::<f32>(input.len());
    backend.copy_to_device(&input, &mut device_input);

    for (op, expected) in [
        (ReductionOp::Sum, [6.0_f32, 15.0]),
        (ReductionOp::Prod, [6.0_f32, 120.0]),
        (ReductionOp::Mean, [2.0_f32, 5.0]),
        (ReductionOp::Min, [1.0_f32, 4.0]),
        (ReductionOp::Max, [3.0_f32, 6.0]),
    ] {
        let mut expected_values = [0.0_f32; 2];
        coeus_leto::reduce_into(
            op,
            &layout,
            &input,
            1,
            &Layout::new([2, 1].into()),
            &mut expected_values,
        )
        .expect("Leto reduction oracle failed");
        assert_eq!(expected_values, expected, "Leto oracle contract");

        let mut actual = backend.allocate::<f32>(2);
        ReductionOps::reduce(
            &backend,
            op,
            &device_input,
            &layout,
            1,
            &mut actual,
            &Layout::new([2, 1].into()),
        )
        .expect("ROCm reduction failed");
        let mut actual_values = [0.0_f32; 2];
        backend.copy_to_host(&actual, &mut actual_values);
        assert_eq!(actual_values, expected_values, "ROCm {op:?} parity");
    }

    let mut expected_scan = [0.0_f32; 6];
    coeus_leto::cumsum_into(&layout, &input, 1, &layout, &mut expected_scan)
        .expect("Leto cumulative-sum oracle failed");
    let mut scan = backend.allocate::<f32>(input.len());
    ReductionOps::cumsum(&backend, &device_input, &layout, 1, &mut scan, &layout)
        .expect("ROCm cumulative sum failed");
    let mut actual_scan = [0.0_f32; 6];
    backend.copy_to_host(&scan, &mut actual_scan);
    assert_eq!(actual_scan, expected_scan);

    let mut expected_suffix_sum = [0.0_f32; 6];
    coeus_leto::suffix_sum_into(&layout, &input, 1, &layout, &mut expected_suffix_sum)
        .expect("Leto suffix-sum oracle failed");
    let mut suffix_sum = backend.allocate::<f32>(input.len());
    ReductionOps::suffix_sum(
        &backend,
        &device_input,
        &layout,
        1,
        &mut suffix_sum,
        &layout,
    )
    .expect("ROCm suffix sum failed");
    let mut actual_suffix_sum = [0.0_f32; 6];
    backend.copy_to_host(&suffix_sum, &mut actual_suffix_sum);
    assert_eq!(actual_suffix_sum, expected_suffix_sum);

    let mut expected_product_scan = [0.0_f32; 6];
    coeus_leto::cumprod_into(&layout, &input, 1, &layout, &mut expected_product_scan)
        .expect("Leto cumulative-product oracle failed");
    let mut product_scan = backend.allocate::<f32>(input.len());
    ReductionOps::cumprod(
        &backend,
        &device_input,
        &layout,
        1,
        &mut product_scan,
        &layout,
    )
    .expect("ROCm cumulative product failed");
    let mut actual_product_scan = [0.0_f32; 6];
    backend.copy_to_host(&product_scan, &mut actual_product_scan);
    assert_eq!(actual_product_scan, expected_product_scan);

    let mut expected_suffix_product = [0.0_f32; 6];
    coeus_leto::suffix_prod_into(&layout, &input, 1, &layout, &mut expected_suffix_product)
        .expect("Leto suffix-product oracle failed");
    let mut suffix_product = backend.allocate::<f32>(input.len());
    ReductionOps::suffix_prod(
        &backend,
        &device_input,
        &layout,
        1,
        &mut suffix_product,
        &layout,
    )
    .expect("ROCm suffix product failed");
    let mut actual_suffix_product = [0.0_f32; 6];
    backend.copy_to_host(&suffix_product, &mut actual_suffix_product);
    assert_eq!(actual_suffix_product, expected_suffix_product);
}

#[test]
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn norm_p_dispatches_with_rocm_provider_parity() {
    require_device();
    if hephaestus_rocm::RocmDevice::try_default().is_err() {
        return;
    }

    let backend = RocmBackend::new();
    let input = coeus_tensor::Tensor::<f32, RocmBackend>::from_slice_on(
        vec![2, 3],
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        &backend,
    );
    let actual = coeus_ops::norm_p(&input, 2.0, &backend);
    let expected = 91.0_f32.sqrt();
    assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);

    let actual_axis = coeus_ops::norm_p_axis(&input, 2.0, 1, &backend);
    let mut actual_axis_values = [0.0_f32; 2];
    backend.copy_to_host(actual_axis.storage(), &mut actual_axis_values);
    for (&actual, &expected) in actual_axis_values
        .iter()
        .zip([14.0_f32.sqrt(), 77.0_f32.sqrt()])
    {
        assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);
    }
}
