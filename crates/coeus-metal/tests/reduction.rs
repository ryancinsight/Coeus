use coeus_core::{ComputeBackend, Layout, ReductionOp};
use coeus_hephaestus::HephaestusBackend;
use coeus_metal::MetalProvider;
use coeus_ops::ReductionOps;

type Backend = HephaestusBackend<MetalProvider>;

// Axis 1 partitions this row-major matrix into [1, 2, 3] and [4, 5, 6].
// Each reduction and inclusive scan below follows its finite sum/product
// definition. All intermediate integers are below 2^24, and the row means
// are integers, so every expected value is exactly representable in f32.
const INPUT: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const REDUCTIONS: [(ReductionOp, [f32; 2]); 5] = [
    (ReductionOp::Sum, [6.0, 15.0]),
    (ReductionOp::Prod, [6.0, 120.0]),
    (ReductionOp::Mean, [2.0, 5.0]),
    (ReductionOp::Min, [1.0, 4.0]),
    (ReductionOp::Max, [3.0, 6.0]),
];
const PREFIX_SUM: [f32; 6] = [1.0, 3.0, 6.0, 4.0, 9.0, 15.0];
const SUFFIX_SUM: [f32; 6] = [6.0, 5.0, 3.0, 15.0, 11.0, 6.0];
const PREFIX_PRODUCT: [f32; 6] = [1.0, 2.0, 6.0, 4.0, 20.0, 120.0];
const SUFFIX_PRODUCT: [f32; 6] = [6.0, 6.0, 3.0, 120.0, 30.0, 6.0];

#[test]
fn leto_reductions_and_scans_match_axis_values() {
    let layout = Layout::new([2, 3].into());
    let output_layout = Layout::new([2, 1].into());
    for (op, expected) in REDUCTIONS {
        let mut actual = [0.0; 2];
        match op {
            ReductionOp::Mean => {
                coeus_leto::reduce_mean_into(&layout, &INPUT, 1, &output_layout, &mut actual)
            }
            ReductionOp::Prod => {
                coeus_leto::reduce_prod_into(&layout, &INPUT, 1, &output_layout, &mut actual)
            }
            ReductionOp::Sum | ReductionOp::Min | ReductionOp::Max => {
                coeus_leto::reduce_into(op, &layout, &INPUT, 1, &output_layout, &mut actual)
            }
        }
        .expect("Leto reduction failed");
        assert_eq!(actual, expected, "Leto {op:?} axis values");
    }

    let mut actual = [0.0; 6];
    coeus_leto::cumsum_into(&layout, &INPUT, 1, &layout, &mut actual)
        .expect("Leto cumulative sum failed");
    assert_eq!(actual, PREFIX_SUM, "Leto inclusive prefix sum");
    coeus_leto::suffix_sum_into(&layout, &INPUT, 1, &layout, &mut actual)
        .expect("Leto suffix sum failed");
    assert_eq!(actual, SUFFIX_SUM, "Leto inclusive suffix sum");
    coeus_leto::cumprod_into(&layout, &INPUT, 1, &layout, &mut actual)
        .expect("Leto cumulative product failed");
    assert_eq!(actual, PREFIX_PRODUCT, "Leto inclusive prefix product");
    coeus_leto::suffix_prod_into(&layout, &INPUT, 1, &layout, &mut actual)
        .expect("Leto suffix product failed");
    assert_eq!(actual, SUFFIX_PRODUCT, "Leto inclusive suffix product");
}

fn require_device() {
    if hephaestus_metal::MetalDevice::try_default().is_err() {
        assert_ne!(
            std::env::var("HEPHAESTUS_METAL_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "Metal CI requires an acquired device"
        );
    }
}

#[test]
fn reductions_and_scans_match_axis_values() {
    require_device();
    if hephaestus_metal::MetalDevice::try_default().is_err() {
        return;
    }

    let backend = Backend::new();
    assert_eq!(backend.name(), "metal");
    let layout = Layout::new([2, 3].into());
    let mut device_input = backend.allocate::<f32>(INPUT.len());
    backend.copy_to_device(&INPUT, &mut device_input);

    for (op, expected) in REDUCTIONS {
        let mut actual = backend.allocate::<f32>(2);
        let output_layout = Layout::new([2, 1].into());
        match op {
            ReductionOp::Mean => ReductionOps::reduce_mean(
                &backend,
                &device_input,
                &layout,
                1,
                &mut actual,
                &output_layout,
            ),
            ReductionOp::Prod => ReductionOps::reduce_prod(
                &backend,
                &device_input,
                &layout,
                1,
                &mut actual,
                &output_layout,
            ),
            ReductionOp::Sum | ReductionOp::Min | ReductionOp::Max => ReductionOps::reduce(
                &backend,
                op,
                &device_input,
                &layout,
                1,
                &mut actual,
                &output_layout,
            ),
        }
        .expect("Metal reduction failed");
        let mut actual_values = [0.0_f32; 2];
        backend.copy_to_host(&actual, &mut actual_values);
        assert_eq!(actual_values, expected, "Metal {op:?} axis values");
    }

    let mut scan = backend.allocate::<f32>(INPUT.len());
    ReductionOps::cumsum(&backend, &device_input, &layout, 1, &mut scan, &layout)
        .expect("Metal cumulative sum failed");
    let mut actual_scan = [0.0_f32; 6];
    backend.copy_to_host(&scan, &mut actual_scan);
    assert_eq!(actual_scan, PREFIX_SUM, "Metal inclusive prefix sum");

    let mut suffix_sum = backend.allocate::<f32>(INPUT.len());
    ReductionOps::suffix_sum(
        &backend,
        &device_input,
        &layout,
        1,
        &mut suffix_sum,
        &layout,
    )
    .expect("Metal suffix sum failed");
    let mut actual_suffix_sum = [0.0_f32; 6];
    backend.copy_to_host(&suffix_sum, &mut actual_suffix_sum);
    assert_eq!(actual_suffix_sum, SUFFIX_SUM, "Metal inclusive suffix sum");

    let mut product_scan = backend.allocate::<f32>(INPUT.len());
    ReductionOps::cumprod(
        &backend,
        &device_input,
        &layout,
        1,
        &mut product_scan,
        &layout,
    )
    .expect("Metal cumulative product failed");
    let mut actual_product_scan = [0.0_f32; 6];
    backend.copy_to_host(&product_scan, &mut actual_product_scan);
    assert_eq!(
        actual_product_scan, PREFIX_PRODUCT,
        "Metal inclusive prefix product"
    );

    let mut suffix_product = backend.allocate::<f32>(INPUT.len());
    ReductionOps::suffix_prod(
        &backend,
        &device_input,
        &layout,
        1,
        &mut suffix_product,
        &layout,
    )
    .expect("Metal suffix product failed");
    let mut actual_suffix_product = [0.0_f32; 6];
    backend.copy_to_host(&suffix_product, &mut actual_suffix_product);
    assert_eq!(
        actual_suffix_product, SUFFIX_PRODUCT,
        "Metal inclusive suffix product"
    );
}

#[test]
fn norm_p_dispatches_with_metal_provider_parity() {
    require_device();
    if hephaestus_metal::MetalDevice::try_default().is_err() {
        return;
    }

    let backend = Backend::new();
    let input = coeus_tensor::Tensor::<f32, Backend>::from_slice_on(
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
    for (&actual, expected) in actual_axis_values
        .iter()
        .zip([14.0_f32.sqrt(), 77.0_f32.sqrt()])
    {
        assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);
    }
}
