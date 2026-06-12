//! Differential verification for the CPU conv2d backward Hermes dot path.
//!
//! Contiguous, unpadded, unit-stride/unit-dilation conv2d weight-gradient
//! width rows are row-contiguous dot products. CPU `BackendOps::conv2d_backward`
//! routes those reductions through `Scalar::dot_slice` (`hermes_simd::dot` for
//! native floats), while padded or dilated cases keep the general layout-indexed
//! path. The reference below is an independent scalar implementation over the
//! public shape contract.

use coeus_core::{CpuAddressableStorageMut, Layout, MoiraiBackend, SequentialBackend, Shape};
use coeus_ops::{BackendOps, CpuBackend};

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[derive(Clone, Copy)]
struct Conv2dBackwardCase<'a> {
    grad_out: &'a [f32],
    grad_out_shape: [usize; 4],
    input: &'a [f32],
    input_shape: [usize; 4],
    weight: &'a [f32],
    weight_shape: [usize; 4],
    initial_grad_weight: &'a [f32],
    stride: usize,
    padding: usize,
    dilation: usize,
}

fn grad_weight_reference(case: &Conv2dBackwardCase<'_>) -> Vec<f32> {
    let [n, c_in, h, w] = case.input_shape;
    let [c_out, weight_c_in, kh, kw] = case.weight_shape;
    let [out_n, out_c, h_out, w_out] = case.grad_out_shape;
    assert_eq!([out_n, out_c], [n, c_out]);
    assert_eq!(weight_c_in, c_in);

    let mut grad_weight = case.initial_grad_weight.to_vec();
    for oc in 0..c_out {
        for ic in 0..c_in {
            for ikh in 0..kh {
                for ikw in 0..kw {
                    let mut acc = 0.0;
                    for ni in 0..n {
                        for oh in 0..h_out {
                            let h_in = oh as isize * case.stride as isize
                                + ikh as isize * case.dilation as isize
                                - case.padding as isize;
                            if h_in >= 0 && (h_in as usize) < h {
                                for ow in 0..w_out {
                                    let w_in = ow as isize * case.stride as isize
                                        + ikw as isize * case.dilation as isize
                                        - case.padding as isize;
                                    if w_in >= 0 && (w_in as usize) < w {
                                        let go_index =
                                            ((ni * c_out + oc) * h_out + oh) * w_out + ow;
                                        let input_index = ((ni * c_in + ic) * h + h_in as usize)
                                            * w
                                            + w_in as usize;
                                        acc += case.grad_out[go_index] * case.input[input_index];
                                    }
                                }
                            }
                        }
                    }
                    grad_weight[((oc * c_in + ic) * kh + ikh) * kw + ikw] += acc;
                }
            }
        }
    }
    grad_weight
}

fn device_grad_weight<B>(backend: &B, case: &Conv2dBackwardCase<'_>) -> Vec<f32>
where
    B: CpuBackend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    let grad_out_layout = layout(&case.grad_out_shape);
    let input_layout = layout(&case.input_shape);
    let weight_layout = layout(&case.weight_shape);

    let mut grad_out = backend.allocate::<f32>(case.grad_out.len());
    let mut input = backend.allocate::<f32>(case.input.len());
    let mut weight = backend.allocate::<f32>(case.weight.len());
    let mut grad_weight = backend.allocate::<f32>(case.initial_grad_weight.len());

    backend.copy_to_device(case.grad_out, &mut grad_out);
    backend.copy_to_device(case.input, &mut input);
    backend.copy_to_device(case.weight, &mut weight);
    backend.copy_to_device(case.initial_grad_weight, &mut grad_weight);

    backend.conv2d_backward(
        &grad_out,
        &grad_out_layout,
        &input,
        &input_layout,
        &weight,
        &weight_layout,
        None,
        &input_layout,
        Some(&mut grad_weight),
        &weight_layout,
        None,
        case.stride,
        case.padding,
        case.dilation,
    );

    let mut out = vec![0.0; case.initial_grad_weight.len()];
    backend.copy_to_host(&grad_weight, &mut out);
    out
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // The Hermes path may reassociate per-output-row dot products. The
        // tested contiguous case uses at most three products per row, three
        // output rows, and two batches, so this bound covers row-dot
        // reassociation plus row/batch accumulation.
        let tol = 128.0 * f32::EPSILON * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{label}[{index}]: got {got}, expected {want}, tol {tol}",
        );
    }
}

fn check_backend<B>(backend: &B)
where
    B: CpuBackend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    let input: Vec<f32> = (0..80).map(|i| ((i as f32 % 11.0) - 5.0) * 0.125).collect();
    let weight: Vec<f32> = (0..24).map(|i| ((i as f32 % 7.0) - 3.0) * 0.1875).collect();
    let initial_grad_weight: Vec<f32> = (0..24).map(|i| ((i as f32 % 5.0) - 2.0) * 0.01).collect();
    let fast_grad_out: Vec<f32> = (0..36)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.0625)
        .collect();
    let fallback_grad_out: Vec<f32> = (0..48)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect();

    let fast_case = Conv2dBackwardCase {
        grad_out: &fast_grad_out,
        grad_out_shape: [2, 2, 3, 3],
        input: &input,
        input_shape: [2, 2, 4, 5],
        weight: &weight,
        weight_shape: [2, 2, 2, 3],
        initial_grad_weight: &initial_grad_weight,
        stride: 1,
        padding: 0,
        dilation: 1,
    };
    let fallback_case = Conv2dBackwardCase {
        grad_out: &fallback_grad_out,
        grad_out_shape: [2, 2, 4, 3],
        padding: 1,
        dilation: 2,
        ..fast_case
    };

    let fast_expected = grad_weight_reference(&fast_case);
    let fast_actual = device_grad_weight(backend, &fast_case);
    assert_close("contiguous", &fast_actual, &fast_expected);

    let fallback_expected = grad_weight_reference(&fallback_case);
    let fallback_actual = device_grad_weight(backend, &fallback_case);
    assert_close("fallback", &fallback_actual, &fallback_expected);
}

#[test]
fn sequential_conv2d_backward_grad_weight_matches_reference() {
    check_backend(&SequentialBackend);
}

#[test]
fn moirai_conv2d_backward_grad_weight_matches_reference() {
    check_backend(&MoiraiBackend);
}
