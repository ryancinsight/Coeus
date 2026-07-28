//! Differential verification for the CPU conv1d backward Hermes dot path.
//!
//! Contiguous, unpadded, unit-stride/unit-dilation conv1d weight-gradient
//! windows are row-contiguous dot products. CPU `BackendOps::conv1d_backward`
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
struct Conv1dBackwardCase<'a> {
    grad_out: &'a [f32],
    grad_out_shape: [usize; 3],
    input: &'a [f32],
    input_shape: [usize; 3],
    weight: &'a [f32],
    weight_shape: [usize; 3],
    initial_grad_weight: &'a [f32],
    stride: usize,
    padding: usize,
    dilation: usize,
}

fn grad_weight_reference(case: &Conv1dBackwardCase<'_>) -> Vec<f32> {
    let [n, c_in, l] = case.input_shape;
    let [c_out, weight_c_in, k] = case.weight_shape;
    let [out_n, out_c, l_out] = case.grad_out_shape;
    assert_eq!([out_n, out_c], [n, c_out]);
    assert_eq!(weight_c_in, c_in);

    let mut grad_weight = case.initial_grad_weight.to_vec();
    for oc in 0..c_out {
        for ic in 0..c_in {
            for ik in 0..k {
                let mut acc = 0.0;
                for ni in 0..n {
                    for ol in 0..l_out {
                        let l_in = ol as isize * case.stride as isize
                            + ik as isize * case.dilation as isize
                            - case.padding as isize;
                        if l_in >= 0 && (l_in as usize) < l {
                            let go_index = (ni * c_out + oc) * l_out + ol;
                            let input_index = (ni * c_in + ic) * l + l_in as usize;
                            acc += case.grad_out[go_index] * case.input[input_index];
                        }
                    }
                }
                grad_weight[(oc * c_in + ic) * k + ik] += acc;
            }
        }
    }
    grad_weight
}

fn device_grad_weight<B>(backend: &B, case: &Conv1dBackwardCase<'_>) -> Vec<f32>
where
    B: CpuBackend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    let grad_out_layout = layout(&case.grad_out_shape);
    let input_layout = layout(&case.input_shape);
    let weight_layout = layout(&case.weight_shape);

    let mut grad_out = backend.allocate::<f32>(case.grad_out.len()).expect("backend fixture operation");
    let mut input = backend.allocate::<f32>(case.input.len()).expect("backend fixture operation");
    let mut weight = backend.allocate::<f32>(case.weight.len()).expect("backend fixture operation");
    let mut grad_weight = backend.allocate::<f32>(case.initial_grad_weight.len()).expect("backend fixture operation");

    backend.copy_to_device(case.grad_out, &mut grad_out).expect("backend fixture operation");
    backend.copy_to_device(case.input, &mut input).expect("backend fixture operation");
    backend.copy_to_device(case.weight, &mut weight).expect("backend fixture operation");
    backend.copy_to_device(case.initial_grad_weight, &mut grad_weight).expect("backend fixture operation");

    backend.conv1d_backward(
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
    ).expect("run convolution backward");

    let mut out = vec![0.0; case.initial_grad_weight.len()];
    backend.copy_to_host(&grad_weight, &mut out).expect("backend fixture operation");
    out
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // The Hermes path may reassociate per-batch row-dot products. The
        // tested cases use at most four products per batch and two batches, so
        // this bound covers row-dot reassociation plus batch accumulation.
        let tol = 64.0 * f32::EPSILON * (1.0 + want.abs());
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
    let fast_case = Conv1dBackwardCase {
        grad_out: &[
            0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, -0.125, 0.375, -0.625, 0.875, -1.125,
            1.375, -1.625, 1.875,
        ],
        grad_out_shape: [2, 2, 4],
        input: &[
            0.5, -0.25, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25, -2.5, 2.75, -3.0, -0.5, 0.25,
            -0.75, 1.0, -1.25, 1.5, -1.75, 2.0, -2.25, 2.5, -2.75, 3.0,
        ],
        input_shape: [2, 2, 6],
        weight: &[
            0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0, 1.125, -1.25, 1.375, -1.5,
        ],
        weight_shape: [2, 2, 3],
        initial_grad_weight: &[
            0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12,
        ],
        stride: 1,
        padding: 0,
        dilation: 1,
    };
    let fallback_case = Conv1dBackwardCase {
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
fn sequential_conv1d_backward_grad_weight_matches_reference() {
    check_backend(&SequentialBackend);
}

#[test]
fn moirai_conv1d_backward_grad_weight_matches_reference() {
    check_backend(&MoiraiBackend);
}
