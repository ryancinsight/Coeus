//! Differential verification for the CPU conv1d Hermes dot fast path.
//!
//! Contiguous, unpadded, unit-dilation conv1d windows are row-contiguous dot
//! products. CPU `BackendOps::conv1d` routes those reductions through
//! `Scalar::dot_slice` (`hermes_simd::dot` for native floats), while padded
//! cases keep the general layout-indexed path. The reference below is an
//! independent scalar convolution over the public shape contract.

use coeus_core::{CpuAddressableStorageMut, Layout, MoiraiBackend, SequentialBackend, Shape};
use coeus_ops::{BackendOps, CpuBackend};

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[derive(Clone, Copy)]
struct Conv1dCase<'a> {
    input: &'a [f32],
    input_shape: [usize; 3],
    weight: &'a [f32],
    weight_shape: [usize; 3],
    bias: &'a [f32],
    stride: usize,
    padding: usize,
    dilation: usize,
    output_shape: [usize; 3],
}

fn conv1d_reference(case: &Conv1dCase<'_>) -> Vec<f32> {
    let [n, c_in, l] = case.input_shape;
    let [c_out, weight_c_in, k] = case.weight_shape;
    let [out_n, out_c, l_out] = case.output_shape;
    assert_eq!([out_n, out_c], [n, c_out]);
    assert_eq!(weight_c_in, c_in);

    let mut out = vec![0.0; n * c_out * l_out];
    for ni in 0..n {
        for oc in 0..c_out {
            for ol in 0..l_out {
                let mut acc = case.bias[oc];
                for ic in 0..c_in {
                    for ik in 0..k {
                        let l_in = ol as isize * case.stride as isize
                            + ik as isize * case.dilation as isize
                            - case.padding as isize;
                        if l_in >= 0 && (l_in as usize) < l {
                            let input_index = (ni * c_in + ic) * l + l_in as usize;
                            let weight_index = (oc * c_in + ic) * k + ik;
                            acc += case.input[input_index] * case.weight[weight_index];
                        }
                    }
                }
                out[(ni * c_out + oc) * l_out + ol] = acc;
            }
        }
    }
    out
}

fn device_conv1d<B>(backend: &B, case: &Conv1dCase<'_>) -> Vec<f32>
where
    B: CpuBackend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    let input_layout = layout(&case.input_shape);
    let weight_layout = layout(&case.weight_shape);
    let output_layout = layout(&case.output_shape);

    let mut input = backend.allocate::<f32>(case.input.len()).expect("backend fixture operation");
    let mut weight = backend.allocate::<f32>(case.weight.len()).expect("backend fixture operation");
    let mut bias = backend.allocate::<f32>(case.bias.len()).expect("backend fixture operation");
    let mut output = backend.allocate::<f32>(output_layout.numel()).expect("backend fixture operation");

    backend.copy_to_device(case.input, &mut input).expect("backend fixture operation");
    backend.copy_to_device(case.weight, &mut weight).expect("backend fixture operation");
    backend.copy_to_device(case.bias, &mut bias).expect("backend fixture operation");

    backend.conv1d(
        &input,
        &input_layout,
        &weight,
        &weight_layout,
        Some(&bias),
        case.stride,
        case.padding,
        case.dilation,
        &mut output,
        &output_layout,
    ).expect("run convolution");

    let mut out = vec![0.0; output_layout.numel()];
    backend.copy_to_host(&output, &mut out).expect("backend fixture operation");
    out
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // The Hermes path may reassociate each kernel-row dot product. The
        // tested cases use at most six products per output, so this bound
        // covers row-dot reassociation plus cross-channel accumulation.
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
    let fast_case = Conv1dCase {
        input: &[
            0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0, 0.5, -0.75, 1.25, -1.5,
        ],
        input_shape: [1, 2, 6],
        weight: &[
            0.5, -0.25, 0.75, -1.0, 0.125, 0.375, -0.5, 0.875, -0.625, 0.25, 1.125, -0.75,
        ],
        weight_shape: [2, 2, 3],
        bias: &[0.125, -0.375],
        stride: 2,
        padding: 0,
        dilation: 1,
        output_shape: [1, 2, 2],
    };
    let fallback_case = Conv1dCase {
        stride: 1,
        padding: 1,
        dilation: 2,
        output_shape: [1, 2, 4],
        ..fast_case
    };

    let fast_expected = conv1d_reference(&fast_case);
    let fast_actual = device_conv1d(backend, &fast_case);
    assert_close("contiguous", &fast_actual, &fast_expected);

    let fallback_expected = conv1d_reference(&fallback_case);
    let fallback_actual = device_conv1d(backend, &fallback_case);
    assert_close("fallback", &fallback_actual, &fallback_expected);
}

#[test]
fn sequential_conv1d_matches_reference() {
    check_backend(&SequentialBackend);
}

#[test]
fn moirai_conv1d_matches_reference() {
    check_backend(&MoiraiBackend);
}
