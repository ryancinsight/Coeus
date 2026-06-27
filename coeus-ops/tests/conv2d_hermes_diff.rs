//! Differential verification for the CPU conv2d Hermes AXPY fast path.
//!
//! Contiguous, unpadded, unit-dilation, unit-stride conv2d rows are accumulated
//! as output-stationary AXPY operations. CPU `BackendOps::conv2d` routes those
//! row accumulations through `Scalar::axpy_slice` (`hermes_simd::axpy` for
//! native floats), while strided, padded, or dilated cases keep scalar or
//! layout-indexed paths.

use coeus_core::{CpuAddressableStorageMut, Layout, MoiraiBackend, SequentialBackend, Shape};
use coeus_ops::{BackendOps, CpuBackend};

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[derive(Clone, Copy)]
struct Conv2dCase<'a> {
    input: &'a [f32],
    input_shape: [usize; 4],
    weight: &'a [f32],
    weight_shape: [usize; 4],
    bias: &'a [f32],
    stride: usize,
    padding: usize,
    dilation: usize,
    output_shape: [usize; 4],
}

fn conv2d_reference(case: &Conv2dCase<'_>) -> Vec<f32> {
    let [n, c_in, h, w] = case.input_shape;
    let [c_out, weight_c_in, kh, kw] = case.weight_shape;
    let [out_n, out_c, h_out, w_out] = case.output_shape;
    assert_eq!([out_n, out_c], [n, c_out]);
    assert_eq!(weight_c_in, c_in);

    let mut out = vec![0.0; n * c_out * h_out * w_out];
    for ni in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = case.bias[oc];
                    for ic in 0..c_in {
                        for ikh in 0..kh {
                            let h_in = oh as isize * case.stride as isize
                                + ikh as isize * case.dilation as isize
                                - case.padding as isize;
                            if h_in >= 0 && (h_in as usize) < h {
                                for ikw in 0..kw {
                                    let w_in = ow as isize * case.stride as isize
                                        + ikw as isize * case.dilation as isize
                                        - case.padding as isize;
                                    if w_in >= 0 && (w_in as usize) < w {
                                        let input_index = ((ni * c_in + ic) * h + h_in as usize)
                                            * w
                                            + w_in as usize;
                                        let weight_index = ((oc * c_in + ic) * kh + ikh) * kw + ikw;
                                        acc += case.input[input_index] * case.weight[weight_index];
                                    }
                                }
                            }
                        }
                    }
                    out[((ni * c_out + oc) * h_out + oh) * w_out + ow] = acc;
                }
            }
        }
    }
    out
}

fn device_conv2d<B>(backend: &B, case: &Conv2dCase<'_>) -> Vec<f32>
where
    B: CpuBackend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    let input_layout = layout(&case.input_shape);
    let weight_layout = layout(&case.weight_shape);
    let output_layout = layout(&case.output_shape);

    let mut input = backend.allocate::<f32>(case.input.len());
    let mut weight = backend.allocate::<f32>(case.weight.len());
    let mut bias = backend.allocate::<f32>(case.bias.len());
    let mut output = backend.allocate::<f32>(output_layout.numel());

    backend.copy_to_device(case.input, &mut input);
    backend.copy_to_device(case.weight, &mut weight);
    backend.copy_to_device(case.bias, &mut bias);

    backend.conv2d(
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
    );

    let mut out = vec![0.0; output_layout.numel()];
    backend.copy_to_host(&output, &mut out);
    out
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // The Hermes AXPY path may use fused multiply-add and SIMD lane order.
        // The tested cases use at most twelve products per output, so this bound
        // covers one rounding difference per product plus channel/row
        // accumulation.
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
    let input: Vec<f32> = (0..32).map(|i| ((i as f32 % 9.0) - 4.0) * 0.125).collect();
    let weight: Vec<f32> = (0..24).map(|i| ((i as f32 % 7.0) - 3.0) * 0.1875).collect();
    let bias = [0.125, -0.25, 0.375];

    let axpy_case = Conv2dCase {
        input: &input,
        input_shape: [1, 2, 4, 4],
        weight: &weight,
        weight_shape: [3, 2, 2, 2],
        bias: &bias,
        stride: 1,
        padding: 0,
        dilation: 1,
        output_shape: [1, 3, 3, 3],
    };
    let strided_case = Conv2dCase {
        stride: 2,
        output_shape: [1, 3, 2, 2],
        ..axpy_case
    };
    let fallback_case = Conv2dCase {
        stride: 1,
        padding: 1,
        dilation: 2,
        output_shape: [1, 3, 4, 4],
        ..axpy_case
    };

    let axpy_expected = conv2d_reference(&axpy_case);
    let axpy_actual = device_conv2d(backend, &axpy_case);
    assert_close("contiguous_axpy", &axpy_actual, &axpy_expected);

    let strided_expected = conv2d_reference(&strided_case);
    let strided_actual = device_conv2d(backend, &strided_case);
    assert_close("contiguous_strided", &strided_actual, &strided_expected);

    let fallback_expected = conv2d_reference(&fallback_case);
    let fallback_actual = device_conv2d(backend, &fallback_case);
    assert_close("fallback", &fallback_actual, &fallback_expected);
}

#[test]
fn sequential_conv2d_matches_reference() {
    check_backend(&SequentialBackend);
}

#[test]
fn moirai_conv2d_matches_reference() {
    check_backend(&MoiraiBackend);
}
