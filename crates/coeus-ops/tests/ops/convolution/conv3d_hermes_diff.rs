//! Differential verification for the CPU conv3d Hermes dot fast path.
//!
//! Contiguous, unpadded, unit-dilation conv3d kernel rows are row-contiguous dot
//! products. CPU `BackendOps::conv3d` routes those reductions through
//! `Scalar::dot_slice` (`hermes_simd::dot` for native floats), while padded or
//! dilated cases keep the general layout-indexed path.

use coeus_core::{CpuAddressableStorageMut, Layout, MoiraiBackend, SequentialBackend, Shape};
use coeus_ops::{BackendOps, CpuBackend};

fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}

#[derive(Clone, Copy)]
struct Conv3dCase<'a> {
    input: &'a [f32],
    input_shape: [usize; 5],
    weight: &'a [f32],
    weight_shape: [usize; 5],
    bias: &'a [f32],
    stride: usize,
    padding: usize,
    dilation: usize,
    output_shape: [usize; 5],
}

fn conv3d_reference(case: &Conv3dCase<'_>) -> Vec<f32> {
    let [n, c_in, d, h, w] = case.input_shape;
    let [c_out, weight_c_in, kd, kh, kw] = case.weight_shape;
    let [out_n, out_c, d_out, h_out, w_out] = case.output_shape;
    assert_eq!([out_n, out_c], [n, c_out]);
    assert_eq!(weight_c_in, c_in);

    let mut out = vec![0.0; n * c_out * d_out * h_out * w_out];
    for ni in 0..n {
        for oc in 0..c_out {
            for od in 0..d_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = case.bias[oc];
                        for ic in 0..c_in {
                            for ikd in 0..kd {
                                let d_in = od as isize * case.stride as isize
                                    + ikd as isize * case.dilation as isize
                                    - case.padding as isize;
                                if d_in >= 0 && (d_in as usize) < d {
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
                                                    let input_index = (((ni * c_in + ic) * d
                                                        + d_in as usize)
                                                        * h
                                                        + h_in as usize)
                                                        * w
                                                        + w_in as usize;
                                                    let weight_index =
                                                        (((oc * c_in + ic) * kd + ikd) * kh + ikh)
                                                            * kw
                                                            + ikw;
                                                    acc += case.input[input_index]
                                                        * case.weight[weight_index];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        out[(((ni * c_out + oc) * d_out + od) * h_out + oh) * w_out + ow] = acc;
                    }
                }
            }
        }
    }
    out
}

fn device_conv3d<B>(backend: &B, case: &Conv3dCase<'_>) -> Vec<f32>
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

    backend
        .conv3d(
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
        )
        .expect("convolution provider dispatch");

    let mut out = vec![0.0; output_layout.numel()];
    backend.copy_to_host(&output, &mut out);
    out
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // The Hermes path may reassociate each kernel-row dot product. The
        // tested fast case uses at most sixteen products per output, so this
        // bound covers row-dot reassociation plus depth/channel accumulation.
        let tol = 256.0 * f32::EPSILON * (1.0 + want.abs());
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
    let input: Vec<f32> = (0..54)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.0625)
        .collect();
    let weight: Vec<f32> = (0..32).map(|i| ((i as f32 % 9.0) - 4.0) * 0.125).collect();
    let bias = [0.1875, -0.3125];

    let fast_case = Conv3dCase {
        input: &input,
        input_shape: [1, 2, 3, 3, 3],
        weight: &weight,
        weight_shape: [2, 2, 2, 2, 2],
        bias: &bias,
        stride: 1,
        padding: 0,
        dilation: 1,
        output_shape: [1, 2, 2, 2, 2],
    };
    let fallback_case = Conv3dCase {
        stride: 1,
        padding: 1,
        dilation: 2,
        output_shape: [1, 2, 3, 3, 3],
        ..fast_case
    };

    let fast_expected = conv3d_reference(&fast_case);
    let fast_actual = device_conv3d(backend, &fast_case);
    assert_close("contiguous", &fast_actual, &fast_expected);

    let fallback_expected = conv3d_reference(&fallback_case);
    let fallback_actual = device_conv3d(backend, &fallback_case);
    assert_close("fallback", &fallback_actual, &fallback_expected);
}

#[test]
fn sequential_conv3d_matches_reference() {
    check_backend(&SequentialBackend);
}

#[test]
fn moirai_conv3d_matches_reference() {
    check_backend(&MoiraiBackend);
}
