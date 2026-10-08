//! The fixed-scheme seam reaches either backend from one call site.
//!
//! `coeus_ops::FiniteDifference3DOps` is implemented for the CPU backend over
//! Leto and for this one over Hephaestus. These tests call the same trait
//! method on both and compare, which is the claim the seam exists to support:
//! a consumer binds the trait, not a device.

use coeus_core::{BackendError, ComputeBackend, Layout, SequentialBackend};
use coeus_ops::{Axis, FiniteDifference3DOps, FiniteDifference3DScheme};
use coeus_tensor::Tensor;
use coeus_wgpu::{WgpuBackend, WgpuBackendError};

const SHAPE: [usize; 3] = [8, 7, 10];
const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
const SCHEMES: [FiniteDifference3DScheme; 5] = [
    FiniteDifference3DScheme::CentralSecondOrder,
    FiniteDifference3DScheme::CentralFourthOrder,
    FiniteDifference3DScheme::CentralSixthOrder,
    FiniteDifference3DScheme::StaggeredForward,
    FiniteDifference3DScheme::StaggeredBackward,
];
const SPACING: [f32; 3] = [1.5e-3, 2.5e-3, 0.5e-3];

fn layout_for(shape: &[usize]) -> Layout {
    Layout::new(shape.to_vec().into())
}

fn output_shape(scheme: FiniteDifference3DScheme, axis: Axis, shape: [usize; 3]) -> Vec<usize> {
    let mut out = shape.to_vec();
    if matches!(scheme, FiniteDifference3DScheme::StaggeredForward) {
        let lane = match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        };
        out[lane] -= 1;
    }
    out
}

/// A non-separable field, so an axis or stride mistake cannot cancel out.
fn field(shape: [usize; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape.iter().product());
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let x = i as f32 * 0.37;
                let y = j as f32 * 0.53;
                let z = k as f32 * 0.71;
                values.push(x.sin() * y.cos() + z.sin() * 0.75 + 0.25);
            }
        }
    }
    values
}

/// The kernels keep the provider's operation order, so the claim is a tight
/// bound over shader-compiler reassociation rather than bitwise equality.
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let scale = expected.iter().fold(1.0_f32, |acc, v| acc.max(v.abs()));
    let bound = 32.0 * f32::EPSILON * scale;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= bound,
            "{what}: cell {index}: {actual} vs {expected}, bound {bound:e}"
        );
    }
}

fn through_both(
    scheme: FiniteDifference3DScheme,
    axis: Axis,
    shape: [usize; 3],
) -> (Vec<f32>, Vec<f32>) {
    let sequential = SequentialBackend;
    let wgpu = WgpuBackend::new();
    let host = field(shape);
    let input = Tensor::<f32, SequentialBackend>::from_slice(shape.to_vec(), &host);
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);
    let out_shape = output_shape(scheme, axis, shape);
    let out_cells: usize = out_shape.iter().product();
    let out_layout = layout_for(&out_shape);

    let mut cpu_storage = sequential.allocate_zeroed::<f32>(out_cells);
    FiniteDifference3DOps::<f32>::finite_difference(
        &sequential,
        scheme,
        axis,
        SPACING,
        input.storage(),
        input.layout(),
        &mut cpu_storage,
        &out_layout,
    )
    .expect("sequential sweep");
    let expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_storage, out_layout.clone())
            .as_slice()
            .to_vec();

    let mut device_storage = wgpu.allocate_zeroed::<f32>(out_cells);
    FiniteDifference3DOps::<f32>::finite_difference(
        &wgpu,
        scheme,
        axis,
        SPACING,
        input_wgpu.storage(),
        input_wgpu.layout(),
        &mut device_storage,
        &out_layout,
    )
    .expect("wgpu sweep");
    let actual = Tensor::<f32, WgpuBackend>::from_raw_parts(device_storage, out_layout)
        .to_backend_on(&wgpu, &sequential)
        .as_slice()
        .to_vec();

    (actual, expected)
}

#[test]
fn wgpu_fixed_fd_matches_sequential_on_every_scheme_and_axis() {
    for scheme in SCHEMES {
        for axis in AXES {
            let (actual, expected) = through_both(scheme, axis, SHAPE);
            assert_close(
                &actual,
                &expected,
                &format!("{scheme:?} on {axis:?} over {SHAPE:?}"),
            );
        }
    }
}

/// Each scheme's minimum axis extent, where every boundary fall-back branch
/// is live: central-2 on 3 points, central-4 on 1 (flat) and 2, central-6 on
/// exactly 7, staggered on 2.
#[test]
fn wgpu_fixed_fd_matches_sequential_on_minimum_extents() {
    let minima: [(FiniteDifference3DScheme, &[usize]); 5] = [
        (FiniteDifference3DScheme::CentralSecondOrder, &[3]),
        (FiniteDifference3DScheme::CentralFourthOrder, &[1, 2]),
        (FiniteDifference3DScheme::CentralSixthOrder, &[7]),
        (FiniteDifference3DScheme::StaggeredForward, &[2]),
        (FiniteDifference3DScheme::StaggeredBackward, &[2]),
    ];
    for (scheme, extents) in minima {
        for axis in AXES {
            for extent in extents {
                let mut shape = [5_usize, 4, 6];
                shape[match axis {
                    Axis::X => 0,
                    Axis::Y => 1,
                    Axis::Z => 2,
                }] = *extent;
                let (actual, expected) = through_both(scheme, axis, shape);
                assert_close(
                    &actual,
                    &expected,
                    &format!("{scheme:?} on {axis:?} over {shape:?}"),
                );
            }
        }
    }
}

/// A grid thinner than the scheme's minimum is refused by the provider
/// parameters, carrying the dispatch error rather than sweeping a shape the
/// kernel cannot resolve.
#[test]
fn wgpu_fixed_fd_rejects_a_grid_thinner_than_the_scheme() {
    let wgpu = WgpuBackend::new();
    for (scheme, axis, extent) in [
        (
            FiniteDifference3DScheme::CentralSecondOrder,
            Axis::X,
            2_usize,
        ),
        (
            FiniteDifference3DScheme::CentralSixthOrder,
            Axis::Y,
            6_usize,
        ),
        (
            FiniteDifference3DScheme::StaggeredBackward,
            Axis::Z,
            1_usize,
        ),
    ] {
        let mut shape = [8_usize, 8, 8];
        shape[match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }] = extent;
        let count: usize = shape.iter().product();
        let input = wgpu.allocate_zeroed::<f32>(count);
        let mut output = wgpu.allocate_zeroed::<f32>(count);
        let layout = layout_for(&shape);
        match FiniteDifference3DOps::<f32>::finite_difference(
            &wgpu,
            scheme,
            axis,
            SPACING,
            &input,
            &layout,
            &mut output,
            &layout,
        ) {
            Err(WgpuBackendError::Dispatch { operation, .. }) => {
                assert_eq!(operation, "finite_difference");
            }
            other => panic!("expected a dispatch rejection for {scheme:?}, got {other:?}"),
        }
    }
}

#[test]
fn wgpu_fixed_fd_rejects_unrepresentable_operand_layouts() {
    let sequential = SequentialBackend;
    let wgpu = WgpuBackend::new();
    let host = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &field(SHAPE));
    let input = host.to_backend_on(&sequential, &wgpu);
    let shape = Layout::new(SHAPE.to_vec().into());
    let strided = Layout::from_shape_strides(SHAPE.into(), [70, 1, 7].as_slice().into(), 0);
    let offset = Layout::from_shape_strides(SHAPE.into(), [70, 10, 1].as_slice().into(), 1);
    let input_strides = "fixed-fd input layout must be contiguous with zero offset, got strides [70, 1, 7] and offset 0";
    let output_strides = "fixed-fd output layout must be contiguous with zero offset, got strides [70, 1, 7] and offset 0";
    let input_offset = "fixed-fd input layout must be contiguous with zero offset, got strides [70, 10, 1] and offset 1";
    let output_offset = "fixed-fd output layout must be contiguous with zero offset, got strides [70, 10, 1] and offset 1";
    // A forward sweep must shrink the output; a central sweep must not.
    let forward_full =
        "fixed-fd StaggeredForward output shape [8, 7, 10] must be [7, 7, 10] for input shape [8, 7, 10]";
    let central_shrunk =
        "fixed-fd CentralSecondOrder output shape [7, 7, 10] must be [8, 7, 10] for input shape [8, 7, 10]";
    let shrunk = Layout::new([7, 7, 10].into());
    for (scheme, input_layout, output_layout, expected_reason) in [
        (
            FiniteDifference3DScheme::CentralSecondOrder,
            strided.clone(),
            shape.clone(),
            input_strides,
        ),
        (
            FiniteDifference3DScheme::CentralSecondOrder,
            shape.clone(),
            strided,
            output_strides,
        ),
        (
            FiniteDifference3DScheme::CentralSecondOrder,
            offset.clone(),
            shape.clone(),
            input_offset,
        ),
        (
            FiniteDifference3DScheme::CentralSecondOrder,
            shape.clone(),
            offset,
            output_offset,
        ),
        (
            FiniteDifference3DScheme::StaggeredForward,
            shape.clone(),
            shape.clone(),
            forward_full,
        ),
        (
            FiniteDifference3DScheme::CentralSecondOrder,
            shape.clone(),
            shrunk,
            central_shrunk,
        ),
    ] {
        let sentinel = vec![13.0_f32; SHAPE.iter().product()];
        let initial = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &sentinel);
        let mut output = initial.to_backend_on(&sequential, &wgpu);
        match FiniteDifference3DOps::<f32>::finite_difference(
            &wgpu,
            scheme,
            Axis::X,
            SPACING,
            input.storage(),
            &input_layout,
            output.storage_mut(),
            &output_layout,
        ) {
            Err(WgpuBackendError::Validation(BackendError::Storage { operation, reason })) => {
                assert_eq!(operation, "finite_difference");
                assert_eq!(reason, expected_reason);
            }
            other => panic!("expected a layout rejection, got {other:?}"),
        }
        // Refusal leaves the destination untouched.
        let actual = output.to_backend_on(&wgpu, &sequential);
        assert_eq!(actual.as_slice(), sentinel.as_slice());
    }
}
