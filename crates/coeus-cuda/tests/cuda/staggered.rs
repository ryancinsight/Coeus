//! The finite-difference seam reaches either backend from one call site.
//!
//! `coeus_ops::StaggeredPairOps` is implemented for the CPU backend over Leto
//! and for this one over Hephaestus. These tests call the same trait methods on
//! both and compare, which is the claim the seam exists to support: a consumer
//! binds the trait, not a device.

use coeus_core::{BackendError, ComputeBackend, Layout, SequentialBackend};
use coeus_cuda::{CudaBackend, CudaBackendError};
use coeus_ops::{Axis, StaggeredPairOps};
use coeus_tensor::Tensor;

const SHAPE: [usize; 3] = [8, 6, 10];
const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
const SPACING: [f32; 3] = [1.5e-3, 2.5e-3, 0.5e-3];

fn cells() -> usize {
    SHAPE.iter().product()
}

fn layout() -> Layout {
    Layout::new(SHAPE.to_vec().into())
}

/// A non-separable field, so an axis or stride mistake cannot cancel out.
fn field() -> Vec<f32> {
    let mut values = Vec::with_capacity(cells());
    for i in 0..SHAPE[0] {
        for j in 0..SHAPE[1] {
            for k in 0..SHAPE[2] {
                let x = i as f32 * 0.37;
                let y = j as f32 * 0.53;
                let z = k as f32 * 0.71;
                values.push(x.sin() * y.cos() + z.sin() * 0.75 + 0.25);
            }
        }
    }
    values
}

/// The stencil sums `2N` taps and a GPU thread accumulates in a different order
/// from the CPU sweep, so the claim is an epsilon bound rather than bitwise
/// equality.
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

fn through_both(axis: Axis, order: usize, divergence: bool) -> (Vec<f32>, Vec<f32>) {
    let sequential = SequentialBackend;
    let cuda = CudaBackend::new();
    let host = field();
    let input = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &host);
    let input_cuda = input.to_backend_on(&sequential, &cuda);

    let cpu_pair = StaggeredPairOps::<f32>::prepare_staggered_pair(&sequential, order, SPACING)
        .expect("sequential staggered preparation");
    let mut cpu_storage = sequential.allocate_zeroed::<f32>(cells());
    if divergence {
        StaggeredPairOps::<f32>::staggered_divergence(
            &sequential,
            &cpu_pair,
            axis,
            input.storage(),
            input.layout(),
            &mut cpu_storage,
            &layout(),
        )
    } else {
        StaggeredPairOps::<f32>::staggered_gradient(
            &sequential,
            &cpu_pair,
            axis,
            input.storage(),
            input.layout(),
            &mut cpu_storage,
            &layout(),
        )
    }
    .expect("sequential staggered dispatch");
    let expected = Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_storage, layout())
        .as_slice()
        .to_vec();

    let device_pair = StaggeredPairOps::<f32>::prepare_staggered_pair(&cuda, order, SPACING)
        .expect("cuda staggered preparation");
    let mut device_storage = cuda.allocate_zeroed::<f32>(cells());
    if divergence {
        StaggeredPairOps::<f32>::staggered_divergence(
            &cuda,
            &device_pair,
            axis,
            input_cuda.storage(),
            input_cuda.layout(),
            &mut device_storage,
            &layout(),
        )
    } else {
        StaggeredPairOps::<f32>::staggered_gradient(
            &cuda,
            &device_pair,
            axis,
            input_cuda.storage(),
            input_cuda.layout(),
            &mut device_storage,
            &layout(),
        )
    }
    .expect("cuda staggered dispatch");
    let actual = Tensor::<f32, CudaBackend>::from_raw_parts(device_storage, layout())
        .to_backend_on(&cuda, &sequential)
        .as_slice()
        .to_vec();

    (actual, expected)
}

#[test]
fn cuda_staggered_gradient_matches_sequential() {
    if !crate::availability::device_available() {
        return;
    }
    for axis in AXES {
        for order in [2_usize, 4] {
            let (actual, expected) = through_both(axis, order, false);
            assert_close(
                &actual,
                &expected,
                &format!("gradient {axis:?} order {order}"),
            );
        }
    }
}

#[test]
fn cuda_staggered_divergence_matches_sequential() {
    if !crate::availability::device_available() {
        return;
    }
    for axis in AXES {
        for order in [2_usize, 4] {
            let (actual, expected) = through_both(axis, order, true);
            assert_close(
                &actual,
                &expected,
                &format!("divergence {axis:?} order {order}"),
            );
        }
    }
}

/// `D = -Gᵀ` measured through the seam on the device's own outputs. Agreeing
/// with the CPU and being an adjoint are different claims, and the conservative
/// leapfrog this pair exists for rests on the second.
#[test]
fn cuda_staggered_pair_is_a_negative_adjoint_through_the_seam() {
    if !crate::availability::device_available() {
        return;
    }
    let sequential = SequentialBackend;
    let cuda = CudaBackend::new();
    let p = field();
    let u: Vec<f32> = p.iter().rev().map(|v| v * 0.6 + 0.1).collect();
    let pair = StaggeredPairOps::<f32>::prepare_staggered_pair(&cuda, 4, [1.0, 1.0, 1.0])
        .expect("cuda staggered preparation");

    for axis in AXES {
        let p_host = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &p);
        let p_device = p_host.to_backend_on(&sequential, &cuda);
        let mut gradient = cuda.allocate_zeroed::<f32>(cells());
        StaggeredPairOps::<f32>::staggered_gradient(
            &cuda,
            &pair,
            axis,
            p_device.storage(),
            p_device.layout(),
            &mut gradient,
            &layout(),
        )
        .expect("cuda gradient dispatch");
        let gradient = Tensor::<f32, CudaBackend>::from_raw_parts(gradient, layout())
            .to_backend_on(&cuda, &sequential)
            .as_slice()
            .to_vec();

        let u_host = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &u);
        let u_device = u_host.to_backend_on(&sequential, &cuda);
        let mut divergence = cuda.allocate_zeroed::<f32>(cells());
        StaggeredPairOps::<f32>::staggered_divergence(
            &cuda,
            &pair,
            axis,
            u_device.storage(),
            u_device.layout(),
            &mut divergence,
            &layout(),
        )
        .expect("cuda divergence dispatch");
        let divergence = Tensor::<f32, CudaBackend>::from_raw_parts(divergence, layout())
            .to_backend_on(&cuda, &sequential)
            .as_slice()
            .to_vec();

        let left: f32 = gradient.iter().zip(&u).map(|(a, b)| a * b).sum();
        let right: f32 = -p.iter().zip(&divergence).map(|(a, b)| a * b).sum::<f32>();
        // Both sides sum the same products in a different order, so the bound
        // is the accumulated rounding of a length-N f32 sum.
        let bound = 64.0 * f32::EPSILON * left.abs().max(right.abs()).max(1.0) * cells() as f32;
        assert!(
            (left - right).abs() <= bound,
            "{axis:?}: <Gp,u> {left:e} vs -<p,Du> {right:e} (bound {bound:e})"
        );
        assert!(
            left.abs() > 1e-3,
            "{axis:?}: the identity held trivially, inner product {left:e}"
        );
    }
}

/// A grid thinner than the stencil is refused when the parameters are built,
/// carrying the backend's typed error rather than sweeping a shape the kernel
/// cannot resolve in one reflection step.
#[test]
fn cuda_staggered_rejects_a_grid_thinner_than_the_stencil() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda = CudaBackend::new();
    let thin = vec![4_usize, 8, 8];
    let count: usize = thin.iter().product();
    let pair = StaggeredPairOps::<f32>::prepare_staggered_pair(&cuda, 6, [1.0, 1.0, 1.0])
        .expect("cuda staggered preparation");
    let input = cuda.allocate_zeroed::<f32>(count);
    let mut output = cuda.allocate_zeroed::<f32>(count);
    let thin_layout = Layout::new(thin.into());

    assert!(StaggeredPairOps::<f32>::staggered_gradient(
        &cuda,
        &pair,
        Axis::X,
        &input,
        &thin_layout,
        &mut output,
        &thin_layout,
    )
    .is_err());
}

#[test]
fn cuda_staggered_rejects_unrepresentable_operand_layouts() {
    if !crate::availability::device_available() {
        return;
    }
    let sequential = SequentialBackend;
    let cuda = CudaBackend::new();
    let pair = StaggeredPairOps::<f32>::prepare_staggered_pair(&cuda, 4, SPACING)
        .expect("cuda staggered preparation");
    let host = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &field());
    let input = host.to_backend_on(&sequential, &cuda);
    let strided = Layout::from_shape_strides(SHAPE.into(), [60, 1, 6].as_slice().into(), 0);
    let offset = Layout::from_shape_strides(SHAPE.into(), [60, 10, 1].as_slice().into(), 1);
    let input_strides = "staggered input layout must be contiguous with zero offset, got strides [60, 1, 6] and offset 0";
    let output_strides = "staggered output layout must be contiguous with zero offset, got strides [60, 1, 6] and offset 0";
    let input_offset = "staggered input layout must be contiguous with zero offset, got strides [60, 10, 1] and offset 1";
    let output_offset = "staggered output layout must be contiguous with zero offset, got strides [60, 10, 1] and offset 1";
    let shape_mismatch = "staggered input shape [6, 8, 10] must equal output shape [8, 6, 10]";
    for (input_layout, output_layout, expected_reason) in [
        (strided.clone(), layout(), input_strides),
        (layout(), strided, output_strides),
        (offset.clone(), layout(), input_offset),
        (layout(), offset, output_offset),
        (Layout::new([6, 8, 10].into()), layout(), shape_mismatch),
    ] {
        // Equal buffer lengths cannot detect a shape permutation, and the
        // sentinel confirms refusal leaves the destination untouched.
        let sentinel = vec![13.0_f32; cells()];
        let initial = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &sentinel);
        let mut output = initial.to_backend_on(&sequential, &cuda);
        for (expected_operation, result) in [
            (
                "staggered_gradient",
                StaggeredPairOps::<f32>::staggered_gradient(
                    &cuda,
                    &pair,
                    Axis::X,
                    input.storage(),
                    &input_layout,
                    output.storage_mut(),
                    &output_layout,
                ),
            ),
            (
                "staggered_divergence",
                StaggeredPairOps::<f32>::staggered_divergence(
                    &cuda,
                    &pair,
                    Axis::X,
                    input.storage(),
                    &input_layout,
                    output.storage_mut(),
                    &output_layout,
                ),
            ),
        ] {
            match result {
                Err(CudaBackendError::Validation {
                    source: BackendError::Storage { operation, reason },
                }) => {
                    assert_eq!(operation, expected_operation);
                    assert_eq!(reason, expected_reason);
                }
                other => panic!("expected a layout rejection, got {other:?}"),
            }
        }
        let actual = output.to_backend_on(&cuda, &sequential);
        assert_eq!(actual.as_slice(), sentinel.as_slice());
    }
}

#[test]
fn staggered_preparation_rejects_invalid_spacing_without_a_device() {
    // Backend construction is a ZST operation; invalid preparation must not
    // reach device discovery or kernel compilation.
    let cuda = CudaBackend::new();
    for axis in 0..3 {
        for value in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut spacing = SPACING;
            spacing[axis] = value;
            match StaggeredPairOps::<f32>::prepare_staggered_pair(&cuda, 4, spacing) {
                Err(CudaBackendError::Validation {
                    source: BackendError::Storage { operation, reason },
                }) => {
                    assert_eq!(operation, "prepare_staggered_pair");
                    assert_eq!(
                        reason,
                        format!(
                            "staggered spacing axis {axis} must be finite and positive, got {value}",
                        ),
                    );
                }
                Err(other) => panic!("expected a spacing rejection, got {other:?}"),
                Ok(_) => panic!("accepted invalid spacing on axis {axis}: {value}"),
            }
        }
    }
}
