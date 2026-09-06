//! Contract tests for the CPU finite-difference seam.
//!
//! These run through a real backend rather than the provider directly, because
//! the backend is what a consumer calls. The oracles are the provider's own
//! result (bitwise, since the seam is supposed to be a borrow and not a
//! recomputation) and the adjoint identity the FDTD leapfrog rests on.

use super::*;
use crate::backend_ops::traits::{FiniteDifference3DOps, StaggeredPairOps};
use coeus_core::{ComputeBackend, CpuAddressableStorage, Layout, SequentialBackend};

const SHAPE: [usize; 3] = [6, 5, 7];

fn layout() -> Layout {
    Layout::new(SHAPE.to_vec().into())
}

fn leto_layout() -> LetoLayout<3> {
    LetoLayout::<3>::try_new(
        SHAPE,
        [(SHAPE[1] * SHAPE[2]) as isize, SHAPE[2] as isize, 1],
        0,
    )
    .expect("row-major layout over a non-empty shape")
}

/// A non-separable field, so an axis mix-up cannot pass unnoticed.
fn field() -> Vec<f64> {
    let mut values = Vec::with_capacity(SHAPE[0] * SHAPE[1] * SHAPE[2]);
    for i in 0..SHAPE[0] {
        for j in 0..SHAPE[1] {
            for k in 0..SHAPE[2] {
                let x = i as f64 * 0.37;
                let y = j as f64 * 0.53;
                let z = k as f64 * 0.71;
                values.push(x.sin() * y.cos() + z.sin() * 0.75);
            }
        }
    }
    values
}

fn on_device(
    backend: &SequentialBackend,
    values: &[f64],
) -> <SequentialBackend as ComputeBackend>::DeviceBuffer<f64> {
    let mut buffer = backend.allocate::<f64>(values.len());
    backend.copy_to_device(values, &mut buffer);
    buffer
}

/// The seam computes what the provider computes, bitwise — because it is the
/// provider's kernel writing into the caller's own storage. Any difference
/// would mean the adaptation copied, reordered, or reimplemented something.
#[test]
fn the_staggered_pair_matches_the_provider_bitwise() {
    use leto::{Array3, ArrayView3 as View};

    let backend = SequentialBackend;
    let host = field();
    let spacing = [1.5e-3_f64, 2.5e-3, 0.5e-3];
    let pair = backend.prepare_staggered_pair(4, spacing).unwrap();
    let reference = StaggeredLeapfrog3D::<f64>::new(4, spacing[0], spacing[1], spacing[2]).unwrap();

    for axis in [Axis::X, Axis::Y, Axis::Z] {
        for divergence in [false, true] {
            let input = on_device(&backend, &host);
            let mut output = backend.allocate_zeroed::<f64>(host.len());
            if divergence {
                backend
                    .staggered_divergence(&pair, axis, &input, &layout(), &mut output, &layout())
                    .unwrap();
            } else {
                backend
                    .staggered_gradient(&pair, axis, &input, &layout(), &mut output, &layout())
                    .unwrap();
            }

            let mut expected = Array3::<f64>::zeros(SHAPE);
            let view = View::try_new(leto_layout(), host.as_slice()).unwrap();
            if divergence {
                reference
                    .divergence_into(axis, view, &mut expected.view_mut())
                    .unwrap();
            } else {
                reference
                    .gradient_into(axis, view, &mut expected.view_mut())
                    .unwrap();
            }

            let operator = if divergence { "divergence" } else { "gradient" };
            for (index, &value) in output.as_slice().iter().enumerate() {
                let i = index / (SHAPE[1] * SHAPE[2]);
                let j = (index / SHAPE[2]) % SHAPE[1];
                let k = index % SHAPE[2];
                assert_eq!(
                    value,
                    expected[[i, j, k]],
                    "{operator}, axis {axis:?}, cell ({i}, {j}, {k})"
                );
            }
        }
    }
}

/// The adjoint identity survives the seam. This is the property a conservative
/// leapfrog rests on, asserted through the backend because that is what a
/// consumer calls.
#[test]
fn the_pair_stays_a_negative_adjoint_through_the_seam() {
    let backend = SequentialBackend;
    let p = field();
    let u: Vec<f64> = field().iter().rev().map(|v| v * 0.6 + 0.1).collect();
    let pair = backend.prepare_staggered_pair(6, [1.0, 1.0, 1.0]).unwrap();

    for axis in [Axis::X, Axis::Y, Axis::Z] {
        let p_device = on_device(&backend, &p);
        let mut grad = backend.allocate_zeroed::<f64>(p.len());
        backend
            .staggered_gradient(&pair, axis, &p_device, &layout(), &mut grad, &layout())
            .unwrap();

        let u_device = on_device(&backend, &u);
        let mut div = backend.allocate_zeroed::<f64>(u.len());
        backend
            .staggered_divergence(&pair, axis, &u_device, &layout(), &mut div, &layout())
            .unwrap();

        let left: f64 = grad.as_slice().iter().zip(&u).map(|(a, b)| a * b).sum();
        let right: f64 = -p
            .iter()
            .zip(div.as_slice())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        // Both sides sum the same products in different orders, so the bound is
        // the accumulated rounding of a length-N sum.
        let bound = 64.0 * f64::EPSILON * left.abs().max(right.abs()) * p.len() as f64;
        assert!(
            (left - right).abs() <= bound,
            "axis {axis:?}: {left:e} vs {right:e} (bound {bound:e})"
        );
        assert!(
            left.abs() > 1e-6,
            "axis {axis:?}: the identity held trivially"
        );
    }
}

#[test]
fn the_fixed_schemes_reach_every_axis() {
    let backend = SequentialBackend;
    // A constant field differentiates to zero under every central scheme, walls
    // included, on every axis.
    let flat = vec![2.75_f64; SHAPE[0] * SHAPE[1] * SHAPE[2]];
    for scheme in [
        FiniteDifference3DScheme::CentralSecondOrder,
        FiniteDifference3DScheme::CentralFourthOrder,
    ] {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let input = on_device(&backend, &flat);
            let mut output = backend.allocate_zeroed::<f64>(flat.len());
            backend
                .finite_difference(
                    scheme,
                    axis,
                    [1.0, 1.0, 1.0],
                    &input,
                    &layout(),
                    &mut output,
                    &layout(),
                )
                .unwrap();
            for &value in output.as_slice() {
                assert_eq!(value, 0.0, "{scheme:?} on {axis:?}");
            }
        }
    }
}

/// A layout the stencils cannot serve is refused by name at the boundary, not
/// mis-swept inside a kernel.
#[test]
fn a_layout_the_stencils_cannot_serve_is_refused() {
    let backend = SequentialBackend;
    let host = field();
    let pair = backend.prepare_staggered_pair(2, [1.0, 1.0, 1.0]).unwrap();
    let input = on_device(&backend, &host);
    let mut output = backend.allocate_zeroed::<f64>(host.len());

    let rank2 = Layout::new(vec![SHAPE[0], SHAPE[1] * SHAPE[2]].into());
    assert!(backend
        .staggered_gradient(&pair, Axis::X, &input, &rank2, &mut output, &layout())
        .is_err());
    assert!(backend
        .staggered_gradient(&pair, Axis::X, &input, &layout(), &mut output, &rank2)
        .is_err());
}

#[test]
fn preparation_rejects_orders_the_derivation_does_not_cover() {
    let backend = SequentialBackend;
    assert!(backend.prepare_staggered_pair(3, [1.0, 1.0, 1.0]).is_err());
    assert!(backend.prepare_staggered_pair(0, [1.0, 1.0, 1.0]).is_err());
    assert!(backend.prepare_staggered_pair(2, [0.0, 1.0, 1.0]).is_err());
    assert!(backend.prepare_staggered_pair(8, [1.0, 1.0, 1.0]).is_ok());
}
