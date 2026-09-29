use coeus_core::{BinaryOp, ComputeBackend, CpuUnaryOp, Layout, Scalar};
use coeus_hephaestus::HephaestusBackend;
use coeus_metal::MetalProvider;
use coeus_ops::{ElementwiseOps, RotateHalfOps};
use std::fmt::Debug;

type Backend = HephaestusBackend<MetalProvider>;

#[test]
fn rotate_half_dispatches_with_metal_parity() {
    if !require_device() {
        return;
    }
    let metal = Backend::new();
    let layout = Layout::new([2, 4].into());
    let mut input = metal
        .allocate::<f32>(8)
        .expect("invariant: test backend operation succeeds");
    metal
        .copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &mut input)
        .expect("invariant: test backend operation succeeds");
    let output = metal
        .rotate_half_storage(&input, &layout)
        .expect("Metal rotate-half dispatch");
    let mut actual = vec![0.0; 8];
    metal
        .copy_to_host(&output, &mut actual)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(actual, [-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]);
}

#[test]
fn partial_update_preserves_metal_parent_and_shared_source() {
    if !require_device() {
        return;
    }

    let backend = Backend::new();
    let parent_layout = Layout::new([2, 3].into());
    let destination_layout = parent_layout.slice(&[(0, 2), (1, 3)]);
    let rhs_layout = Layout::new([2, 2].into());
    let mut destination = backend
        .allocate::<f32>(6)
        .expect("invariant: test backend operation succeeds");
    let mut rhs = backend
        .allocate::<f32>(4)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut destination)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[10.0, 20.0, 30.0, 40.0], &mut rhs)
        .expect("invariant: test backend operation succeeds");
    let shared = destination.clone();

    backend
        .elementwise_binary_update(
            BinaryOp::Add,
            &mut destination,
            &destination_layout,
            &rhs,
            &rhs_layout,
        )
        .expect("Metal partial update");

    let mut actual = [0.0; 6];
    backend
        .copy_to_host(&destination, &mut actual)
        .expect("invariant: test backend operation succeeds");
    assert_close(
        &actual,
        &[1.0, 12.0, 23.0, 4.0, 35.0, 46.0],
        "partial update",
    );
    let mut shared_values = [0.0; 6];
    backend
        .copy_to_host(&shared, &mut shared_values)
        .expect("invariant: test backend operation succeeds");
    assert_close(
        &shared_values,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "partial update shared source",
    );
}

fn require_device() -> bool {
    let available = hephaestus_metal::MetalDevice::try_default().is_ok();
    if !available {
        assert_ne!(
            std::env::var("HEPHAESTUS_METAL_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "Metal CI requires an acquired device"
        );
    }
    available
}

fn assert_close(actual: &[f32], expected: &[f32], operation: &str) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "Metal {operation} expected NaN at {index}");
            continue;
        }
        if expected.is_infinite() {
            assert!(
                actual.is_infinite() && actual.is_sign_positive() == expected.is_sign_positive(),
                "Metal {operation} expected {expected} at {index}, got {actual}"
            );
            continue;
        }
        let tolerance = f32::EPSILON * 512.0 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "Metal {operation} mismatch at {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn assert_integer_comparisons<T>(backend: &Backend, lhs: &[T], rhs: &[T])
where
    T: Scalar + leto_ops::Scalar + Debug + PartialEq,
    coeus_metal::MetalProvider: coeus_hephaestus::ElementwiseProvider<T>,
{
    let layout = Layout::new([lhs.len()].into());
    let mut device_lhs = backend
        .allocate::<T>(lhs.len())
        .expect("invariant: test backend operation succeeds");
    let mut device_rhs = backend
        .allocate::<T>(rhs.len())
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(lhs, &mut device_lhs)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(rhs, &mut device_rhs)
        .expect("invariant: test backend operation succeeds");

    for operation in [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Gt,
        BinaryOp::Le,
        BinaryOp::Ge,
    ] {
        let mut expected = vec![T::zero(); lhs.len()];
        coeus_leto::elementwise_binary_into(
            operation,
            &layout,
            lhs,
            &layout,
            rhs,
            &layout,
            &mut expected,
        )
        .expect("Leto integer comparison oracle failed");
        let mut actual = backend
            .allocate::<T>(lhs.len())
            .expect("invariant: test backend operation succeeds");
        backend
            .elementwise_binary(
                operation,
                &device_lhs,
                &layout,
                &device_rhs,
                &layout,
                &mut actual,
                &layout,
            )
            .expect("Metal integer comparison dispatch failed");
        let mut actual_values = vec![T::zero(); lhs.len()];
        backend
            .copy_to_host(&actual, &mut actual_values)
            .expect("invariant: test backend operation succeeds");
        assert_eq!(
            actual_values, expected,
            "Metal integer {operation:?} mismatch"
        );
    }
}

#[path = "elementwise/extended.rs"]
mod extended;
