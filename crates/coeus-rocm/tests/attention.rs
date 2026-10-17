//! ROCm attention dispatch through the generic coeus-accelerator bridge.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use coeus_accelerator::{HephaestusBackend, HephaestusBackendError};
use coeus_core::{ComputeBackend, Layout};
use coeus_ops::AttentionOps;
use coeus_rocm::RocmProvider;

type Backend = HephaestusBackend<RocmProvider>;

#[expect(
    clippy::print_stderr,
    reason = "a skipped device probe reports why on stderr, where the test harness shows it"
)]
fn device() -> Option<Backend> {
    match hephaestus_rocm::RocmDevice::try_default() {
        Ok(_) => Some(Backend::new()),
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm attention bridge: device unavailable ({error})");
            None
        }
        Err(error) => panic!("ROCm attention bridge requires a physical device: {error}"),
    }
}

#[test]
fn native_attention_dispatches_and_preserves_provider_errors() {
    let Some(backend) = device() else {
        return;
    };
    let layout = Layout::new([1, 1, 1].into());
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut query =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut key =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut value =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the `sdp_attention` dispatch overwrites every element before the buffer is read.
    let mut output =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the `sdp_attention` dispatch overwrites every element before the buffer is read.
    let mut weights =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[2.0], &mut query)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[3.0], &mut key)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[7.0], &mut value)
        .expect("invariant: test backend operation succeeds");

    backend
        .sdp_attention(
            &query,
            &layout,
            &key,
            &layout,
            &value,
            &layout,
            None,
            None,
            false,
            0.5,
            &mut output,
            &layout,
            &mut weights,
            &layout,
        )
        .expect("ROCm provider dispatch");
    let mut actual_output = [0.0];
    let mut actual_weights = [0.0];
    backend
        .copy_to_host(&output, &mut actual_output)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_host(&weights, &mut actual_weights)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(actual_output, [7.0]);
    assert_eq!(actual_weights, [1.0]);

    backend
        .copy_to_device(&[f32::NAN], &mut query)
        .expect("invariant: test backend operation succeeds");
    let error = backend
        .sdp_attention(
            &query,
            &layout,
            &key,
            &layout,
            &value,
            &layout,
            None,
            None,
            false,
            1.0,
            &mut output,
            &layout,
            &mut weights,
            &layout,
        )
        .expect_err("non-finite query must remain a provider error");
    match error {
        HephaestusBackendError::Device { operation, source } => {
            assert_eq!(operation, "attention forward");
            assert_eq!(
                source.to_string(),
                "invalid configuration: attention query contains a non-finite value"
            );
        }
        other => panic!("expected typed provider error, got {other}"),
    }
}
