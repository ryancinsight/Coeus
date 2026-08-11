//! ROCm attention dispatch through the generic Coeus-Hephaestus bridge.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use coeus_core::{ComputeBackend, Layout};
use coeus_hephaestus::{HephaestusBackend, HephaestusBackendError};
use coeus_ops::AttentionOps;
use coeus_rocm::RocmProvider;

type Backend = HephaestusBackend<RocmProvider>;

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
    let mut query = backend.allocate::<f32>(1);
    let mut key = backend.allocate::<f32>(1);
    let mut value = backend.allocate::<f32>(1);
    let mut output = backend.allocate::<f32>(1);
    let mut weights = backend.allocate::<f32>(1);
    backend.copy_to_device(&[2.0], &mut query);
    backend.copy_to_device(&[3.0], &mut key);
    backend.copy_to_device(&[7.0], &mut value);

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
    backend.copy_to_host(&output, &mut actual_output);
    backend.copy_to_host(&weights, &mut actual_weights);
    assert_eq!(actual_output, [7.0]);
    assert_eq!(actual_weights, [1.0]);

    backend.copy_to_device(&[f32::NAN], &mut query);
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
