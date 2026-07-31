//! Metal attention dispatch through the generic Coeus-Hephaestus bridge.

#![cfg(target_os = "macos")]

use coeus_core::{ComputeBackend, Layout};
use coeus_hephaestus::HephaestusBackendError;
use coeus_metal::MetalBackend;
use coeus_ops::AttentionOps;

#[test]
fn native_attention_dispatches_and_preserves_provider_errors() {
    let backend = MetalBackend::new();
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
        .expect("Metal provider dispatch");
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
