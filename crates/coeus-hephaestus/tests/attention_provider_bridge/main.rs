use std::sync::atomic::Ordering;

use coeus_core::Layout;
use coeus_hephaestus::{HephaestusBackend, HephaestusBackendError};
use coeus_ops::AttentionOps as CoeusAttentionOps;

mod attention_ops;
mod device;

use attention_ops::*;
use device::*;

#[test]
fn attention_bridge_binds_provider_operands_and_maps_errors() {
    let backend = HephaestusBackend::<TestProvider>::new();
    let tensor_layout = Layout::new([2, 2, 2].into());
    let mask_layout = Layout::new([1, 2].into());
    let query = storage(8);
    let key = storage(8);
    let value = storage(8);
    let mask = storage(2);
    let mut output = storage(8);
    let mut weights = storage(8);
    write_storage(&query, &[0.0; 8]);
    write_storage(&key, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    write_storage(&value, &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]);
    write_storage(&mask, &[1.0, 1.0]);

    backend
        .sdp_attention(
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            Some(&mask),
            Some(&mask_layout),
            true,
            0.5,
            &mut output,
            &tensor_layout,
            &mut weights,
            &tensor_layout,
        )
        .expect("forward provider dispatch");
    assert_eq!(FORWARD_BATCHES.load(Ordering::SeqCst), 2);
    assert!(FORWARD_CAUSAL.load(Ordering::SeqCst));
    assert_eq!(FORWARD_MASK_GROUP.load(Ordering::SeqCst), 2);
    assert_eq!(
        read_storage(&output),
        [10.0, 11.0, 15.0, 16.0, 30.0, 31.0, 35.0, 36.0]
    );
    assert_eq!(
        read_storage(&weights),
        [1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5]
    );

    let grad_output = storage(8);
    let mut grad_query = storage(8);
    let mut grad_value = storage(8);
    write_storage(&grad_output, &[1.0; 8]);
    write_storage(&grad_query, &[1.0; 8]);
    write_storage(&grad_value, &[1.0; 8]);
    backend
        .sdp_attention_backward(
            &grad_output,
            &tensor_layout,
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            &weights,
            &tensor_layout,
            0.5,
            Some((&mut grad_query, &tensor_layout)),
            None,
            Some((&mut grad_value, &tensor_layout)),
        )
        .expect("backward provider dispatch");
    assert_eq!(BACKWARD_GRADIENTS.load(Ordering::SeqCst), 0b101);
    assert_eq!(
        read_storage(&grad_query),
        [1.0, 1.0, -1.5, 3.5, 1.0, 1.0, -1.5, 3.5]
    );
    assert_eq!(
        read_storage(&grad_value),
        [2.5, 2.5, 1.5, 1.5, 2.5, 2.5, 1.5, 1.5]
    );

    FAIL_FORWARD.store(true, Ordering::SeqCst);
    let error = backend
        .sdp_attention(
            &query,
            &tensor_layout,
            &key,
            &tensor_layout,
            &value,
            &tensor_layout,
            None,
            None,
            false,
            1.0,
            &mut output,
            &tensor_layout,
            &mut weights,
            &tensor_layout,
        )
        .expect_err("provider failure must remain typed");
    match error {
        HephaestusBackendError::Device { operation, source } => {
            assert_eq!(operation, "attention forward");
            assert_eq!(
                source.to_string(),
                "kernel dispatch failed: injected provider failure"
            );
        }
        other => panic!("expected typed provider failure, got {other}"),
    }
}
