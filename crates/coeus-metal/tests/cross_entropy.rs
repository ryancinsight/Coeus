//! Metal cross-entropy dispatch through the generic coeus-accelerator bridge.

#![cfg(target_os = "macos")]

use coeus_accelerator::HephaestusBackend;
use coeus_core::{ComputeBackend, Layout};
use coeus_metal::MetalProvider;
use coeus_ops::CrossEntropyOps;

type Backend = HephaestusBackend<MetalProvider>;

#[test]
fn cross_entropy_dispatches_with_metal_value_and_gradient_contract() {
    let backend = Backend::new();
    let logits_layout = Layout::new([2, 3].into());
    let scalar_layout = Layout::new([1].into());
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut logits =
        unsafe { backend.allocate::<f32>(6) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the `cross_entropy_forward` dispatch overwrites every element before the buffer is read.
    let mut loss =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the `cross_entropy_forward` dispatch overwrites every element before the buffer is read.
    let mut probabilities =
        unsafe { backend.allocate::<f32>(6) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut output_gradient =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut logit_gradient =
        unsafe { backend.allocate::<f32>(6) }.expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[1.5, 0.5, -0.5, -1.0, 2.0, 0.0], &mut logits)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[1.0], &mut output_gradient)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[0.0; 6], &mut logit_gradient)
        .expect("invariant: test backend operation succeeds");
    let targets = backend
        .prepare_cross_entropy_targets(&[0, 1])
        .expect("Metal target upload");

    backend
        .cross_entropy_forward(
            &logits,
            &logits_layout,
            &targets,
            &mut loss,
            &scalar_layout,
            &mut probabilities,
            &logits_layout,
        )
        .expect("Metal cross-entropy forward");
    backend
        .cross_entropy_backward_accumulate(
            &output_gradient,
            &scalar_layout,
            &probabilities,
            &logits_layout,
            &targets,
            &mut logit_gradient,
            &logits_layout,
        )
        .expect("Metal cross-entropy backward");

    let mut actual_loss = [0.0];
    let mut actual_gradient = [0.0; 6];
    backend
        .copy_to_host(&loss, &mut actual_loss)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_host(&logit_gradient, &mut actual_gradient)
        .expect("invariant: test backend operation succeeds");
    assert!((actual_loss[0] - 0.288_726).abs() < 1.0e-4);
    for (actual, expected) in actual_gradient.iter().zip([
        -0.167_379, 0.122_364, 0.045_015, 0.021_005, -0.078_103, 0.057_098,
    ]) {
        assert!((*actual - expected).abs() < 1.0e-4);
    }
}
