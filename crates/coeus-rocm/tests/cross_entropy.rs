//! ROCm cross-entropy dispatch through the generic Coeus-Hephaestus bridge.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use coeus_core::{ComputeBackend, Layout};
use coeus_hephaestus::HephaestusBackend;
use coeus_ops::CrossEntropyOps;
use coeus_rocm::RocmProvider;

type Backend = HephaestusBackend<RocmProvider>;

#[test]
fn cross_entropy_dispatches_with_rocm_value_and_gradient_contract() {
    if let Err(error) = hephaestus_rocm::RocmDevice::try_default() {
        assert!(
            std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none(),
            "ROCm cross-entropy requires a physical device: {error}"
        );
        return;
    }
    let backend = Backend::new();
    let logits_layout = Layout::new([2, 3].into());
    let scalar_layout = Layout::new([1].into());
    let mut logits = backend.allocate::<f32>(6);
    let mut loss = backend.allocate::<f32>(1);
    let mut probabilities = backend.allocate::<f32>(6);
    let mut output_gradient = backend.allocate::<f32>(1);
    let mut logit_gradient = backend.allocate::<f32>(6);
    backend.copy_to_device(&[1.5, 0.5, -0.5, -1.0, 2.0, 0.0], &mut logits);
    backend.copy_to_device(&[1.0], &mut output_gradient);
    backend.copy_to_device(&[0.0; 6], &mut logit_gradient);
    let targets = <_ as CrossEntropyOps<f32>>::prepare_cross_entropy_targets(&backend, &[0, 1])
        .expect("ROCm target upload");

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
        .expect("ROCm cross-entropy forward");
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
        .expect("ROCm cross-entropy backward");

    let mut actual_loss = [0.0];
    let mut actual_gradient = [0.0; 6];
    backend.copy_to_host(&loss, &mut actual_loss);
    backend.copy_to_host(&logit_gradient, &mut actual_gradient);
    assert!((actual_loss[0] - 0.288_726).abs() < 1.0e-4);
    for (actual, expected) in actual_gradient.iter().zip([
        -0.167_379, 0.122_364, 0.045_015, 0.021_005, -0.078_103, 0.057_098,
    ]) {
        assert!((*actual - expected).abs() < 1.0e-4);
    }
}

#[test]
fn cross_entropy_dispatches_with_rocm_f64_value_and_gradient_contract() {
    if let Err(error) = hephaestus_rocm::RocmDevice::try_default() {
        assert!(
            std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none(),
            "ROCm f64 cross-entropy requires a physical device: {error}"
        );
        return;
    }
    // The f64 expectations come from the CPU backend at runtime, so no
    // transcribed constants can drift from the Leto oracle.
    let cpu = coeus_core::SequentialBackend::new();
    let backend = Backend::new();
    let logits_layout = Layout::new([2, 3].into());
    let scalar_layout = Layout::new([1].into());
    let logits_host = [1.5_f64, 0.5, -0.5, -1.0, 2.0, 0.0];

    let mut cpu_logits = cpu.allocate::<f64>(6);
    let mut cpu_loss = cpu.allocate::<f64>(1);
    let mut cpu_probabilities = cpu.allocate::<f64>(6);
    let mut cpu_output_gradient = cpu.allocate::<f64>(1);
    let mut cpu_logit_gradient = cpu.allocate::<f64>(6);
    cpu.copy_to_device(&logits_host, &mut cpu_logits);
    cpu.copy_to_device(&[1.0_f64], &mut cpu_output_gradient);
    cpu.copy_to_device(&[0.0_f64; 6], &mut cpu_logit_gradient);
    let cpu_targets = <_ as CrossEntropyOps<f64>>::prepare_cross_entropy_targets(&cpu, &[0, 1])
        .expect("CPU target upload");
    cpu.cross_entropy_forward(
        &cpu_logits,
        &logits_layout,
        &cpu_targets,
        &mut cpu_loss,
        &scalar_layout,
        &mut cpu_probabilities,
        &logits_layout,
    )
    .expect("CPU f64 cross-entropy forward");
    cpu.cross_entropy_backward_accumulate(
        &cpu_output_gradient,
        &scalar_layout,
        &cpu_probabilities,
        &logits_layout,
        &cpu_targets,
        &mut cpu_logit_gradient,
        &logits_layout,
    )
    .expect("CPU f64 cross-entropy backward");

    let mut logits = backend.allocate::<f64>(6);
    let mut loss = backend.allocate::<f64>(1);
    let mut probabilities = backend.allocate::<f64>(6);
    let mut output_gradient = backend.allocate::<f64>(1);
    let mut logit_gradient = backend.allocate::<f64>(6);
    backend.copy_to_device(&logits_host, &mut logits);
    backend.copy_to_device(&[1.0_f64], &mut output_gradient);
    backend.copy_to_device(&[0.0_f64; 6], &mut logit_gradient);
    let targets = <_ as CrossEntropyOps<f64>>::prepare_cross_entropy_targets(&backend, &[0, 1])
        .expect("ROCm f64 target upload");
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
        .expect("ROCm f64 cross-entropy forward");
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
        .expect("ROCm f64 cross-entropy backward");

    let mut expected_loss = [0.0_f64; 1];
    let mut expected_gradient = [0.0_f64; 6];
    let mut actual_loss = [0.0_f64; 1];
    let mut actual_gradient = [0.0_f64; 6];
    cpu.copy_to_host(&cpu_loss, &mut expected_loss);
    cpu.copy_to_host(&cpu_logit_gradient, &mut expected_gradient);
    backend.copy_to_host(&loss, &mut actual_loss);
    backend.copy_to_host(&logit_gradient, &mut actual_gradient);
    assert!((actual_loss[0] - expected_loss[0]).abs() < 1.0e-9);
    for (actual, expected) in actual_gradient.iter().zip(expected_gradient.iter()) {
        assert!((actual - expected).abs() < 1.0e-9);
    }
}
