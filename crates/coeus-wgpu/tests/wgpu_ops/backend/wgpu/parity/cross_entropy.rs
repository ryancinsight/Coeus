use super::{assert_parity, seq, to_cpu, to_gpu};
use coeus_autograd::Var;
use coeus_core::{BackendError, ComputeBackend, Layout};
use coeus_ops::CrossEntropyOps;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackendError;

#[test]
fn cross_entropy_dispatches_with_wgpu_value_and_gradient_parity() {
    if !crate::availability::device_available("coeus-wgpu-cross-entropy-test") {
        return;
    }

    let logits = Tensor::from_slice_on(
        [3, 4],
        &[
            1.5_f32, 0.5, -0.5, 0.25, -1.0, 2.0, 0.0, 0.75, 3.0, -2.0, 1.0, 0.0,
        ],
        &seq(),
    )
    .expect("invariant: test backend operation succeeds");
    let cpu_logits =
        Var::new(logits.clone(), true).expect("invariant: test backend operation succeeds");
    let wgpu_logits =
        Var::new(to_gpu(&logits), true).expect("invariant: test backend operation succeeds");
    let targets = [0_usize, 1, 2];

    let invalid_target = coeus_nn::cross_entropy_loss(&wgpu_logits, &[0, 1, 4]);
    assert!(matches!(
        invalid_target,
        Err(WgpuBackendError::Validation(
            coeus_core::BackendError::IndexOutOfRange {
                position: 2,
                index: 4,
                bound: 4,
                ..
            }
        ))
    ));

    let cpu_loss = coeus_nn::cross_entropy_loss(&cpu_logits, &targets)
        .expect("CPU cross-entropy dispatch must succeed");
    let wgpu_loss = coeus_nn::cross_entropy_loss(&wgpu_logits, &targets)
        .expect("WGPU cross-entropy dispatch must succeed");
    cpu_loss
        .backward()
        .expect("CPU cross-entropy backward must succeed");
    wgpu_loss
        .backward()
        .expect("WGPU cross-entropy backward must succeed");

    let wgpu_loss = to_cpu(&wgpu_loss.tensor);
    let cpu_gradient = cpu_logits.grad().expect("tracked CPU logits gradient");
    let wgpu_gradient = to_cpu(&wgpu_logits.grad().expect("tracked WGPU logits gradient"));
    assert_parity(
        "cross-entropy loss",
        cpu_loss.tensor.as_slice(),
        wgpu_loss.as_slice(),
    );
    assert_parity(
        "cross-entropy logits gradient",
        cpu_gradient.as_slice(),
        wgpu_gradient.as_slice(),
    );

    let backend = super::wgpu();
    let logits_layout = Layout::new([3, 4].into());
    let scalar_layout = Layout::new([1].into());
    // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
    let mut loss =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
    let mut probabilities =
        unsafe { backend.allocate::<f32>(12) }.expect("invariant: test backend operation succeeds");
    let targets = backend
        .prepare_cross_entropy_targets(&targets)
        .expect("WGPU target preparation must succeed");

    let malformed_layout = Layout::new([12].into());
    let error = backend
        .cross_entropy_forward(
            wgpu_logits.tensor.storage(),
            &malformed_layout,
            &targets,
            &mut loss,
            &scalar_layout,
            &mut probabilities,
            &logits_layout,
        )
        .expect_err("rank-one logits must fail before provider dispatch");
    assert!(matches!(
        error,
        WgpuBackendError::Validation(BackendError::LayoutRankMismatch { lhs: 1, rhs: 2, .. })
    ));

    let short_targets = backend
        .prepare_cross_entropy_targets(&[0, 1])
        .expect("WGPU short target preparation must succeed");
    let error = backend
        .cross_entropy_forward(
            wgpu_logits.tensor.storage(),
            &logits_layout,
            &short_targets,
            &mut loss,
            &scalar_layout,
            &mut probabilities,
            &logits_layout,
        )
        .expect_err("provider-native target count must match the batch");
    assert!(matches!(
        error,
        WgpuBackendError::Validation(BackendError::ShapeMismatch { .. })
    ));

    let loss_parent = loss.clone();
    let probabilities_parent = probabilities.clone();
    backend
        .cross_entropy_forward(
            wgpu_logits.tensor.storage(),
            &logits_layout,
            &targets,
            &mut loss,
            &scalar_layout,
            &mut probabilities,
            &logits_layout,
        )
        .expect("direct WGPU forward must succeed");
    let mut parent_loss = [f32::NAN; 1];
    let mut parent_probabilities = [f32::NAN; 12];
    backend
        .copy_to_host(&loss_parent, &mut parent_loss)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_host(&probabilities_parent, &mut parent_probabilities)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(parent_loss, [0.0]);
    assert_eq!(parent_probabilities, [0.0; 12]);

    // SAFETY: the following `copy_to_device` initializes every element before any read.
    let mut output_gradient =
        unsafe { backend.allocate::<f32>(1) }.expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[1.0], &mut output_gradient)
        .expect("invariant: test backend operation succeeds");
    // SAFETY: the dispatched provider operation overwrites every logical output element before the buffer is read.
    let mut logit_gradient =
        unsafe { backend.allocate::<f32>(12) }.expect("invariant: test backend operation succeeds");
    backend
        .copy_to_device(&[0.25; 12], &mut logit_gradient)
        .expect("invariant: test backend operation succeeds");
    let gradient_parent = logit_gradient.clone();
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
        .expect("direct WGPU backward must succeed");
    let mut parent_gradient = [f32::NAN; 12];
    let mut accumulated_gradient = [f32::NAN; 12];
    backend
        .copy_to_host(&gradient_parent, &mut parent_gradient)
        .expect("invariant: test backend operation succeeds");
    backend
        .copy_to_host(&logit_gradient, &mut accumulated_gradient)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(parent_gradient, [0.25; 12]);
    let expected_accumulated = cpu_gradient
        .as_slice()
        .iter()
        .map(|gradient| gradient + 0.25)
        .collect::<Vec<_>>();
    assert_parity(
        "cross-entropy additive candidate gradient",
        &expected_accumulated,
        &accumulated_gradient,
    );
}
