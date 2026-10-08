//! The CTC seam reaches either backend from one call site.
//!
//! `coeus_ops::CtcOps` is implemented for the CPU backend over Leto and for
//! this one over Hephaestus. These tests call the same trait methods on
//! both and compare, which is the claim the seam exists to support: a
//! consumer binds the trait, not a device.

use coeus_core::{ComputeBackend, Layout, SequentialBackend};
use coeus_ops::{CtcBatch, CtcOps};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

const SHAPE: [usize; 3] = [6, 3, 5];
const INPUT_LENGTHS: [usize; 3] = [6, 5, 0];
const TARGET_LENGTHS: [usize; 3] = [2, 1, 0];
const TARGETS: [usize; 3] = [2, 2, 1];
const BLANK: usize = 0;
const UPSTREAM: f32 = 1.5;

fn layout_for(shape: &[usize]) -> Layout {
    Layout::new(shape.to_vec().into())
}

/// Deterministic nonpositive lanes, so the provider's input contract holds
/// on both backends without normalization.
fn log_probs(shape: [usize; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape.iter().product());
    for t in 0..shape[0] {
        for b in 0..shape[1] {
            for c in 0..shape[2] {
                let hash = (t * 13 + b * 7 + c * 3 + 11) % 97;
                values.push(-(hash as f32 * 0.05 + 0.01));
            }
        }
    }
    values
}

fn batch<'a>(
    targets: &'a [usize],
    input_lengths: &'a [usize],
    target_lengths: &'a [usize],
) -> CtcBatch<'a> {
    CtcBatch {
        targets,
        input_lengths,
        target_lengths,
        blank: BLANK,
    }
}

/// The kernels keep the provider's operation order, so the claim is a
/// 64-ulp bound over libm rounding rather than bitwise equality.
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let scale = expected.iter().fold(1.0_f32, |acc, v| acc.max(v.abs()));
    let bound = 64.0 * f32::EPSILON * scale;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= bound,
            "{what}: lane {index}: {actual} vs {expected}, bound {bound:e}"
        );
    }
}

fn through_both() -> ((f32, Vec<f32>), (f32, Vec<f32>)) {
    let sequential = SequentialBackend;
    let wgpu = WgpuBackend::new();
    let host = log_probs(SHAPE);
    let grid_layout = layout_for(&SHAPE);
    let scalar_layout = layout_for(&[1]);
    let cells: usize = SHAPE.iter().product();

    let input = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &host);
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);

    let mut cpu_loss_storage = sequential.allocate_zeroed::<f32>(1);
    let cpu_state = CtcOps::<f32>::ctc_forward(
        &sequential,
        input.storage(),
        input.layout(),
        batch(&TARGETS, &INPUT_LENGTHS, &TARGET_LENGTHS),
        &mut cpu_loss_storage,
        &scalar_layout,
    )
    .expect("sequential ctc forward");
    let mut cpu_grad_storage = sequential.allocate_zeroed::<f32>(cells);
    let upstream = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[UPSTREAM]);
    CtcOps::<f32>::ctc_backward_accumulate(
        &sequential,
        &cpu_state,
        upstream.storage(),
        upstream.layout(),
        &mut cpu_grad_storage,
        &grid_layout,
    )
    .expect("sequential ctc backward");
    let expected_loss =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_loss_storage, scalar_layout.clone())
            .as_slice()[0];
    let expected_grad =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_grad_storage, grid_layout.clone())
            .as_slice()
            .to_vec();

    let mut device_loss_storage = wgpu.allocate_zeroed::<f32>(1);
    let device_state = CtcOps::<f32>::ctc_forward(
        &wgpu,
        input_wgpu.storage(),
        input_wgpu.layout(),
        batch(&TARGETS, &INPUT_LENGTHS, &TARGET_LENGTHS),
        &mut device_loss_storage,
        &scalar_layout,
    )
    .expect("wgpu ctc forward");
    let mut device_grad_storage = wgpu.allocate_zeroed::<f32>(cells);
    let upstream_wgpu = upstream.to_backend_on(&sequential, &wgpu);
    CtcOps::<f32>::ctc_backward_accumulate(
        &wgpu,
        &device_state,
        upstream_wgpu.storage(),
        upstream_wgpu.layout(),
        &mut device_grad_storage,
        &grid_layout,
    )
    .expect("wgpu ctc backward");
    let actual_loss =
        Tensor::<f32, WgpuBackend>::from_raw_parts(device_loss_storage, scalar_layout)
            .to_backend_on(&wgpu, &sequential)
            .as_slice()[0];
    let actual_grad = Tensor::<f32, WgpuBackend>::from_raw_parts(device_grad_storage, grid_layout)
        .to_backend_on(&wgpu, &sequential)
        .as_slice()
        .to_vec();

    ((actual_loss, actual_grad), (expected_loss, expected_grad))
}

#[test]
fn wgpu_ctc_matches_sequential_forward_and_backward() {
    let ((actual_loss, actual_grad), (expected_loss, expected_grad)) = through_both();
    assert_close(&[actual_loss], &[expected_loss], "ctc loss");
    assert_close(&actual_grad, &expected_grad, "ctc gradient");
}

/// An impossible alignment — a nonempty target with zero valid frames —
/// writes positive infinity on both backends and rejects the backward
/// pass on both, before the gradient changes.
#[test]
fn wgpu_ctc_rejects_an_impossible_alignment_like_sequential() {
    let sequential = SequentialBackend;
    let wgpu = WgpuBackend::new();
    let shape = [4_usize, 2, 3];
    let host = log_probs(shape);
    let grid_layout = layout_for(&shape);
    let scalar_layout = layout_for(&[1]);
    let input = Tensor::<f32, SequentialBackend>::from_slice(shape.to_vec(), &host);
    let input_wgpu = input.to_backend_on(&sequential, &wgpu);
    let targets = [1_usize, 2];
    let input_lengths = [4_usize, 0];
    let target_lengths = [1_usize, 1];

    let mut cpu_loss = sequential.allocate_zeroed::<f32>(1);
    let cpu_state = CtcOps::<f32>::ctc_forward(
        &sequential,
        input.storage(),
        input.layout(),
        batch(&targets, &input_lengths, &target_lengths),
        &mut cpu_loss,
        &scalar_layout,
    )
    .expect("sequential ctc forward reports the impossible loss");
    let mut device_loss = wgpu.allocate_zeroed::<f32>(1);
    let device_state = CtcOps::<f32>::ctc_forward(
        &wgpu,
        input_wgpu.storage(),
        input_wgpu.layout(),
        batch(&targets, &input_lengths, &target_lengths),
        &mut device_loss,
        &scalar_layout,
    )
    .expect("wgpu ctc forward reports the impossible loss");
    let cpu_loss_value =
        Tensor::<f32, SequentialBackend>::from_raw_parts(cpu_loss, scalar_layout.clone())
            .as_slice()[0];
    let device_loss_value =
        Tensor::<f32, WgpuBackend>::from_raw_parts(device_loss, scalar_layout.clone())
            .to_backend_on(&wgpu, &sequential)
            .as_slice()[0];
    assert_eq!(cpu_loss_value, f32::INFINITY);
    assert_eq!(device_loss_value, f32::INFINITY);

    let upstream = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[UPSTREAM]);
    let upstream_wgpu = upstream.to_backend_on(&sequential, &wgpu);
    let cells: usize = shape.iter().product();
    let mut cpu_grad = sequential.allocate_zeroed::<f32>(cells);
    let mut device_grad = wgpu.allocate_zeroed::<f32>(cells);
    assert!(
        CtcOps::<f32>::ctc_backward_accumulate(
            &sequential,
            &cpu_state,
            upstream.storage(),
            upstream.layout(),
            &mut cpu_grad,
            &grid_layout,
        )
        .is_err(),
        "sequential ctc backward rejects the impossible alignment"
    );
    assert!(
        CtcOps::<f32>::ctc_backward_accumulate(
            &wgpu,
            &device_state,
            upstream_wgpu.storage(),
            upstream_wgpu.layout(),
            &mut device_grad,
            &grid_layout,
        )
        .is_err(),
        "wgpu ctc backward rejects the impossible alignment"
    );
}

/// WGPU dispatches with value and gradient parity through the tracked
/// autograd op: backward runs the posterior sweep on the device, so this
/// is the claim differentiation tracking exists to support.
#[test]
fn wgpu_ctc_tracking_matches_sequential_value_and_gradient() {
    use coeus_autograd::{ctc_loss, Var};
    let sequential = SequentialBackend;
    let wgpu = WgpuBackend::new();
    let host = log_probs(SHAPE);
    let input = Tensor::<f32, SequentialBackend>::from_slice(SHAPE.to_vec(), &host);
    let cpu_input = Var::new(input.clone(), true);
    let wgpu_input = Var::new(input.to_backend_on(&sequential, &wgpu), true);
    let cpu_loss = ctc_loss(&cpu_input, &TARGETS, &INPUT_LENGTHS, &TARGET_LENGTHS, BLANK)
        .expect("CPU ctc forward must succeed");
    let wgpu_loss = ctc_loss(
        &wgpu_input,
        &TARGETS,
        &INPUT_LENGTHS,
        &TARGET_LENGTHS,
        BLANK,
    )
    .expect("wgpu ctc forward must succeed");
    let wgpu_value = wgpu_loss.tensor.to_backend_on(&wgpu, &sequential);
    assert_close(
        wgpu_value.as_slice(),
        cpu_loss.tensor.as_slice(),
        "tracked ctc loss",
    );

    cpu_loss.backward().expect("CPU ctc backward must succeed");
    wgpu_loss
        .backward()
        .expect("wgpu ctc backward must succeed");
    let cpu_grad = cpu_input.grad().expect("CPU input tracks a gradient");
    let wgpu_grad = wgpu_input.grad().expect("WGPU input tracks a gradient");
    let wgpu_grad_cpu = wgpu_grad.to_backend_on(&wgpu, &sequential);
    assert_close(
        wgpu_grad_cpu.as_slice(),
        cpu_grad.as_slice(),
        "tracked ctc gradient",
    );
}
